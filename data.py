"""Data generation, feature engineering, and dataset preparation."""

import numpy as np
import pandas as pd                              # only for bdate_range
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler

from config import (SEED, WINDOWS,
                    N_TRAIN_SERIES, N_TEST_SERIES, N_SERIES,
                    N_TRAIN_DAYS, N_TEST_DAYS, WARMUP,
                    TRAIN_IDS, TEST_IDS)


# ── A-stock market parameter tables ──────────────────────────────
MARKET_PARAMS = {
    "main": {
        "daily_drift": 0.0002, "bull_drift": 0.0003, "bear_drift": -0.0002,
        "daily_vol": 0.0113,
        "omega": 5e-5, "beta": 0.88, "alpha": 0.10, "gamma": -0.06,
        "jump_prob": 0.008, "jump_mean": -0.008, "jump_std": 0.015,
        "limit": 0.10, "st_limit": 0.05,
    },
    "chinext": {
        "daily_drift": 0.0003, "bull_drift": 0.00045, "bear_drift": -0.0003,
        "daily_vol": 0.0189,
        "omega": 1.5e-4, "beta": 0.85, "alpha": 0.12, "gamma": -0.10,
        "jump_prob": 0.012, "jump_mean": -0.012, "jump_std": 0.020,
        "limit": 0.20, "st_limit": 0.20,
    },
    "star": {
        "daily_drift": 0.0004, "bull_drift": 0.0006, "bear_drift": -0.0004,
        "daily_vol": 0.0221,
        "omega": 2e-4, "beta": 0.82, "alpha": 0.15, "gamma": -0.13,
        "jump_prob": 0.015, "jump_mean": -0.015, "jump_std": 0.025,
        "limit": 0.20, "st_limit": 0.20,
    },
}

# Column layout for param tensor built from MARKET_PARAMS
# 0:daily_drift  1:bull_drift  2:bear_drift  3:daily_vol
# 4:omega  5:beta  6:alpha  7:gamma
# 8:jump_prob  9:jump_mean  10:jump_std  11:limit
_MT_ORDER = ["main", "chinext", "star"]


def _build_param_table(device):
    """(3, 12) tensor of market params."""
    rows = []
    for mt in _MT_ORDER:
        mp = MARKET_PARAMS[mt]
        rows.append([
            mp["daily_drift"], mp["bull_drift"], mp["bear_drift"], mp["daily_vol"],
            mp["omega"], mp["beta"], mp["alpha"], mp["gamma"],
            mp["jump_prob"], mp["jump_mean"], mp["jump_std"], mp["limit"],
        ])
    return torch.tensor(rows, device=device)


def generate_price_series_batch(n_days_list, market_types, annual_returns,
                                max_dd=0.9, seed=None, device="cpu"):
    """
    Batch-generate all price series in parallel on GPU.

    GJR-GARCH loop iterates over time but processes all N series
    simultaneously via vectorised tensor ops.
    Returns list of numpy price arrays.
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = len(n_days_list)
    max_days = max(n_days_list)

    # ── per-series parameters ──
    param_table = _build_param_table(device)                      # (3, 12)
    mt_idx = torch.tensor([_MT_ORDER.index(mt) for mt in market_types],
                          device=device)
    p = param_table[mt_idx]                                       # (N, 12)

    mu       = torch.tensor(annual_returns, device=device) / 252
    omega    = p[:, 4]
    beta     = p[:, 5]
    alpha    = p[:, 6]
    gamma    = p[:, 7]
    jmp_prob = p[:, 8]
    jmp_mean = p[:, 9]
    jmp_std  = p[:, 10]
    limit    = p[:, 11]
    bull_dr  = p[:, 1]
    bear_dr  = p[:, 2]
    base_vol = p[:, 3]

    # ── pre-generate all randomness ──
    z       = torch.randn(N, max_days, device=device)
    u_jump  = torch.rand(N, max_days, device=device)
    j_z     = torch.randn(N, max_days, device=device)
    u_state = torch.rand(N, max_days, device=device)
    u_mag   = torch.rand(N, max_days, device=device)

    # ── state machine vectors ──
    state     = torch.zeros(N, dtype=torch.long, device=device)
    state_age = torch.zeros(N, dtype=torch.long, device=device)
    vol_scales = torch.tensor([1.0, 0.8, 1.3], device=device)
    jm_adj     = torch.tensor([0.0, 0.003, -0.012], device=device)
    trans      = torch.tensor([[1, 2], [0, 2], [0, 1]],
                              dtype=torch.long, device=device)

    # ── GARCH init ──
    sigma2 = base_vol ** 2                                       # (N,)

    # ── length mask ──
    lengths = torch.tensor(n_days_list, dtype=torch.long, device=device)
    t_range = torch.arange(max_days, device=device)
    mask = t_range.unsqueeze(0) < lengths.unsqueeze(1)           # (N, max_days)

    log_rets = torch.zeros(N, max_days, device=device)

    for t in range(max_days):
        # ── L1: state machine ──
        state_age += 1
        switch = (state_age > 20) & (u_state[:, t] < 0.02)
        rand_pick = (u_state[:, t] < 0.01).long()
        new_state = trans[state, rand_pick]
        state     = torch.where(switch, new_state, state)
        state_age = torch.where(switch, torch.zeros_like(state_age), state_age)

        # state-dependent params
        cur_drift = torch.where(state == 1, bull_dr,
                    torch.where(state == 2, bear_dr, mu))
        cur_vs    = vol_scales[state]
        cur_jm    = jmp_mean + jm_adj[state]

        # ── L3: base shock ──
        r_base = cur_drift + sigma2.sqrt() * cur_vs * z[:, t]

        # ── L4: jump injection ──
        is_jump = u_jump[:, t] < jmp_prob
        r_raw   = torch.where(is_jump,
                              r_base + cur_jm + jmp_std * j_z[:, t],
                              r_base)

        # ── L5: limit + magnetic effect ──
        mag_up   = (r_raw >  limit * 0.80) & (u_mag[:, t] < 0.70)
        mag_down = (r_raw < -limit * 0.80) & (u_mag[:, t] < 0.70)
        r_eff    = torch.where(mag_up,  limit,
                  torch.where(mag_down, -limit,
                  torch.clamp(r_raw, -limit, limit)))

        # mask out padded timesteps
        t_mask   = mask[:, t]
        r_masked = torch.where(t_mask, r_eff, torch.zeros_like(r_eff))
        log_rets[:, t] = torch.where(t_mask,
                                    torch.log(1.0 + r_masked),
                                    torch.zeros(N, device=device))

        # ── L2: GJR-GARCH (feeds back clipped return) ──
        eps2   = r_masked ** 2
        neg    = (r_masked < 0).float()
        sigma2 = omega + beta * sigma2 + alpha * eps2 + gamma * eps2 * neg

    # ── L6: price synthesis ──
    log_price = log_rets.cumsum(dim=1)                            # (N, max_days)

    # drawdown floor
    log_peak = torch.cummax(log_price, dim=1).values
    log_price = torch.maximum(log_price,
                              log_peak + np.log(1 - max_dd))

    prices = torch.exp(log_price) * 100

    # trim to actual lengths → list of numpy arrays
    prices_np  = prices.cpu().numpy()
    lengths_np = lengths.cpu().numpy()
    return [prices_np[i, :lengths_np[i]] for i in range(N)]


def compute_mmn_features(close, series_id):
    """Compute MMn diff features for a single price series using polars."""
    log_c = np.log(close)

    feat = pl.DataFrame({"log_c": log_c})
    exprs = []
    for w in WINDOWS:
        exprs.append(pl.col("log_c").rolling_min(window_size=w, min_periods=w)
                     .diff().alias(f"d_mm_min_{w}"))
        exprs.append(pl.col("log_c").rolling_max(window_size=w, min_periods=w)
                     .diff().alias(f"d_mm_max_{w}"))
    feat = feat.with_columns(exprs).drop("log_c")

    daily_ret = np.empty(len(close))
    daily_ret[0] = np.nan
    daily_ret[1:] = close[1:] / close[:-1] - 1

    return feat.with_columns([
        pl.lit(series_id).cast(pl.Int64).alias("series_id"),
        pl.Series("close", close),
        pl.Series("daily_return", daily_ret),
    ]).drop_nulls()


def prepare_datasets(device):
    """
    Generate synthetic data, compute features, split, scale.
    - 每条序列随机分配市场类型（主板/创业板/科创板）
    - 训练序列生成 N_TRAIN_DAYS 天，测试序列生成 N_TEST_DAYS 天
    Returns dict with GPU tensors ready for training.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    param_rng = np.random.RandomState(SEED)
    market_type_list   = []
    annual_return_list = []
    n_days_list        = []

    mt_choices = ["main", "chinext", "star"]
    mt_weights = [0.60, 0.25, 0.15]

    for i in range(N_SERIES):
        mt = param_rng.choice(mt_choices, p=mt_weights)
        ar = param_rng.uniform(-0.05, 0.25)
        nd = N_TRAIN_DAYS if i in TRAIN_IDS else N_TEST_DAYS
        market_type_list.append(mt)
        annual_return_list.append(ar)
        n_days_list.append(nd)
        print(f"\r  [{i+1}/{N_SERIES}] Series {i} ({'train' if i in TRAIN_IDS else 'test':>5}): "
              f"days={nd}  market={mt:<7}  return={ar:+.1%}", end="", flush=True)
    print()

    # batch generate all series on GPU
    print(f"  Generating {N_SERIES} series on GPU...")
    prices_arrays = generate_price_series_batch(
        n_days_list, market_type_list, annual_return_list,
        seed=SEED, device=device)
    print(f"  Done.")

    # build DataFrames (keep pandas for bdate_range)
    print(f"  Building DataFrames...")
    prices_list = [
        pl.DataFrame({
            "date":   pd.bdate_range("2016-01-01", periods=len(prices_arrays[i])).to_numpy(),
            "close":  prices_arrays[i],
            "series_id": i,
        })
        for i in range(N_SERIES)]

    print(f"  Computing features ({N_SERIES} series)...")
    feat_list = []
    for i in range(N_SERIES):
        feat_list.append(compute_mmn_features(prices_arrays[i], i))
        if (i + 1) % 500 == 0 or i == N_SERIES - 1:
            print(f"\r  Features: {i+1}/{N_SERIES}", end="", flush=True)
    print()

    df_feat = pl.concat(feat_list, how="vertical")
    feat_cols = [f"d_mm_min_{w}" for w in WINDOWS] + [f"d_mm_max_{w}" for w in WINDOWS]

    print(f"  Fitting scaler...")
    train_df = df_feat.filter(pl.col("series_id").is_in(TRAIN_IDS))
    test_df  = df_feat.filter(pl.col("series_id").is_in(TEST_IDS))

    scaler = StandardScaler()
    scaler.fit(train_df.select(feat_cols).to_numpy())

    print(f"  Uploading to GPU...")
    # sort + batch transform + numpy slice (avoids per-series filter)
    train_sorted = train_df.sort("series_id")
    train_X = scaler.transform(train_sorted.select(feat_cols).to_numpy())
    train_rets_all = train_sorted["daily_return"].to_numpy()
    train_sids = train_sorted["series_id"].to_numpy()

    unique_sids, starts = np.unique(train_sids, return_index=True)
    sid_to_idx = {s: i for i, s in enumerate(unique_sids)}
    ends = np.concatenate([starts[1:], [len(train_sids)]])

    train_feats, train_rets = [], []
    for sid in TRAIN_IDS:
        j = sid_to_idx[sid]
        train_feats.append(torch.FloatTensor(train_X[starts[j]:ends[j]]).to(device))
        train_rets.append(torch.FloatTensor(train_rets_all[starts[j]:ends[j]]).to(device))

    test_sorted = test_df.sort("series_id")
    test_X = scaler.transform(test_sorted.select(feat_cols).to_numpy())
    test_rets_all = test_sorted["daily_return"].to_numpy()
    test_sids = test_sorted["series_id"].to_numpy()

    unique_tsids, tstarts = np.unique(test_sids, return_index=True)
    tsid_to_idx = {s: i for i, s in enumerate(unique_tsids)}
    tends = np.concatenate([tstarts[1:], [len(test_sids)]])

    test_feats, test_rets = [], []
    for sid in TEST_IDS:
        j = tsid_to_idx[sid]
        test_feats.append(torch.FloatTensor(test_X[tstarts[j]:tends[j]]).to(device))
        test_rets.append(torch.FloatTensor(test_rets_all[tstarts[j]:tends[j]]).to(device))

    return {
        "prices_list":  prices_list,
        "feat_cols":    feat_cols,
        "scaler":       scaler,
        "train_feats":  train_feats,
        "train_rets":   train_rets,
        "test_feats":   test_feats,
        "test_rets":    test_rets,
    }