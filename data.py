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


def _generate_series_block(n_series, n_days, param_rng, seed, device,
                           mt_choices, mt_weights):
    """Generate, compute features, and return (prices_list, feat_df, params_info)."""
    market_types, annual_returns, n_days_list = [], [], []
    for i in range(n_series):
        mt = param_rng.choice(mt_choices, p=mt_weights)
        ar = param_rng.uniform(-0.05, 0.25)
        market_types.append(mt)
        annual_returns.append(ar)
        n_days_list.append(n_days)

    prices_arrays = generate_price_series_batch(
        n_days_list, market_types, annual_returns,
        seed=seed, device=device)

    prices_list = [
        pl.DataFrame({
            "date":   pd.bdate_range("2016-01-01", periods=len(prices_arrays[i])).to_numpy(),
            "close":  prices_arrays[i],
        })
        for i in range(n_series)]

    feat_list = [compute_mmn_features(prices_arrays[i], i) for i in range(n_series)]
    return prices_list, pl.concat(feat_list, how="vertical"), market_types, annual_returns


def _df_to_tensors(df, feat_cols, scaler, series_ids, device):
    """Sort by series_id, batch-transform, split into per-series GPU tensors."""
    sorted_df = df.sort("series_id")
    X = scaler.transform(sorted_df.select(feat_cols).to_numpy())
    rets = sorted_df["daily_return"].to_numpy()
    sids = sorted_df["series_id"].to_numpy()

    unique_sids, starts = np.unique(sids, return_index=True)
    sid_map = {s: i for i, s in enumerate(unique_sids)}
    ends = np.concatenate([starts[1:], [len(sids)]])

    feats_out, rets_out = [], []
    for sid in series_ids:
        j = sid_map[sid]
        feats_out.append(torch.FloatTensor(X[starts[j]:ends[j]]).to(device))
        rets_out.append(torch.FloatTensor(rets[starts[j]:ends[j]]).to(device))
    return feats_out, rets_out


def prepare_datasets(device, seed=None):
    """
    Generate training data only.
    Returns dict with scaler, feat_cols, and per-series GPU tensors.
    """
    seed = seed if seed is not None else SEED
    np.random.seed(seed)
    torch.manual_seed(seed)

    param_rng = np.random.RandomState(seed)
    mt_choices = ["main", "chinext", "star"]
    mt_weights = [0.60, 0.25, 0.15]

    print(f"  Generating {N_TRAIN_SERIES} train series (GPU, seed={seed})...")
    prices_list, df_feat, mt_list, ar_list = _generate_series_block(
        N_TRAIN_SERIES, N_TRAIN_DAYS, param_rng, seed, device,
        mt_choices, mt_weights)
    print(f"  Done. {len(df_feat)} samples.")

    feat_cols = [f"d_mm_min_{w}" for w in WINDOWS] + [f"d_mm_max_{w}" for w in WINDOWS]
    print(f"  Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(df_feat.select(feat_cols).to_numpy())

    print(f"  Uploading to GPU...")
    train_ids = list(range(N_TRAIN_SERIES))
    train_feats, train_rets = _df_to_tensors(
        df_feat, feat_cols, scaler, train_ids, device)

    return {
        "feat_cols":    feat_cols,
        "scaler":       scaler,
        "train_feats":  train_feats,
        "train_rets":   train_rets,
    }


def prepare_test_data(device, scaler, feat_cols):
    """
    Generate test data using a pre-fitted scaler (rank 0 only).
    Returns dict with prices_list and per-series GPU tensors.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    param_rng = np.random.RandomState(SEED)
    mt_choices = ["main", "chinext", "star"]
    mt_weights = [0.60, 0.25, 0.15]

    print(f"  Generating {N_TEST_SERIES} test series (GPU)...")
    prices_list, df_feat, _, _ = _generate_series_block(
        N_TEST_SERIES, N_TEST_DAYS, param_rng, SEED, device,
        mt_choices, mt_weights)
    print(f"  Done. {len(df_feat)} samples.")

    test_ids = list(range(N_TEST_SERIES))
    test_feats, test_rets = _df_to_tensors(
        df_feat, feat_cols, scaler, test_ids, device)

    return {
        "prices_list":  prices_list,
        "test_feats":   test_feats,
        "test_rets":    test_rets,
    }