"""Data generation, feature engineering, and dataset preparation."""

import numpy as np
import pandas as pd
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


def generate_price_series(n_days, market_type="main", annual_return=None,
                          max_dd=0.9, seed=None):
    """
    6-layer A-stock price generator:
      L1  market state machine  (bull / bear / range)
      L2  GJR-GARCH volatility  (leverage effect)
      L3  base shock
      L4  jump injection        (fat tail + negative skew)
      L5  limit constraints     (+ magnetic effect)
      L6  price synthesis
    """
    if seed is not None:
        np.random.seed(seed)

    p = MARKET_PARAMS[market_type]
    mu = annual_return / 252 if annual_return is not None else p["daily_drift"]

    # state-dependent: drift, vol scale, jump mean
    drifts     = [mu,              p["bull_drift"],     p["bear_drift"]]
    vol_scales = [1.0,             0.8,                  1.3]
    jump_means = [p["jump_mean"],  p["jump_mean"]+0.003, p["jump_mean"]-0.012]

    # pre-generate all randomness
    z       = np.random.normal(0, 1, n_days)
    u_jump  = np.random.uniform(0, 1, n_days)
    j_z     = np.random.normal(0, 1, n_days)
    u_state = np.random.uniform(0, 1, n_days)
    u_mag   = np.random.uniform(0, 1, n_days)

    sigma2    = np.empty(n_days)
    sigma2[0] = p["daily_vol"] ** 2
    log_rets  = np.empty(n_days)

    state, state_age = 0, 0          # 0=range  1=bull  2=bear

    for t in range(n_days):
        # ── L1: state machine ──
        state_age += 1
        if state_age > 20 and u_state[t] < 0.02:
            others = [s for s in (0, 1, 2) if s != state]
            state = others[0] if u_state[t] < 0.01 else others[1]
            state_age = 0

        # ── L3: base shock ──
        r_base = drifts[state] + np.sqrt(sigma2[t]) * vol_scales[state] * z[t]

        # ── L4: jump injection ──
        if u_jump[t] < p["jump_prob"]:
            r_raw = r_base + jump_means[state] + p["jump_std"] * j_z[t]
        else:
            r_raw = r_base

        # ── L5: limit + magnetic effect ──
        lim = p["limit"]
        if r_raw > lim * 0.80 and u_mag[t] < 0.70:
            r_eff = lim
        elif r_raw < -lim * 0.80 and u_mag[t] < 0.70:
            r_eff = -lim
        else:
            r_eff = max(-lim, min(lim, r_raw))

        log_rets[t] = np.log(1.0 + r_eff)

        # ── L2: GJR-GARCH (feeds back clipped return) ──
        if t < n_days - 1:
            eps2 = r_eff * r_eff
            neg  = 1.0 if r_eff < 0 else 0.0
            sigma2[t + 1] = (p["omega"]
                             + p["beta"]  * sigma2[t]
                             + p["alpha"] * eps2
                             + p["gamma"] * eps2 * neg)

    # ── L6: price synthesis ──
    log_price = np.cumsum(log_rets)

    # drawdown floor
    log_peak = np.maximum.accumulate(log_price)
    log_price = np.maximum(log_price, log_peak + np.log(1 - max_dd))

    return np.exp(log_price) * 100


def compute_mmn_features(df):
    log_c = np.log(df["close"].values)
    feat = {}
    for w in WINDOWS:
        s = pd.Series(log_c)
        mm_min = s.rolling(w, min_periods=w).min()
        mm_max = s.rolling(w, min_periods=w).max()
        feat[f"d_mm_min_{w}"] = mm_min.diff()
        feat[f"d_mm_max_{w}"] = mm_max.diff()
    feat = pd.DataFrame(feat)
    feat["series_id"] = df["series_id"].values
    feat["close"]      = df["close"].values
    feat["daily_return"] = df["close"].pct_change()
    return feat.dropna().reset_index(drop=True)


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
    prices_list = []
    market_types = ["main", "chinext", "star"]
    market_weights = [0.60, 0.25, 0.15]

    for i in range(N_SERIES):
        market_type   = param_rng.choice(market_types, p=market_weights)
        annual_return = param_rng.uniform(-0.05, 0.25)
        n_days = N_TRAIN_DAYS if i in TRAIN_IDS else N_TEST_DAYS
        p = generate_price_series(
            n_days=n_days, seed=SEED+i,
            market_type=market_type, annual_return=annual_return)
        df = pd.DataFrame({
            "date": pd.date_range("2016-01-01", periods=len(p), freq="B"),
            "close": p, "series_id": i})
        prices_list.append(df)
        print(f"  Series {i} ({'train' if i in TRAIN_IDS else 'test':>5}): "
              f"days={n_days}  market={market_type:<7}  "
              f"return={annual_return:+.1%}")

    feat_list = [compute_mmn_features(prices_list[i]) for i in range(N_SERIES)]
    df_feat = pd.concat(feat_list, ignore_index=True)
    feat_cols = [f"d_mm_min_{w}" for w in WINDOWS] + [f"d_mm_max_{w}" for w in WINDOWS]

    # fit scaler on all training series combined
    train_df = df_feat[df_feat["series_id"].isin(TRAIN_IDS)].reset_index(drop=True)
    test_df  = df_feat[df_feat["series_id"].isin(TEST_IDS)].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols].values)

    # per-series GPU tensors
    train_feats, train_rets = [], []
    for sid in TRAIN_IDS:
        m = train_df["series_id"] == sid
        X = scaler.transform(train_df.loc[m, feat_cols].values)
        train_feats.append(torch.FloatTensor(X).to(device))
        train_rets.append(torch.FloatTensor(train_df.loc[m, "daily_return"].values).to(device))

    test_feats, test_rets = [], []
    for sid in TEST_IDS:
        m = test_df["series_id"] == sid
        X = scaler.transform(test_df.loc[m, feat_cols].values)
        test_feats.append(torch.FloatTensor(X).to(device))
        test_rets.append(torch.FloatTensor(test_df.loc[m, "daily_return"].values).to(device))

    return {
        "prices_list":  prices_list,
        "feat_cols":    feat_cols,
        "scaler":       scaler,
        "train_feats":  train_feats,
        "train_rets":   train_rets,
        "test_feats":   test_feats,
        "test_rets":    test_rets,
    }