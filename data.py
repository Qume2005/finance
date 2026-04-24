"""Data generation, feature engineering, and dataset preparation."""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import (SEED, WINDOWS,
                    N_TRAIN_SERIES, N_TEST_SERIES, N_SERIES,
                    N_TRAIN_DAYS, N_TEST_DAYS, WARMUP,
                    TRAIN_IDS, TEST_IDS)


def generate_price_series(n_days, annual_return=0.08,
                          daily_vol=0.06, theta_ou=0.08,
                          regime_switch=0.02, jump_freq=20,
                          max_dd=0.9, seed=None):
    if seed is not None:
        np.random.seed(seed)
    daily_drift = annual_return / 252
    log_trend = np.cumsum(np.full(n_days, daily_drift))
    # random walk (double vol)
    noise_rw = np.cumsum(np.random.normal(0, daily_vol, n_days))
    # mean-reverting (OU)
    sigma_ou = daily_vol * 0.7
    noise_ou = np.zeros(n_days)
    for t in range(1, n_days):
        noise_ou[t] = noise_ou[t-1]*(1-theta_ou) + np.random.normal(0, sigma_ou)
    # regime-switching volatility clusters
    regime = np.zeros(n_days)
    r = 0
    for t in range(n_days):
        if np.random.rand() < regime_switch:
            r = 1 - r  # switch regime
        regime[t] = r
    vol_cluster = np.random.normal(0, 1, n_days) * daily_vol * (1 + 2 * regime)
    vol_cluster = np.cumsum(vol_cluster * 0.3)
    # jumps
    n_jumps = max(n_days // jump_freq, 1)
    jt = np.random.choice(n_days, n_jumps, replace=False)
    noise_jump = np.zeros(n_days)
    noise_jump[jt] = np.random.normal(0, daily_vol*5, n_jumps)
    # GARCH-like volatility feedback
    garch = np.zeros(n_days)
    garch[0] = daily_vol
    omega, alpha_g, beta_g = 1e-5, 0.15, 0.8
    for t in range(1, n_days):
        garch[t] = np.sqrt(omega + alpha_g * noise_rw[t-1]**2 + beta_g * garch[t-1]**2)
    noise_garch = np.random.normal(0, 1, n_days) * garch
    log_price = log_trend + noise_rw + noise_ou + noise_jump + vol_cluster + noise_garch
    # A-stock ±10% limit
    lr = np.diff(log_price)
    lr = np.clip(lr, np.log(0.9), np.log(1.1))
    log_price = np.concatenate([[log_price[0]], np.cumsum(lr)])
    # drawdown 上限：价格不低于历史高点的 (1 - max_dd)
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
    - 训练序列生成 N_TRAIN_DAYS 天，测试序列生成 N_TEST_DAYS 天
    - 每条序列去掉 WARMUP 行后取全部样本
    Returns dict with GPU tensors ready for training.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    param_rng = np.random.RandomState(SEED)
    prices_list = []

    for i in range(N_SERIES):
        annual_return = param_rng.uniform(-0.05, 0.25)    # 漂移: -5% ~ +25%
        daily_vol     = param_rng.uniform(0.03, 0.10)     # 波动率: 3% ~ 10%
        theta_ou      = param_rng.uniform(0.02, 0.15)     # OU 回归速度
        regime_switch = param_rng.uniform(0.01, 0.05)     # 制度切换概率
        jump_freq     = param_rng.randint(10, 40)         # 跳跃频率 (1/N天)
        n_days = N_TRAIN_DAYS if i in TRAIN_IDS else N_TEST_DAYS
        p = generate_price_series(
            n_days=n_days, seed=SEED+i,
            annual_return=annual_return, daily_vol=daily_vol,
            theta_ou=theta_ou, regime_switch=regime_switch,
            jump_freq=jump_freq)
        df = pd.DataFrame({
            "date": pd.date_range("2016-01-01", periods=len(p), freq="B"),
            "close": p, "series_id": i})
        prices_list.append(df)
        print(f"  Series {i} ({'train' if i in TRAIN_IDS else 'test':>5}): "
              f"days={n_days}  return={annual_return:+.1%}  vol={daily_vol:.1%}  "
              f"theta_ou={theta_ou:.3f}  regime_p={regime_switch:.3f}  "
              f"jump_freq=1/{jump_freq}")

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
