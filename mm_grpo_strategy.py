#!/usr/bin/env python3
"""
MMn Differential Features + Mini-KDA + GRPO Position Control
=============================================================
Model : Mini Kimi Delta Attention (KDA) — adapted from Kimi Linear (arXiv 2510.26692)
  Core: delta rule with channel-wise gating for sequential memory
Train : GRPO (Group Relative Policy Optimization) — from DeepSeekMath (arXiv 2402.03300)
  Core: no critic model, group-relative advantage estimation
Features: MMn differentials for n ∈ {5,10,20,30,50,100,200} (14 dims)
Action  : position sizing 0–10 (11 discrete levels)
Reward  : total_log_return − λ · max_drawdown

Usage:
  Single GPU:  python mm_grpo_strategy.py
  Multi-GPU:   torchrun --nproc_per_node=N mm_grpo_strategy.py
"""

# ─────────────────────────── Imports ───────────────────────────
import os, sys, time, math, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions import Categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ──────────────────────── Constants ────────────────────────────
SEED = 42
WINDOWS = [5, 10, 20, 30, 50, 100, 200]
N_TRAIN  = 7000
N_TEST   = 3000
N_DAYS   = N_TRAIN + N_TEST + 250
N_SERIES = 2

EPISODE_LEN = 200
G_SAMPLES   = 24
LAMBDA_DD   = 2.0
EPS_CLIP    = 0.2
BETA_KL     = 0.04
N_EPISODES  = 800
LR          = 5e-4


# ────────────────── Data & Feature Functions ───────────────────
def generate_price_series(n_days=N_DAYS, annual_return=0.08,
                          daily_vol=0.06, seed=None):
    if seed is not None:
        np.random.seed(seed)
    daily_drift = annual_return / 252
    log_trend = np.cumsum(np.full(n_days, daily_drift))
    # random walk (double vol)
    noise_rw = np.cumsum(np.random.normal(0, daily_vol, n_days))
    # mean-reverting (OU) — faster, stronger
    theta, sigma_ou = 0.08, daily_vol * 0.7
    noise_ou = np.zeros(n_days)
    for t in range(1, n_days):
        noise_ou[t] = noise_ou[t-1]*(1-theta) + np.random.normal(0, sigma_ou)
    # regime-switching volatility clusters
    regime = np.zeros(n_days)
    r = 0
    for t in range(n_days):
        if np.random.rand() < 0.02:
            r = 1 - r  # switch regime
        regime[t] = r
    vol_cluster = np.random.normal(0, 1, n_days) * daily_vol * (1 + 2 * regime)
    vol_cluster = np.cumsum(vol_cluster * 0.3)
    # jumps — more frequent, bigger
    n_jumps = n_days // 20
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


def compute_rewards(positions, daily_returns, lam=LAMBDA_DD):
    """
    positions:      (G, T) LongTensor 0-10
    daily_returns:  (T,)   FloatTensor
    returns:        (G,)   reward per trajectory
    """
    w = positions.float() / 10.0                       # (G, T)
    pnl = w * daily_returns.unsqueeze(0)               # (G, T)
    equity = torch.cumprod(1.0 + pnl, dim=1)           # (G, T)
    peak = torch.cummax(equity, dim=1).values
    dd = (peak - equity) / (peak + 1e-8)
    max_dd = dd.max(dim=1).values
    log_ret = torch.log(equity[:, -1] + 1e-8)
    return log_ret - lam * max_dd


# ────────────────── DDP Utilities ──────────────────────────────
def setup_distributed():
    """Initialize DDP. Returns (rank, world_size, local_rank, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ────────────── Step 4: Mini-KDA Policy Network ──────────────
class MiniKDALayer(nn.Module):
    """
    Mini Kimi Delta Attention (from Kimi Linear, arXiv 2510.26692).

    Recurrence (delta rule + channel-wise gate):
        S_t = (I − β_t k_t k_t^⊤) Diag(α_t) S_{t−1} + β_t k_t v_t^⊤
        o_t = S_t^⊤ q_t

    α_t : per-dimension forgetting (fine-grained gating)
    β_t : update strength
    """
    def __init__(self, d_input, d_key=16, d_value=16):
        super().__init__()
        self.dk, self.dv = d_key, d_value
        self.dk_pope = d_key * 2  # PoPE doubles the effective dimension
        self.theta_base = 10000.0
        self.W_q = nn.Linear(d_input, d_key, bias=False)
        self.W_k = nn.Linear(d_input, d_key, bias=False)
        self.W_v = nn.Linear(d_input, d_value, bias=False)
        # PoPE learnable phase bias δ_c (initialized to 0, constrained via sigmoid in forward)
        self.pope_delta_raw = nn.Parameter(torch.zeros(d_key))
        # channel-wise gate α — output matches dk_pope
        self.W_alpha = nn.Sequential(
            nn.Linear(d_input, d_key * 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_key * 2, self.dk_pope, bias=False),
        )
        # update gate β
        self.W_beta = nn.Linear(d_input, 1, bias=False)
        # post-KDA: Norm → gate → linear (+ residual)
        self.post_norm  = nn.RMSNorm(d_value)
        self.out_gate   = nn.Linear(d_input, d_value, bias=False)
        self.W_out      = nn.Linear(d_value, d_input, bias=False)
        # SwiGLU FFN (+ residual)
        self.ffn_norm   = nn.RMSNorm(d_input)
        self.ffn_gate   = nn.Linear(d_input, d_input, bias=False)
        self.ffn_up     = nn.Linear(d_input, d_input, bias=False)
        self.ffn_down   = nn.Linear(d_input, d_input, bias=False)

    def _apply_pope(self, x, positions, is_query):
        """
        PoPE: magnitude = softplus(content), phase = position × frequency.
        Returns (T, 2*dk) with [real, imag] concatenation.
        """
        mu = F.softplus(x)                                                  # (T, dk) content magnitude
        freqs = self.theta_base ** (torch.arange(self.dk, device=x.device).float() / self.dk)  # (dk,)
        phi = positions.unsqueeze(1) * freqs.unsqueeze(0)                   # (T, dk)
        if not is_query:
            # Learnable bias δ_c ∈ [-2π, 0] via sigmoid parameterization
            phi = phi + (-2 * math.pi * torch.sigmoid(self.pope_delta_raw))  # (T, dk)
        real = mu * torch.cos(phi)                                          # (T, dk)
        imag = mu * torch.sin(phi)                                          # (T, dk)
        return torch.cat([real, imag], dim=-1)                              # (T, 2*dk)

    def forward(self, x_seq):
        """x_seq: (T, d_input) → (T, d_input)"""
        T, d = x_seq.shape
        positions = torch.arange(T, device=x_seq.device)
        # PoPE-transformed q and k: content (magnitude) and position (phase) decoupled
        q = self._apply_pope(self.W_q(x_seq), positions, is_query=True)    # (T, 2*dk)
        k = self._apply_pope(self.W_k(x_seq), positions, is_query=False)   # (T, 2*dk)
        v = F.silu(self.W_v(x_seq))                                        # (T, dv)
        alpha = torch.sigmoid(self.W_alpha(x_seq))                         # (T, 2*dk)
        beta  = torch.sigmoid(self.W_beta(x_seq))                          # (T, 1)

        S = x_seq.new_zeros(self.dk_pope, self.dv)
        outputs = []
        for t in range(T):
            qt, kt, vt = q[t], k[t], v[t]
            at, bt = alpha[t], beta[t]
            # Delta rule: S = Diag(α)*S − β*k*(k^T Diag(α)*S) + β*k*v^T
            aS = at.unsqueeze(1) * S                    # (2*dk, dv)
            ktaS = (kt.unsqueeze(0) @ aS)               # (1, dv)
            S = aS - bt * kt.unsqueeze(1) * ktaS + bt * kt.unsqueeze(1) * vt.unsqueeze(0)
            ot = S.T @ qt                                # (dv,)
            outputs.append(ot)
        out = torch.stack(outputs)                       # (T, dv)

        # Norm → ×gate → linear + residual
        out = self.post_norm(out)                                  # (T, dv)
        out = out * torch.sigmoid(self.out_gate(x_seq))            # gate from input
        x_seq = x_seq + self.W_out(out)                            # residual

        # SwiGLU FFN + residual
        h = self.ffn_norm(x_seq)
        x_seq = x_seq + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x_seq


class KDAPolicyNetwork(nn.Module):
    """
    Stacked Mini-KDA with Attention Residuals (AttnRes, arXiv 2603.15031).
    Replaces standard residual accumulation with softmax attention over depth.
    Input : (T, 14)  MMn diff features
    Output: (T, 11)  logits for positions 0-10
    """
    def __init__(self, d_input=14, d_hidden=32, n_actions=11, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.inp_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.RMSNorm(d_hidden))
        self.kda_layers = nn.ModuleList([
            MiniKDALayer(d_hidden, d_key=16, d_value=16)
            for _ in range(n_layers)])
        # AttnRes: one pseudo-query w_l ∈ R^d per layer + one for output
        # Initialized to zero so initial attention weights are uniform (per paper)
        self.attn_res_norm = nn.RMSNorm(d_hidden)
        self.w = nn.ParameterList([
            nn.Parameter(torch.zeros(d_hidden))
            for _ in range(n_layers + 1)])
        # SwiGLU policy head
        self.head_gate = nn.Linear(d_hidden, d_hidden, bias=False)
        self.head_up   = nn.Linear(d_hidden, d_hidden, bias=False)
        self.head_down = nn.Linear(d_hidden, n_actions, bias=False)

    def _attn_res(self, outputs, w_l):
        """
        AttnRes aggregation: h_l = Σ softmax(w_l · RMSNorm(v_i)) · v_i
        outputs: list of (T, d) tensors from previous layers
        w_l:     (d,) learnable pseudo-query for this layer
        """
        V = torch.stack(outputs)                          # (L_prev, T, d)
        K = self.attn_res_norm(V)                         # (L_prev, T, d)
        logits = torch.einsum('d,ltd->lt', w_l, K)       # (L_prev, T)
        alpha = logits.softmax(0)                         # (L_prev, T)
        return torch.einsum('lt,ltd->td', alpha, V)      # (T, d)

    def forward(self, x):
        v = [self.inp_proj(x)]                            # v_0 = embedding
        for l in range(self.n_layers):
            h_l = self._attn_res(v, self.w[l]) if l > 0 else v[0]
            v.append(self.kda_layers[l](h_l))
        # Final AttnRes over all layer outputs
        h_out = self._attn_res(v, self.w[self.n_layers])
        return self.head_down(F.silu(self.head_gate(h_out)) * self.head_up(h_out))


# ────────────────────────── Main ───────────────────────────────
def main():
    rank, world_size, local_rank, device = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"Device: {device}  |  World size: {world_size}")

    # ────────────── Step 1: Simulate Price Data ──────────────
    # Same seed on all ranks → identical data (no broadcast needed)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    prices_list = []
    for i in range(N_SERIES):
        p = generate_price_series(seed=SEED+i)
        df = pd.DataFrame({
            "date": pd.date_range("2016-01-01", periods=len(p), freq="B"),
            "close": p, "series_id": i})
        prices_list.append(df)

    if is_main:
        print(f"Generated {N_SERIES} series × {N_DAYS} days")
        fig, axes = plt.subplots(N_SERIES, 1, figsize=(14, 2.5*N_SERIES), sharex=True)
        for i in range(N_SERIES):
            s = prices_list[i]
            axes[i].plot(s["date"], s["close"], lw=.5)
            axes[i].set_ylabel(f"Series {i}")
        plt.suptitle("Synthetic A-Stock Price Series", y=1.01)
        plt.tight_layout()
        plt.savefig("fig_01_prices.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved fig_01_prices.png")

    # ────────────── Step 2: Feature Engineering — MMn Diffs ──────────────
    feat_list = [compute_mmn_features(prices_list[i]) for i in range(N_SERIES)]
    df_feat = pd.concat(feat_list, ignore_index=True)
    feat_cols = [f"d_mm_min_{w}" for w in WINDOWS] + [f"d_mm_max_{w}" for w in WINDOWS]

    if is_main:
        print(f"Features ({len(feat_cols)}): {feat_cols}")
        print(f"Total samples after warm-up: {len(df_feat)}")

    # ────────────── Step 3: Train / Test Split + Scale ──────────────
    TRAIN_IDS = [0]
    TEST_IDS  = [1]

    train_df = df_feat[df_feat["series_id"].isin(TRAIN_IDS)].head(N_TRAIN).reset_index(drop=True)
    test_df  = df_feat[df_feat["series_id"].isin(TEST_IDS)].head(N_TEST).reset_index(drop=True)

    scaler = StandardScaler()
    train_X_all = scaler.fit_transform(train_df[feat_cols].values)
    test_X_all  = scaler.transform(test_df[feat_cols].values)

    # keep per-series arrays for episode sampling
    train_feats_by_series, train_rets_by_series = [], []
    for sid in TRAIN_IDS:
        m = train_df["series_id"] == sid
        train_feats_by_series.append(torch.FloatTensor(train_X_all[m]).to(device))
        train_rets_by_series.append(torch.FloatTensor(train_df.loc[m, "daily_return"].values).to(device))

    test_feats_by_series, test_rets_by_series = [], []
    for sid in TEST_IDS:
        m = test_df["series_id"] == sid
        test_feats_by_series.append(torch.FloatTensor(test_X_all[m]).to(device))
        test_rets_by_series.append(torch.FloatTensor(test_df.loc[m, "daily_return"].values).to(device))

    if is_main:
        print(f"Train: {sum(f.shape[0] for f in train_feats_by_series)} samples")
        print(f"Test : {sum(f.shape[0] for f in test_feats_by_series)} samples")

    # ────────────── Step 4: Model + torch.compile + DDP ──────────────
    # Same seed on all ranks → identical initial weights
    torch.manual_seed(SEED)
    policy = KDAPolicyNetwork().to(device)
    torch.manual_seed(SEED)
    ref_policy = KDAPolicyNetwork().to(device)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    raw_policy = policy  # keep unwrapped reference for evaluation

    if is_main:
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"Parameters: {n_params:,}")

    # torch.compile: unrolls KDA sequential loop into a fused static graph
    if is_main:
        print("Compiling models with torch.compile ...")
    policy     = torch.compile(policy)
    ref_policy = torch.compile(ref_policy)

    # DDP: gradient all-reduce across GPUs
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank])

    if is_main:
        print("Model ready (compiled + DDP).")

    # ────────────── Step 5: GRPO Training ──────────────
    # Per-rank seeds → each GPU samples different episodes for diversity
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPISODES)

    def sample_episode():
        idx = np.random.randint(len(train_feats_by_series))
        feats = train_feats_by_series[idx]
        rets  = train_rets_by_series[idx]
        T = feats.shape[0]
        start = np.random.randint(0, T - EPISODE_LEN - 1)    # -1: need rets[start+EPISODE_LEN]
        # features at t → position[t] → earn return at t+1 (causal)
        f = feats[start:start+EPISODE_LEN]                    # features for t=0..T-1
        r = rets[start+1:start+EPISODE_LEN+1]                 # FUTURE returns for t=1..T
        return f, r

    def grpo_step():
        feats, rets = sample_episode()
        T = feats.shape[0]

        # forward (with grad)
        logits = policy(feats)                          # (T, 11)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            old_log_probs = log_probs.detach()
            old_probs = old_log_probs.exp()
            # sample G trajectories
            dist_cat = Categorical(old_probs)               # (T,)
            actions = dist_cat.sample((G_SAMPLES,))         # (G, T)
            # gather old log probs
            old_lp = old_log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1) \
                                .gather(2, actions.unsqueeze(2)).squeeze(2)  # (G, T)
            # rewards & advantages
            rw = compute_rewards(actions, rets)         # (G,)
        adv = ((rw - rw.mean()) / (rw.std() + 1e-8)).detach()  # (G,)

        # current log probs of sampled actions (with grad)
        cur_lp = log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1) \
                          .gather(2, actions.unsqueeze(2)).squeeze(2)  # (G, T)
        ratio = (cur_lp - old_lp).exp()                # (G, T)
        adv_exp = adv.unsqueeze(1)
        surr1 = ratio * adv_exp
        surr2 = ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * adv_exp
        loss_clip = -torch.min(surr1, surr2).mean()

        # KL(π_ref || π_θ)
        with torch.no_grad():
            ref_lp = F.log_softmax(ref_policy(feats), dim=-1)
            ref_p  = ref_lp.exp()
        kl = (ref_p * (ref_lp - log_probs)).sum(dim=-1).mean()

        loss = loss_clip + BETA_KL * kl
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return loss.item(), rw.mean().item(), rw.std().item()

    # ── Training loop ──
    if is_main:
        print(f"\nTraining {N_EPISODES} episodes  (G={G_SAMPLES}, T={EPISODE_LEN}, "
              f"world_size={world_size})")
    history = {"loss": [], "reward_mean": [], "reward_std": []}
    t0 = time.time()

    for ep in range(1, N_EPISODES + 1):
        loss, rm, rs = grpo_step()
        if is_main:
            history["loss"].append(loss)
            history["reward_mean"].append(rm)
            history["reward_std"].append(rs)
            if ep % 50 == 0 or ep == 1:
                elapsed = time.time() - t0
                print(f"  Ep {ep:4d}/{N_EPISODES}  loss={loss:+.4f}  "
                      f"reward={rm:+.4f}±{rs:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"[{elapsed:.0f}s]")

    if is_main:
        print(f"Training done in {time.time()-t0:.1f}s")

        # training curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history["loss"]);       axes[0].set_title("GRPO Loss")
        axes[1].plot(history["reward_mean"]);axes[1].set_title("Mean Reward")
        axes[2].plot(history["reward_std"]); axes[2].set_title("Reward Std")
        for ax in axes: ax.set_xlabel("Episode")
        plt.tight_layout()
        plt.savefig("fig_02_training.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved fig_02_training.png")

    # Barrier: ensure all ranks finish training before rank 0 evaluates
    if world_size > 1:
        dist.barrier()

    # ────────────── Step 6: Test-Set Evaluation (rank 0 only) ──────────────
    if is_main:
        raw_policy.eval()

        def backtest_series(policy, feats, rets, label=""):
            T = feats.shape[0]
            with torch.no_grad():
                # features[0..T-2] → position, earn return[1..T-1] (causal, no look-ahead)
                logits = policy(feats[:-1])
                positions = logits.argmax(dim=-1).cpu().numpy()  # (T-1,)

            rets_np = rets[1:].cpu().numpy()          # future returns: t+1
            pos_w = positions / 10.0

            # strategy equity
            strat_daily = pos_w * rets_np
            strat_eq = np.cumprod(1 + strat_daily)

            # buy-and-hold
            bh_eq = np.cumprod(1 + rets_np)

            # metrics
            strat_total = strat_eq[-1]
            bh_total    = bh_eq[-1]
            strat_peak  = np.maximum.accumulate(strat_eq)
            strat_dd    = (strat_peak - strat_eq) / strat_peak
            bh_peak     = np.maximum.accumulate(bh_eq)
            bh_dd       = (bh_peak - bh_eq) / bh_peak

            strat_maxdd = strat_dd.max()
            bh_maxdd    = bh_dd.max()

            # annualised Sharpe (252 trading days)
            strat_sharpe = np.mean(strat_daily) / (np.std(strat_daily) + 1e-8) * np.sqrt(252)
            bh_sharpe    = np.mean(rets_np) / (np.std(rets_np) + 1e-8) * np.sqrt(252)

            n_days = len(positions)  # actual trading days = T-1
            strat_calmar = ((strat_total ** (252/n_days) - 1)) / (strat_maxdd + 1e-8)
            bh_calmar    = ((bh_total ** (252/n_days) - 1)) / (bh_maxdd + 1e-8)

            print(f"\n{'='*55}")
            print(f"  {label}")
            print(f"{'='*55}")
            print(f"  {'Metric':<20} {'Strategy':>14} {'Buy&Hold':>14}")
            print(f"  {'-'*48}")
            print(f"  {'Total Return':<20} {strat_total:>13.2f}x {bh_total:>13.2f}x")
            print(f"  {'Max Drawdown':<20} {strat_maxdd:>13.2%} {bh_maxdd:>13.2%}")
            print(f"  {'Sharpe (ann.)':<20} {strat_sharpe:>13.2f}  {bh_sharpe:>13.2f}")
            print(f"  {'Calmar':<20} {strat_calmar:>13.2f}  {bh_calmar:>13.2f}")

            return {
                "positions": positions, "strat_eq": strat_eq, "bh_eq": bh_eq,
                "strat_dd": strat_dd, "bh_dd": bh_dd, "rets": rets_np,
            }

        results = []
        for i, sid in enumerate(TEST_IDS):
            r = backtest_series(raw_policy, test_feats_by_series[i],
                                test_rets_by_series[i],
                                label=f"Test Series {sid}")
            results.append(r)

        # ────────────── Step 7: Visualization ──────────────
        for i, sid in enumerate(TEST_IDS):
            r = results[i]
            T = len(r["strat_eq"])
            days = np.arange(T)

            fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 3, 1]})

            # 1) Equity curves
            axes[0].plot(days, r["bh_eq"],    lw=1, alpha=.6, label="Buy & Hold", color="gray")
            axes[0].plot(days, r["strat_eq"], lw=1, label="KDA+GRPO Strategy", color="tab:blue")
            axes[0].set_ylabel("Equity (start=1)")
            axes[0].legend()
            axes[0].set_title(f"Test Series {sid} — Equity Curves")

            # 2) Drawdown
            axes[1].fill_between(days, -r["bh_dd"],    alpha=.3, color="gray", label="B&H DD")
            axes[1].fill_between(days, -r["strat_dd"], alpha=.4, color="tab:blue", label="Strategy DD")
            axes[1].set_ylabel("Drawdown")
            axes[1].legend()

            # 3) Position heatmap
            axes[2].imshow(r["positions"].reshape(1, -1), aspect="auto",
                           cmap="RdYlGn", vmin=0, vmax=10, interpolation="nearest")
            axes[2].set_ylabel("Position")
            axes[2].set_xlabel("Trading Day")
            axes[2].set_yticks([])

            plt.tight_layout()
            plt.savefig(f"fig_03_test_series_{sid}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved fig_03_test_series_{sid}.png")

        # ── Position distribution ──
        n_test = len(TEST_IDS)
        fig, axes = plt.subplots(1, n_test, figsize=(4*n_test, 4), squeeze=False)
        axes = axes[0]  # always a flat list
        for i, sid in enumerate(TEST_IDS):
            pos = results[i]["positions"]
            axes[i].hist(pos, bins=11, range=(-0.5, 10.5), edgecolor="black", align="mid")
            axes[i].set_title(f"Series {sid} — Position Distribution")
            axes[i].set_xlabel("Position (0–10)")
            axes[i].set_ylabel("Count")
        plt.tight_layout()
        plt.savefig("fig_04_position_dist.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved fig_04_position_dist.png")

        # ────────────── Conclusions ──────────────
        print("\n" + "="*60)
        print("  CONCLUSIONS")
        print("="*60)
        print("""
Model: Mini-KDA Policy Network (adapted from Kimi Linear)
  - 3 stacked KDA layers with delta-rule recurrence
  - Channel-wise gating (α) for fine-grained memory control
  - PoPE position encoding (content-position decoupling)
  - AttnRes depth aggregation
  - ~26K parameters total

Training: GRPO (from DeepSeekMath)
  - No critic/value model needed
  - Group-relative advantage estimation (G=24 samples)
  - Reward = log_return − λ·max_drawdown  (λ=2.0)
  - torch.compile static graph compilation
  - DDP multi-GPU via torchrun

Key observations:
  1. MMn differential features capture short/long-term trend shifts
     through the rolling log-min and log-max dynamics.
  2. The KDA recurrence acts as an adaptive memory that tracks
     market regime changes across multiple time horizons.
  3. GRPO directly optimizes the return-vs-drawdown trade-off
     without needing hand-crafted labels or a value function.

Compare the strategy vs buy-and-hold metrics printed above
to assess whether the MMn signal + KDA + GRPO combination
provides effective position control.
""")
        print("All figures saved to current directory.")
        print("Done.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
