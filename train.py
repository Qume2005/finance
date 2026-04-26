"""GRPO training loop."""

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (SEED, EPISODE_LEN, G_SAMPLES, LAMBDA_DD,
                    EPS_CLIP, BETA_KL, N_EPISODES, LR, BATCH_SIZE,
                    WEIGHT_DECAY, SAVE_EVERY, OUTPUT_DIR, ENTROPY_COEFF,
                    USE_AMP)
from muon import NewtonMuon


def compute_bh_metrics(daily_returns):
    """Pre-compute Buy & Hold return and max drawdown (same across all G samples)."""
    bh_equity = torch.cumprod(1.0 + daily_returns, dim=-1)
    bh_return = bh_equity[..., -1]
    bh_peak = torch.cummax(bh_equity, dim=-1).values
    bh_dd = (bh_peak - bh_equity) / (bh_peak + 1e-8)
    bh_maxdd = bh_dd.max(dim=-1).values
    return bh_return, bh_maxdd


def compute_rewards(positions, daily_returns, bh_return, bh_maxdd, lam=LAMBDA_DD):
    """
    positions:      (G, T) or (G, B, T) LongTensor 0-10
    daily_returns:  (T,)   or (B, T)  FloatTensor
    bh_return:      scalar or (B,)    — pre-computed
    bh_maxdd:       scalar or (B,)    — pre-computed
    returns:        (G,)   or (G, B)  reward per trajectory

    reward = (1-λ)(strat_return - bh_return) + λ(strat_maxdd - bh_maxdd)
    """
    w = positions.float() / 10.0
    pnl = w * daily_returns.unsqueeze(0)
    equity = torch.cumprod(1.0 + pnl, dim=-1)
    strat_return = equity[..., -1]
    peak = torch.cummax(equity, dim=-1).values
    dd = (peak - equity) / (peak + 1e-8)
    strat_maxdd = dd.max(dim=-1).values
    return (1 - lam) * (strat_return - bh_return) + lam * (strat_maxdd - bh_maxdd)


def train_grpo(policy, ref_policy, train_feats, train_rets,
               rank, world_size, is_main):
    """
    Run GRPO training loop.

    Returns history dict (only meaningful on rank 0).
    """
    # Per-rank seeds → each GPU samples different episodes for diversity
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)

    # NewtonMuon for 2D params, SGD for the rest (biases, norms, etc.)
    muon_params, sgd_params = [], []
    for p in policy.parameters():
        (muon_params if p.dim() >= 2 else sgd_params).append(p)
    muon_opt = NewtonMuon(muon_params, lr=LR, momentum=0.618, weight_decay=WEIGHT_DECAY)
    sgd_opt = torch.optim.SGD(sgd_params, lr=LR, momentum=0.618, weight_decay=WEIGHT_DECAY)
    optimizers = [muon_opt, sgd_opt]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPISODES)
        for opt in optimizers
    ]

    # Register activation hooks on unwrapped model for Newton-Muon preconditioner
    raw_model = policy.module if isinstance(policy, (nn.parallel.DistributedDataParallel,)) else policy
    muon_opt.register_hooks(raw_model)

    _global_step = [0]  # mutable counter for Newton-Muon preconditioner refresh

    # ── Resume from checkpoint ──
    start_ep = 1
    history = {"loss": [], "reward_mean": [], "reward_std": []}
    ckpt_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "ckpt_*.pt")))
    if ckpt_files:
        ckpt_path = ckpt_files[-1]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        muon_opt.load_state_dict(ckpt["muon_opt"])
        sgd_opt.load_state_dict(ckpt["sgd_opt"])
        start_ep = ckpt["step"] + 1
        _global_step[0] = start_ep
        history = ckpt.get("history", history)
        if is_main:
            print(f"Resumed from {ckpt_path} (step {ckpt['step']})")
        # fast-forward schedulers to correct lr
        for _ in range(ckpt["step"]):
            for sch in schedulers:
                sch.step()

    def sample_series_batch():
        """Sample full training series (feats[:-1] → rets[1:] alignment)."""
        T = train_feats[0].shape[0]
        seq_len = T - 1
        n_series = len(train_feats)
        indices = torch.randint(n_series, (BATCH_SIZE,))
        f_batch = torch.stack([train_feats[idx.item()][:seq_len] for idx in indices])
        r_batch = torch.stack([train_rets[idx.item()][1:T] for idx in indices])
        return f_batch, r_batch

    def sliding_forward(model, feats):
        """Sliding window forward with KDA state carryover across windows."""
        B, T, D = feats.shape
        n_w = (T + EPISODE_LEN - 1) // EPISODE_LEN
        pad = n_w * EPISODE_LEN - T
        if pad:
            feats = torch.cat([feats, feats.new_zeros(B, pad, D)], dim=1)

        all_logits = []
        kda_states = None
        for w in range(n_w):
            window = feats[:, w * EPISODE_LEN:(w + 1) * EPISODE_LEN, :]
            logits, kda_states = model(window, kda_states=kda_states)
            if model.training:
                kda_states = [s.detach() for s in kda_states]
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)[:, :T, :]

    def grpo_step():
        feats, rets = sample_series_batch()          # (B, SEQ_LEN, 14), (B, SEQ_LEN)

        # Newton-Muon: refresh preconditioner periodically
        _global_step[0] += 1
        if _global_step[0] % muon_opt.refresh_interval == 0:
            muon_opt.update_preconditioner()

        # Pre-compute BH metrics once (same for all G samples)
        bh_return, bh_maxdd = compute_bh_metrics(rets)

        # sliding window forward (with grad + optional AMP)
        with torch.autocast("cuda", enabled=USE_AMP):
            logits = sliding_forward(policy, feats)       # (B, SEQ_LEN, 11)
            log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            old_log_probs = log_probs.detach()
            old_probs = old_log_probs.exp()
            dist_cat = Categorical(old_probs)                   # (B, SEQ_LEN, 11)
            actions = dist_cat.sample((G_SAMPLES,))             # (G, B, SEQ_LEN)
            old_lp = old_log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                                .gather(3, actions.unsqueeze(3)).squeeze(3)
            rw = compute_rewards(actions, rets, bh_return, bh_maxdd)

        adv = rw.detach()

        cur_lp = log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                          .gather(3, actions.unsqueeze(3)).squeeze(3)
        ratio = (cur_lp - old_lp).exp()                        # (G, B, SEQ_LEN)
        adv_exp = adv.unsqueeze(2)                              # (G, B, 1)
        surr1 = ratio * adv_exp
        surr2 = ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * adv_exp
        loss_clip = -torch.min(surr1, surr2).mean()

        # KL(π_ref || π_θ) — 同样滑动窗口
        with torch.no_grad():
            with torch.autocast("cuda", enabled=USE_AMP):
                ref_lp = F.log_softmax(sliding_forward(ref_policy, feats), dim=-1)
            ref_p  = ref_lp.exp()
        kl = (ref_p * (ref_lp - log_probs)).sum(dim=-1).mean()

        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        loss = loss_clip + BETA_KL * kl - ENTROPY_COEFF * entropy
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
        for sch in schedulers:
            sch.step()
        return loss.item(), rw.mean().item(), rw.std().item()

    # ── Training loop ──
    if is_main:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nTraining {N_EPISODES} steps × {BATCH_SIZE} ep/step  "
              f"(G={G_SAMPLES}, window={EPISODE_LEN}, seq={train_feats[0].shape[0]-1}, "
              f"world_size={world_size})"
              + (f"  [resume from step {start_ep-1}]" if start_ep > 1 else ""))
    t0 = time.time()

    for ep in range(start_ep, N_EPISODES + 1):
        loss, rm, rs = grpo_step()
        if is_main:
            history["loss"].append(loss)
            history["reward_mean"].append(rm)
            history["reward_std"].append(rs)
            if ep % 50 == 0 or ep == 1:
                elapsed = time.time() - t0
                print(f"  Step {ep:4d}/{N_EPISODES}  loss={loss:+.4f}  "
                      f"reward={rm:+.4f}±{rs:.4f}  lr={schedulers[0].get_last_lr()[0]:.2e}  "
                      f"[{elapsed:.0f}s]")
            if SAVE_EVERY and ep % SAVE_EVERY == 0:
                # 删旧留新，只保留一个 checkpoint
                for old in glob.glob(os.path.join(OUTPUT_DIR, "ckpt_*.pt")):
                    os.remove(old)
                ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_{ep}.pt")
                torch.save({
                    "step": ep,
                    "model": raw_model.state_dict(),
                    "muon_opt": muon_opt.state_dict(),
                    "sgd_opt": sgd_opt.state_dict(),
                    "history": history,
                }, ckpt_path)
                print(f"  Checkpoint saved → {ckpt_path}")

    if is_main:
        print(f"Training done in {time.time()-t0:.1f}s")

    return history