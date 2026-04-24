"""GRPO training loop."""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (SEED, EPISODE_LEN, G_SAMPLES, LAMBDA_DD,
                    EPS_CLIP, BETA_KL, N_EPISODES, LR, BATCH_SIZE)


def compute_rewards(positions, daily_returns, lam=LAMBDA_DD):
    """
    positions:      (G, T) or (G, B, T) LongTensor 0-10
    daily_returns:  (T,)   or (B, T)  FloatTensor
    returns:        (G,)   or (G, B)  reward per trajectory
    """
    w = positions.float() / 10.0
    pnl = w * daily_returns.unsqueeze(0)
    equity = torch.cumprod(1.0 + pnl, dim=-1)
    peak = torch.cummax(equity, dim=-1).values
    dd = (peak - equity) / (peak + 1e-8)
    max_dd = dd.max(dim=-1).values
    log_ret = torch.log(equity[..., -1] + 1e-8)
    return log_ret - lam * max_dd


def train_grpo(policy, ref_policy, train_feats, train_rets,
               rank, world_size, is_main):
    """
    Run GRPO training loop.

    Returns history dict (only meaningful on rank 0).
    """
    # Per-rank seeds → each GPU samples different episodes for diversity
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPISODES)

    def sample_episode_batch():
        # Pre-allocate output tensors
        f_batch = torch.empty(BATCH_SIZE, EPISODE_LEN, train_feats[0].shape[1],
                              device=train_feats[0].device)
        r_batch = torch.empty(BATCH_SIZE, EPISODE_LEN, device=train_rets[0].device)
        for i in range(BATCH_SIZE):
            idx = torch.randint(len(train_feats), (1,)).item()
            T = train_feats[idx].shape[0]
            start = torch.randint(0, T - EPISODE_LEN - 1, (1,)).item()
            f_batch[i] = train_feats[idx][start:start+EPISODE_LEN]
            r_batch[i] = train_rets[idx][start+1:start+EPISODE_LEN+1]
        return f_batch, r_batch

    def grpo_step():
        feats, rets = sample_episode_batch()            # (B, T, 14), (B, T)

        # forward (with grad)
        logits = policy(feats)                          # (B, T, 11)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            old_log_probs = log_probs.detach()
            old_probs = old_log_probs.exp()
            # sample G trajectories per episode
            dist_cat = Categorical(old_probs)                   # (B, T, 11)
            actions = dist_cat.sample((G_SAMPLES,))             # (G, B, T)
            # gather old log probs
            old_lp = old_log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                                .gather(3, actions.unsqueeze(3)).squeeze(3)
            # rewards & advantages
            rw = compute_rewards(actions, rets)                 # (G, B)

        # normalize advantages per-episode (across G samples)
        adv = ((rw - rw.mean(0)) / (rw.std(0) + 1e-8)).detach()

        # current log probs of sampled actions (with grad)
        cur_lp = log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                          .gather(3, actions.unsqueeze(3)).squeeze(3)
        ratio = (cur_lp - old_lp).exp()                        # (G, B, T)
        adv_exp = adv.unsqueeze(2)                              # (G, B, 1)
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
        print(f"\nTraining {N_EPISODES} steps × {BATCH_SIZE} ep/step  "
              f"(G={G_SAMPLES}, T={EPISODE_LEN}, world_size={world_size})")
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
                print(f"  Step {ep:4d}/{N_EPISODES}  loss={loss:+.4f}  "
                      f"reward={rm:+.4f}±{rs:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"[{elapsed:.0f}s]")

    if is_main:
        print(f"Training done in {time.time()-t0:.1f}s")

    return history
