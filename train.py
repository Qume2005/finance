"""GRPO training loop."""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import SEED, EPISODE_LEN, G_SAMPLES, LAMBDA_DD, EPS_CLIP, BETA_KL, N_EPISODES, LR
from model import KDAPolicyNetwork


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

    def sample_episode():
        idx = torch.randint(len(train_feats), (1,)).item()
        feats = train_feats[idx]
        rets  = train_rets[idx]
        T = feats.shape[0]
        start = torch.randint(0, T - EPISODE_LEN - 1, (1,)).item()
        # features at t → position[t] → earn return at t+1 (causal)
        f = feats[start:start+EPISODE_LEN]
        r = rets[start+1:start+EPISODE_LEN+1]
        return f, r

    def grpo_step():
        feats, rets = sample_episode()

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

    return history
