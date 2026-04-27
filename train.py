"""GRPO training loop."""

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (SEED, EPISODE_LEN, G_SAMPLES, LAMBDA_REWARD, REWARD_SCALE,
                    EPS_CLIP, BETA_KL, N_EPISODES, LR, BATCH_SIZE,
                    WEIGHT_DECAY, SAVE_EVERY, OUTPUT_DIR, ENTROPY_COEFF,
                    MAX_ITERATIONS, MIN_ITERATIONS,
                    ITER_REWARD_START, ITER_REWARD_END, ORTHO_COEFF)
from muon import NewtonMuon


def compute_bh_metrics(daily_returns):
    """Pre-compute B&H baseline metrics (same across all G samples)."""
    bh_sharpe = daily_returns.mean(-1) / (daily_returns.std(-1) + 1e-8)
    bh_return = torch.cumprod(1.0 + daily_returns, dim=-1)[..., -1]
    return bh_sharpe, bh_return


def compute_rewards(positions, daily_returns, bh_sharpe, bh_return):
    """
    reward = (1-λ)*(strat_sharpe - bh_sharpe) + λ*(strat_return - bh_return)
    λ=0: 纯夏普率差值, λ=1: 纯期末收益差值
    """
    w = positions.float() / 10.0
    strat_returns = w * daily_returns.unsqueeze(0)              # (G, B, T)

    strat_sharpe = (strat_returns.mean(-1)
                    / (strat_returns.std(-1) + 1e-8))           # (G, B)
    strat_return = torch.cumprod(1.0 + strat_returns, dim=-1)[..., -1]

    sharpe_diff = strat_sharpe - bh_sharpe.unsqueeze(0)
    return_diff = strat_return - bh_return.unsqueeze(0)
    return ((1 - LAMBDA_REWARD) * sharpe_diff
            + LAMBDA_REWARD * return_diff) * REWARD_SCALE


def train_grpo(policy, ref_policy, train_feats, train_rets,
               rank, world_size, is_main, raw_model=None):
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
        (muon_params if p.dim() == 2 else sgd_params).append(p)
    muon_opt = NewtonMuon(muon_params, lr=LR, momentum=0.618, weight_decay=WEIGHT_DECAY)
    sgd_opt = torch.optim.SGD(sgd_params, lr=LR, momentum=0.618, weight_decay=WEIGHT_DECAY)
    optimizers = [muon_opt, sgd_opt]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPISODES)
        for opt in optimizers
    ]

    # Uncompiled model for Newton-Muon ZZ^T collection (hooks must NOT stay on during compiled forward)
    if raw_model is None:
        raw_model = policy.module if isinstance(policy, (nn.parallel.DistributedDataParallel,)) else policy

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

    def sliding_forward(model, feats, init_kda_states=None):
        """Sliding window forward with KDA state carryover across windows."""
        B, T, D = feats.shape
        n_w = (T + EPISODE_LEN - 1) // EPISODE_LEN
        pad = n_w * EPISODE_LEN - T
        if pad:
            feats = torch.cat([feats, feats.new_zeros(B, pad, D)], dim=1)

        all_logits = []
        all_exit_iters = []
        kda_states = init_kda_states
        for w in range(n_w):
            window = feats[:, w * EPISODE_LEN:(w + 1) * EPISODE_LEN, :]
            logits, kda_states, exit_iter = model(window, kda_states=kda_states)
            if model.training:
                kda_states = [s.detach() for s in kda_states]
            # 中间窗口去掉 EOS，最后一个窗口保留 EOS
            if w < n_w - 1:
                all_logits.append(logits[:, :-1, :])
            else:
                all_logits.append(logits)
            all_exit_iters.append(exit_iter)               # (B,) per window

        total = torch.cat(all_logits, dim=1)
        # 取前 T 步（真实预测）+ 最后 1 步（EOS）= T+1
        # exit_iter: 取最后一个窗口的（代表最终退出深度）
        return (torch.cat([total[:, :T, :], total[:, -1:, :]], dim=1),
                kda_states, all_exit_iters[-1])

    def grpo_step():
        feats, rets = sample_series_batch()          # (B, SEQ_LEN, 14), (B, SEQ_LEN)

        # Newton-Muon: refresh preconditioner periodically
        _global_step[0] += 1
        if _global_step[0] % muon_opt.refresh_interval == 0:
            # Temporarily register hooks → eager forward → collect ZZ^T → remove hooks
            muon_opt.register_hooks(raw_model)
            with torch.no_grad():
                raw_model(feats[:, :EPISODE_LEN, :])
            for h in muon_opt._hooks:
                h.remove()
            muon_opt._hooks = []
            muon_opt.update_preconditioner()

        # KDA 预热：前 EPISODE_LEN 步只积累状态，不预测
        with torch.no_grad():
            _, policy_warmup, _ = sliding_forward(policy, feats[:, :EPISODE_LEN, :])
            _, ref_warmup, _ = sliding_forward(ref_policy, feats[:, :EPISODE_LEN, :])

        # 预热后的特征和收益
        pred_feats = feats[:, EPISODE_LEN:, :]                    # (B, SEQ_LEN-50, 14)
        pred_rets = rets[:, EPISODE_LEN:]                          # (B, SEQ_LEN-50)

        # Pre-compute BH baseline metrics once (same for all G samples)
        bh_sharpe, bh_return = compute_bh_metrics(pred_rets)       # (B,)

        # sliding window forward (with warm-up KDA states)
        logits, _, exit_iters = sliding_forward(policy, pred_feats,
                                                init_kda_states=policy_warmup)

        # logits: (B, T+1, 11) — T real steps + 1 EOS
        T_real = pred_rets.shape[1]

        # float32 log_softmax — avoids -inf in fp16 causing zero-prob Categorical
        log_probs = F.log_softmax(logits.float(), dim=-1)
        log_probs_real = log_probs[:, :T_real, :]            # (B, T, 11) for action/reward

        with torch.no_grad():
            old_log_probs = log_probs_real.detach()
            dist_cat = Categorical(logits=old_log_probs)
            actions = dist_cat.sample((G_SAMPLES,))                 # (G, B, T)
            old_lp = old_log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                                .gather(3, actions.unsqueeze(3)).squeeze(3)
            rw = compute_rewards(actions, pred_rets, bh_sharpe, bh_return)

        adv = rw.detach()

        # ── 迭代深度奖励缩放 ──
        # exit_iters: (B,), 值域 [min_iterations, max_iterations]
        # 线性从 ITER_REWARD_START 到 ITER_REWARD_END
        iter_range = MAX_ITERATIONS - MIN_ITERATIONS
        if iter_range > 0:
            t_frac = ((exit_iters - MIN_ITERATIONS) / iter_range).clamp(0, 1)
        else:
            t_frac = torch.zeros_like(exit_iters)
        iter_scale = ITER_REWARD_START + (ITER_REWARD_END - ITER_REWARD_START) * t_frac
        adv = adv * iter_scale.unsqueeze(0)                 # (G, B)

        cur_lp = log_probs_real.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                          .gather(3, actions.unsqueeze(3)).squeeze(3)
        ratio = (cur_lp - old_lp).exp()
        adv_exp = adv.unsqueeze(2)
        surr1 = ratio * adv_exp
        surr2 = ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * adv_exp
        loss_clip = -torch.min(surr1, surr2).mean()

        # KL(π_ref || π_θ) — includes EOS
        with torch.no_grad():
            ref_logits, _, _ = sliding_forward(ref_policy, pred_feats,
                                                init_kda_states=ref_warmup)
            ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
            ref_p  = ref_lp.exp()
        kl = (ref_p * (ref_lp - log_probs)).sum(dim=-1).mean()

        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        # 路由器行向量正交正则
        if ORTHO_COEFF > 0:
            ortho_loss = torch.tensor(0.0, device=feats.device)
            for router in [raw_model.moa_kda.router_q,
                           raw_model.moa_kda.router_kv,
                           raw_model.moe_swiglu.expert_router,
                           raw_model.router.proj]:                   # halting STE
                W = F.normalize(router.weight, dim=1)               # (E, d) 按行归一
                E = W.shape[0]
                G = W @ W.T                                         # (E, E)
                mask = ~torch.eye(E, dtype=torch.bool, device=G.device)
                dot_sum = G[mask].sum()                             # 两两点乘之和
                ortho_loss = ortho_loss + dot_sum.pow(2) / E
            ortho_loss = ORTHO_COEFF * ortho_loss
        else:
            ortho_loss = torch.tensor(0.0, device=feats.device)

        loss = loss_clip + BETA_KL * kl - ENTROPY_COEFF * entropy + ortho_loss
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

    try:
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
    except KeyboardInterrupt:
        if is_main:
            last_ep = ep if 'ep' in dir() else start_ep
            for old in glob.glob(os.path.join(OUTPUT_DIR, "ckpt_*.pt")):
                os.remove(old)
            ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_{last_ep}.pt")
            torch.save({
                "step": last_ep,
                "model": raw_model.state_dict(),
                "muon_opt": muon_opt.state_dict(),
                "sgd_opt": sgd_opt.state_dict(),
                "history": history,
            }, ckpt_path)
            print(f"\n  Interrupted at step {last_ep}. Checkpoint saved → {ckpt_path}")

    if is_main:
        print(f"Training done in {time.time()-t0:.1f}s")

    return history