"""GRPO training loop."""

import os
import glob
import time
import threading
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (SEED, EPISODE_LEN, G_SAMPLES, LAMBDA_REWARD, REWARD_SCALE,
                    EPS_CLIP, BETA_KL, N_EPISODES, LR, BATCH_SIZE,
                    WEIGHT_DECAY, SAVE_EVERY, OUTPUT_DIR, ENTROPY_COEFF,
                    MAX_ITERATIONS, MIN_ITERATIONS,
                    ITER_REWARD_START, ITER_REWARD_END, ORTHO_COEFF,
                    DEPTH_PENALTY_COEFF,
                    N_ATTN_HEADS, N_FFN_HEADS,
                    N_ATTN_HEADS_PER_CARD, N_FFN_HEADS_PER_CARD,
                    HEAD_ROTATION_INTERVAL)
from muon import NewtonMuon
from torch.cuda.amp import GradScaler


# ────────────────── Head-wise expert parallelism ──────────────────

_ACTION_HEAD_SUFFIXES = ('head_down.weight', 'head_gate.weight', 'head_up.weight')


def compute_head_masks(step, world_size, rank):
    """Return (attn_mask, ffn_mask) — bool tensors, True = gradient on this GPU."""
    if world_size <= 1:
        return None, None

    n_attn = N_ATTN_HEADS
    n_ffn = N_FFN_HEADS
    a_per = N_ATTN_HEADS_PER_CARD or n_attn
    f_per = N_FFN_HEADS_PER_CARD or n_ffn

    if a_per >= n_attn and f_per >= n_ffn:
        return None, None                                  # 退化为普通 DDP

    epoch = step // HEAD_ROTATION_INTERVAL if HEAD_ROTATION_INTERVAL > 0 else 0

    # 确定性随机排列（所有 GPU 同 seed → 同排列 → 各取不同 chunk）
    rng = torch.Generator().manual_seed(SEED + epoch)
    attn_perm = torch.randperm(n_attn, generator=rng)
    rng.manual_seed(SEED + epoch + 10000)
    ffn_perm = torch.randperm(n_ffn, generator=rng)

    def make_mask(perm, n_total, n_per_card):
        chunk = min(n_per_card, (n_total + world_size - 1) // world_size)
        start = rank * chunk
        end = min(start + chunk, n_total)
        mask = torch.zeros(n_total, dtype=torch.bool)
        if start < n_total:
            mask[perm[start:end]] = True
        return mask

    return make_mask(attn_perm, n_attn, a_per), make_mask(ffn_perm, n_ffn, f_per)


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
    scaler = GradScaler()

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
        # 兼容旧 checkpoint：q_router_w + kv_router_w → router_w
        sd = ckpt["model"]
        if "moa_kda.q_router_w" in sd:
            sd["moa_kda.router_w"] = sd.pop("moa_kda.q_router_w")
            sd.pop("moa_kda.kv_router_w", None)
        # 兼容旧 checkpoint 的 _orig_mod. 前缀
        raw_model.load_state_dict({k.replace("_orig_mod.", ""): v
                                    for k, v in ckpt["model"].items()})
        muon_opt.load_state_dict(ckpt["muon_opt"])
        sgd_opt.load_state_dict(ckpt["sgd_opt"])
        start_ep = ckpt["step"] + 1
        _global_step[0] = start_ep
        history = ckpt.get("history", history)
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if is_main:
            print(f"Resumed from {ckpt_path} (step {ckpt['step']})")
        # fast-forward schedulers to correct lr
        for _ in range(ckpt["step"]):
            for sch in schedulers:
                sch.step()

    def sample_series_batch():
        """Sample full training series from CPU, transfer to GPU."""
        T = train_feats[0].shape[0]
        seq_len = T - 1
        n_series = len(train_feats)
        indices = torch.randint(n_series, (BATCH_SIZE,))
        f_batch = torch.stack([train_feats[idx.item()][:seq_len] for idx in indices]).to(rank, non_blocking=True)
        r_batch = torch.stack([train_rets[idx.item()][:seq_len] for idx in indices]).to(rank, non_blocking=True)
        return f_batch, r_batch

    def sliding_forward(model, feats, attn_head_mask=None, ffn_head_mask=None):
        """Parallel window forward — all windows as one batch."""
        B, T, D = feats.shape
        n_w = (T + EPISODE_LEN - 1) // EPISODE_LEN
        pad = n_w * EPISODE_LEN - T
        if pad:
            feats = torch.cat([feats, feats.new_zeros(B, pad, D)], dim=1)

        # (B, n_w*EPISODE_LEN, D) → (B*n_w, EPISODE_LEN, D)
        windows = feats.reshape(B * n_w, EPISODE_LEN, D)

        logits, exit_iters, route_lp, expected_depth = model(
            windows, attn_head_mask=attn_head_mask, ffn_head_mask=ffn_head_mask)

        # logits: (B*n_w, n_actions) — one action per window, predicts NEXT window
        result = logits.reshape(B, n_w, -1)                  # (B, n_w, n_actions)

        exit_avg = exit_iters.reshape(B, n_w).float().mean(dim=-1)
        route_lp_avg = route_lp.reshape(B, n_w).float().mean(dim=-1)  # (B,)
        exp_depth_avg = expected_depth.reshape(B, n_w).float().mean(dim=-1)  # (B,)
        return result, exit_avg, route_lp_avg, exp_depth_avg

    # ── Data prefetcher: overlap CPU sampling with GPU compute ──
    class _Prefetcher:
        def __init__(self, sample_fn):
            self.sample_fn = sample_fn
            self._next = sample_fn()
            self._thread = None

        def _prefetch(self):
            self._next = self.sample_fn()

        def get(self):
            if self._thread is not None:
                self._thread.join()
            batch = self._next
            self._thread = threading.Thread(target=self._prefetch, daemon=True)
            self._thread.start()
            return batch

    prefetcher = _Prefetcher(sample_series_batch)

    def grpo_step():
        feats, rets = prefetcher.get()
        _global_step[0] += 1
        if _global_step[0] % muon_opt.refresh_interval == 0:
            torch.cuda.current_stream(rank).synchronize()
            muon_opt.register_hooks(raw_model)
            with torch.no_grad():
                raw_model(feats[:, :EPISODE_LEN, :])
            for h in muon_opt._hooks:
                h.remove()
            muon_opt._hooks = []
            muon_opt.update_preconditioner()

        # Head-wise expert parallelism masks
        attn_mask, ffn_mask = compute_head_masks(_global_step[0], world_size, rank)
        head_masks_active = (attn_mask is not None or ffn_mask is not None)

        bh_sharpe, bh_return = compute_bh_metrics(rets)

        # ── ref_policy forward 先算（no_grad，峰值低）──
        ref_logits_cache = None
        if BETA_KL > 0:
            with torch.no_grad():
                ref_logits_cache = sliding_forward(ref_policy, feats)[0]

        # ── 单次 forward：同时用于采样和梯度 ──
        logits, exit_iters, route_lp, exp_depth = sliding_forward(
            policy, feats, attn_mask, ffn_mask)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # 采样 actions（detached，不影响梯度）
        T_real = feats.shape[1]
        with torch.no_grad():
            actions = Categorical(logits=log_probs).sample((G_SAMPLES,))
            old_action_lp = log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                                .gather(3, actions.unsqueeze(3)).squeeze(3).detach()

        # Shift + reward
        default = torch.full((G_SAMPLES, feats.shape[0], EPISODE_LEN), 5,
                             dtype=actions.dtype, device=actions.device)
        shifted = actions[:, :, :-1].unsqueeze(-1).expand(-1, -1, -1, EPISODE_LEN) \
                      .reshape(G_SAMPLES, feats.shape[0], -1)
        actions_tiled = torch.cat([default, shifted], dim=-1)[:, :, :T_real]
        rw = compute_rewards(actions_tiled, rets, bh_sharpe, bh_return)
        adv = rw.detach()

        # 迭代深度奖励缩放
        iter_range = MAX_ITERATIONS - MIN_ITERATIONS
        if iter_range > 0:
            t_frac = ((exit_iters - MIN_ITERATIONS) / iter_range).clamp(0, 1)
        else:
            t_frac = torch.zeros_like(exit_iters)
        iter_scale = ITER_REWARD_START + (ITER_REWARD_END - ITER_REWARD_START) * t_frac
        adv = adv * iter_scale.unsqueeze(0)

        # ── Loss 计算 ──
        cur_action_lp = log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                          .gather(3, actions.unsqueeze(3)).squeeze(3)
        ratio = (cur_action_lp.sum(-1) - old_action_lp.sum(-1)).exp()
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - EPS_CLIP, 1 + EPS_CLIP) * adv
        loss_clip = -torch.min(surr1, surr2).mean()

        route_reward = rw.mean(0)
        route_baseline = route_reward.mean().detach()
        route_advantage = (route_reward - route_baseline).detach()
        loss_route_reinforce = -(route_lp * route_advantage).mean()

        loss_depth = DEPTH_PENALTY_COEFF * exp_depth.mean()
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        if ORTHO_COEFF > 0:
            ortho_loss = torch.tensor(0.0, device=feats.device)
            for W in [raw_model.moa_kda.router_w,
                      raw_model.moe_swiglu.router_w,
                      raw_model.router.proj.weight]:
                W_flat = W.reshape(-1, W.shape[-1])
                E, d = W_flat.shape
                G = W_flat @ W_flat.T
                idx = torch.triu_indices(E, E, offset=1, device=G.device)
                pairs = G[idx[0], idx[1]]
                ortho_loss = ortho_loss + pairs.pow(2).mean() * (d / E) / 2
            ortho_loss = ORTHO_COEFF * ortho_loss
        else:
            ortho_loss = torch.tensor(0.0, device=feats.device)

        kl = torch.tensor(0.0, device=feats.device)
        if BETA_KL > 0 and ref_logits_cache is not None:
            ref_lp = F.log_softmax(ref_logits_cache.float(), dim=-1)
            ref_p  = ref_lp.exp()
            kl = (ref_p * (ref_lp - log_probs)).sum(dim=-1).mean()

        loss = loss_clip + loss_route_reinforce + BETA_KL * kl - ENTROPY_COEFF * entropy + ortho_loss + loss_depth
        for opt in optimizers:
            opt.zero_grad()
        scaler.scale(loss).backward()
        for opt in optimizers:
            scaler.unscale_(opt)
        # 梯度缩放：expert 参数只有 1/world_size 的正确值
        if world_size > 1 and head_masks_active:
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    if not any(name.endswith(s) for s in _ACTION_HEAD_SUFFIXES):
                        param.grad *= world_size
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        for sch in schedulers:
            sch.step()

        return loss.item(), rw.mean().item(), rw.std().item()

    # ── Signal handler for graceful shutdown (all processes) ──
    _graceful = [False]
    def _sig_handler(signum, frame):
        _graceful[0] = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.siginterrupt(signal.SIGINT, True)
    signal.siginterrupt(signal.SIGTERM, True)

    def _save_ckpt(ep):
        ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_{ep}.pt")
        tmp_path = ckpt_path + ".tmp"
        torch.save({
            "step": ep,
            "model": raw_model.state_dict(),
            "muon_opt": muon_opt.state_dict(),
            "sgd_opt": sgd_opt.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
        }, tmp_path)
        os.replace(tmp_path, ckpt_path)
        for old in glob.glob(os.path.join(OUTPUT_DIR, "ckpt_*.pt")):
            if old != ckpt_path:
                os.remove(old)
        return ckpt_path

    # ── Training loop ──
    if is_main:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nTraining {N_EPISODES} steps × {BATCH_SIZE} ep/step  "
              f"(G={G_SAMPLES}, window={EPISODE_LEN}, seq={train_feats[0].shape[0]-1}, "
              f"world_size={world_size})"
              + (f"  [resume from step {start_ep-1}]" if start_ep > 1 else ""))

    if is_main:
        print(f"""
                    _ooOoo_
                   o8888888o
                   88" . "88
                   (| -_- |)
                   O\\  =  /O
                ____/`---'\\____
              .'  \\\\|     |//  `.
             /  \\\\|||  :  |||//  \\
            /  _||||| -:- |||||-  \\
            |   | \\\\\\  -  /// |   |
            | \\_|  ''\\---/''  |   |
            \\  .-\\__  `-`  ___/-. /
          ___`. .'  /--.--\\  `. . __
       ."" '<  `.___\\_<|>_/___.'  >'"".
      | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |
      \\  \\ `-.   \\_ __\\ /__ _/   .-` /  /
 ======`-.____`-.___\\_____/___.-`____.-'======
                    `=---='
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         佛祖保佑     不掉驱动
""")

    t0 = time.time()

    last_ep = start_ep - 1
    interrupted = False
    try:
        for ep in range(start_ep, N_EPISODES + 1):
            if _graceful[0]:
                interrupted = True
                break
            loss, rm, rs = grpo_step()
            last_ep = ep
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
                    ckpt_path = _save_ckpt(ep)
                    print(f"  Checkpoint saved → {ckpt_path}")
    except KeyboardInterrupt:
        interrupted = True
    except RuntimeError as e:
        # 多卡时非 rank-0 退出会导致 NCCL 错误，正常保存 checkpoint
        if world_size > 1 and ("NCCL" in str(e) or "CUDA" in str(e)):
            interrupted = True
        else:
            raise
    finally:
        if is_main and interrupted and last_ep >= start_ep:
            try:
                ckpt_path = _save_ckpt(last_ep)
                print(f"\n  Interrupted at step {last_ep}. Checkpoint saved → {ckpt_path}")
            except Exception:
                print(f"\n  Interrupted at step {last_ep}. Checkpoint save FAILED.")
        # 恢复默认信号处理器，避免后续 Ctrl+C 无反应
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if is_main:
        if interrupted:
            print(f"Training interrupted at step {last_ep} ({time.time()-t0:.1f}s)")
        else:
            print(f"Training done in {time.time()-t0:.1f}s")

    return history