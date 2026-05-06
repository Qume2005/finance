"""GRPO training loop."""

import os
import glob
import time
import threading
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
from torch.amp import GradScaler


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
        return None, None

    epoch = step // HEAD_ROTATION_INTERVAL if HEAD_ROTATION_INTERVAL > 0 else 0

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


# ────────────────── Gradient averaging (driver-side) ──────────────────


# ────────────────── GRPOTrainer ──────────────────

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


class GRPOTrainer:
    """Single-GPU GRPO trainer. Gradient sync handled externally by driver."""

    def __init__(self, device, rank, world_size):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

    def setup(self, train_feats, train_rets, feat_mean=None, feat_std=None):
        """Create model, optimizers, load checkpoint, init prefetcher."""
        np.random.seed(SEED + self.rank)
        torch.manual_seed(SEED + self.rank)

        self.train_feats = train_feats
        self.train_rets = train_rets
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        from model import KDAPolicyNetwork
        self.model = KDAPolicyNetwork().to(self.device)
        torch.manual_seed(SEED)
        self.ref_policy = KDAPolicyNetwork().to(self.device)
        self.ref_policy.load_state_dict(self.model.state_dict())
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # NewtonMuon for 2D params, SGD for the rest
        muon_params, sgd_params = [], []
        for p in self.model.parameters():
            (muon_params if p.dim() == 2 else sgd_params).append(p)
        self.muon_opt = NewtonMuon(muon_params, lr=LR, momentum=0.618, weight_decay=WEIGHT_DECAY)
        self.sgd_opt = torch.optim.SGD(sgd_params, lr=LR, momentum=0.618, weight_decay=WEIGHT_DECAY)
        self.optimizers = [self.muon_opt, self.sgd_opt]
        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPISODES)
            for opt in self.optimizers
        ]
        self.scaler = GradScaler("cuda")

        self.start_ep = 1
        self.history = {"loss": [], "reward_mean": [], "reward_std": []}
        self._load_checkpoint()

        self.prefetcher = _Prefetcher(self._sample_series_batch)

    def _load_checkpoint(self):
        ckpt_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "ckpt_*.pt")))
        if not ckpt_files:
            return
        ckpt_path = ckpt_files[-1]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        if "moa_kda.q_router_w" in sd:
            sd["moa_kda.router_w"] = sd.pop("moa_kda.q_router_w")
            sd.pop("moa_kda.kv_router_w", None)
        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        # 只加载形状匹配的参数（架构变更后自动跳过不兼容的）
        model_sd = self.model.state_dict()
        loaded, skipped = {}, []
        for k, v in clean_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                loaded[k] = v
            elif k in model_sd:
                skipped.append(f"{k}: {v.shape}→{model_sd[k].shape}")
        self.model.load_state_dict(loaded, strict=False)
        if skipped:
            print(f"  WARNING: {len(skipped)} params skipped (architecture changed):")
            for s in skipped[:5]:
                print(f"    {s}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped)-5} more")
            print(f"  Training will resume from scratch for incompatible params.")
        # 优化器仅在没有跳过参数时才恢复（否则梯度状态不一致）
        if not skipped:
            self.muon_opt.load_state_dict(ckpt["muon_opt"])
            self.sgd_opt.load_state_dict(ckpt["sgd_opt"])
            self.start_ep = ckpt["step"] + 1
            if "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])
            for _ in range(ckpt["step"]):
                for sch in self.schedulers:
                    sch.step()
        else:
            self.start_ep = 1
        self.history = ckpt.get("history", self.history)
        if self.is_main:
            print(f"Resumed from {ckpt_path} (step {ckpt['step']}, "
                  f"loaded {len(loaded)}/{len(model_sd)} params)")

    def _sample_series_batch(self):
        T = self.train_feats[0].shape[0]
        seq_len = T - 1
        n_series = len(self.train_feats)
        indices = torch.randint(n_series, (BATCH_SIZE,))
        f_batch = torch.stack([self.train_feats[idx.item()][:seq_len]
                               for idx in indices]).to(self.device, non_blocking=True)
        r_batch = torch.stack([self.train_rets[idx.item()][:seq_len]
                               for idx in indices]).to(self.device, non_blocking=True)
        return f_batch, r_batch

    def _sliding_forward(self, model, feats, attn_head_mask=None, ffn_head_mask=None):
        B, T, D = feats.shape
        n_w = (T + EPISODE_LEN - 1) // EPISODE_LEN
        pad = n_w * EPISODE_LEN - T
        if pad:
            feats = torch.cat([feats, feats.new_zeros(B, pad, D)], dim=1)
        windows = feats.reshape(B * n_w, EPISODE_LEN, D)
        logits, exit_iters, route_lp, expected_depth = model(
            windows, attn_head_mask=attn_head_mask, ffn_head_mask=ffn_head_mask)
        result = logits.reshape(B, n_w, -1)
        exit_avg = exit_iters.reshape(B, n_w).float().mean(dim=-1)
        route_lp_avg = route_lp.reshape(B, n_w).float().mean(dim=-1)
        exp_depth_avg = expected_depth.reshape(B, n_w).float().mean(dim=-1)
        return result, exit_avg, route_lp_avg, exp_depth_avg

    def _forward_backward(self, step):
        """Forward + backward + unscale. Returns metrics dict. No optimizer step."""
        feats, rets = self.prefetcher.get()

        # NewtonMuon preconditioner refresh
        if step % self.muon_opt.refresh_interval == 0:
            torch.cuda.current_stream(self.device).synchronize()
            self.muon_opt.register_hooks(self.model)
            with torch.no_grad():
                self.model(feats[:, :EPISODE_LEN, :])
            for h in self.muon_opt._hooks:
                h.remove()
            self.muon_opt._hooks = []
            self.muon_opt.update_preconditioner()

        # Head-wise expert parallelism masks
        attn_mask, ffn_mask = compute_head_masks(step, self.world_size, self.rank)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)
        if ffn_mask is not None:
            ffn_mask = ffn_mask.to(self.device)

        bh_sharpe, bh_return = compute_bh_metrics(rets)

        # ref_policy forward
        ref_logits_cache = None
        if BETA_KL > 0:
            with torch.no_grad():
                ref_logits_cache = self._sliding_forward(self.ref_policy, feats)[0]

        # policy forward
        logits, exit_iters, route_lp, exp_depth = self._sliding_forward(
            self.model, feats, attn_mask, ffn_mask)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        # sample actions
        T_real = feats.shape[1]
        with torch.no_grad():
            actions = Categorical(logits=log_probs).sample((G_SAMPLES,))
            old_action_lp = log_probs.unsqueeze(0).expand(G_SAMPLES, -1, -1, -1) \
                                .gather(3, actions.unsqueeze(3)).squeeze(3).detach()

        # shift + reward
        default = torch.full((G_SAMPLES, feats.shape[0], EPISODE_LEN), 5,
                             dtype=actions.dtype, device=actions.device)
        shifted = actions[:, :, :-1].unsqueeze(-1).expand(-1, -1, -1, EPISODE_LEN) \
                      .reshape(G_SAMPLES, feats.shape[0], -1)
        actions_tiled = torch.cat([default, shifted], dim=-1)[:, :, :T_real]
        rw = compute_rewards(actions_tiled, rets, bh_sharpe, bh_return)
        adv = rw.detach()

        # depth reward scaling
        iter_range = MAX_ITERATIONS - MIN_ITERATIONS
        if iter_range > 0:
            t_frac = ((exit_iters - MIN_ITERATIONS) / iter_range).clamp(0, 1)
        else:
            t_frac = torch.zeros_like(exit_iters)
        iter_scale = ITER_REWARD_START + (ITER_REWARD_END - ITER_REWARD_START) * t_frac
        adv = adv * iter_scale.unsqueeze(0)

        # loss
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
            for W in [self.model.moa_kda.router_w,
                      self.model.moe_swiglu.router_w,
                      self.model.router.proj.weight]:
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

        loss = (loss_clip + loss_route_reinforce + BETA_KL * kl
                - ENTROPY_COEFF * entropy + ortho_loss + loss_depth)

        for opt in self.optimizers:
            opt.zero_grad()
        self.scaler.scale(loss).backward()
        for opt in self.optimizers:
            self.scaler.unscale_(opt)

        return {"loss": loss.item(), "reward_mean": rw.mean().item(),
                "reward_std": rw.std().item()}

    def compute_gradients(self, step):
        """Single-GPU: forward + backward. Returns metrics dict."""
        return self._forward_backward(step)

    def init_nccl(self, master_addr, port):
        """初始化 NCCL 进程组（Ray 多卡模式）。"""
        import torch.distributed as dist
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        self._nccl = True

    def step_synced(self, step):
        """多卡完整一步：forward + backward + NCCL allreduce + clip + step + broadcast。"""
        import torch.distributed as dist

        metrics = self._forward_backward(step)

        # NCCL allreduce on GPU
        for name, param in self.model.named_parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            if any(name.endswith(s) for s in _ACTION_HEAD_SUFFIXES):
                param.grad /= self.world_size

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        for opt in self.optimizers:
            self.scaler.step(opt)
        self.scaler.update()
        for sch in self.schedulers:
            sch.step()

        # 广播 rank 0 的模型权重，防止浮点漂移
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

        return metrics

    def get_gradients(self):
        """Return {name: cpu_tensor} for all params with gradients."""
        return {n: p.grad.cpu().clone()
                for n, p in self.model.named_parameters()
                if p.grad is not None}

    def get_model_state(self):
        """Return model state_dict on CPU."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_feat_stats(self):
        """Return (feat_mean, feat_std) tensors on CPU."""
        return self.feat_mean, self.feat_std

    def apply_gradients(self, avg_grads):
        """Set averaged gradients, clip, step optimizer, update scaler."""
        for n, p in self.model.named_parameters():
            if n in avg_grads:
                p.grad = avg_grads[n].to(self.device)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        for opt in self.optimizers:
            self.scaler.step(opt)
        self.scaler.update()
        for sch in self.schedulers:
            sch.step()

    def save_checkpoint(self, step, history):
        """Save checkpoint (rank 0 only)."""
        if not self.is_main:
            return None
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_{step}.pt")
        tmp_path = ckpt_path + ".tmp"
        torch.save({
            "step": step,
            "model": self.model.state_dict(),
            "muon_opt": self.muon_opt.state_dict(),
            "sgd_opt": self.sgd_opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "history": history,
        }, tmp_path)
        os.replace(tmp_path, ckpt_path)
        for old in glob.glob(os.path.join(OUTPUT_DIR, "ckpt_*.pt")):
            if old != ckpt_path:
                os.remove(old)
        return ckpt_path

    def get_history(self):
        return self.history

    def get_start_ep(self):
        return self.start_ep


# ────────────────── Single-GPU training loop ──────────────────

def train_single(train_feats, train_rets, device):
    """Run GRPO training on single GPU. Returns history dict."""
    trainer = GRPOTrainer(device, rank=0, world_size=1)
    trainer.setup(train_feats, train_rets)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nTraining {N_EPISODES} steps × {BATCH_SIZE} ep/step  "
          f"(G={G_SAMPLES}, window={EPISODE_LEN}, "
          f"seq={train_feats[0].shape[0]-1}, world_size=1)"
          + (f"  [resume from step {trainer.start_ep-1}]"
             if trainer.start_ep > 1 else ""))

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

    history = trainer.history
    t0 = time.time()
    last_ep = trainer.start_ep - 1
    interrupted = False

    try:
        for ep in range(trainer.start_ep, N_EPISODES + 1):
            metrics = trainer.compute_gradients(ep)
            # single GPU: apply own gradients directly
            trainer.apply_gradients(trainer.get_gradients())
            last_ep = ep

            history["loss"].append(metrics["loss"])
            history["reward_mean"].append(metrics["reward_mean"])
            history["reward_std"].append(metrics["reward_std"])
            if ep % 50 == 0 or ep == 1:
                elapsed = time.time() - t0
                print(f"  Step {ep:4d}/{N_EPISODES}  loss={metrics['loss']:+.4f}  "
                      f"reward={metrics['reward_mean']:+.4f}±{metrics['reward_std']:.4f}  "
                      f"lr={trainer.schedulers[0].get_last_lr()[0]:.2e}  "
                      f"[{elapsed:.0f}s]")
            if SAVE_EVERY and ep % SAVE_EVERY == 0:
                ckpt_path = trainer.save_checkpoint(ep, history)
                print(f"  Checkpoint saved → {ckpt_path}")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if last_ep >= trainer.start_ep:
            try:
                ckpt_path = trainer.save_checkpoint(last_ep, history)
                if interrupted:
                    print(f"\n  Interrupted at step {last_ep}. Checkpoint saved → {ckpt_path}")
            except Exception:
                if interrupted:
                    print(f"\n  Interrupted at step {last_ep}. Checkpoint save FAILED.")

    if interrupted:
        print(f"Training interrupted at step {last_ep} ({time.time()-t0:.1f}s)")
    else:
        print(f"Training done in {time.time()-t0:.1f}s")

    return history, trainer.model
