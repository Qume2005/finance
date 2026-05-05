"""Mini-KDA Policy Network with PoPE, mHC, AttnRes, MoA, MoE, dynamic loop."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp

from config import (D_HIDDEN, D_KEY, D_ATTN_HEAD, N_ATTN_HEADS, N_EXPERTS_PER_HEAD,
                    D_FFN_HEAD, N_FFN_HEADS, N_EXPERTS_PER_FFN_HEAD,
                    MAX_ITERATIONS, MIN_ITERATIONS, INNER_STEPS,
                    ROUTE_TEMP, HALT_TEMP,
                    GRADIENT_CHECKPOINTING, HEAD_DROP_PROB)


# ────────────────── Utilities ──────────────────

class _RMSNorm(nn.Module):
    """RMSNorm that preserves input dtype (handles fp16 correctly)."""
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True).add(self.eps)).to(x.dtype) * self.weight


def _einsum(eq, *operands):
    """Einsum with fp16 matmul (V100 Tensor Core), fp32 result."""
    ops = [op.half() if op.is_floating_point() else op for op in operands]
    return torch.einsum(eq, *ops).float()


def _bmm(a, b):
    """Batched matmul with fp16 (V100 Tensor Core), fp32 result."""
    return torch.bmm(a.half(), b.half()).float()


def sinkhorn_knopp(M, n_iters=6):
    """Project (..., n, n) to doubly stochastic matrix."""
    M = torch.exp(M)
    for _ in range(n_iters):
        M = M / M.sum(dim=-1, keepdim=True)
        M = M / M.sum(dim=-2, keepdim=True)
    return M


def top_prob_max_k(logits, threshold=0.8, max_k=4):
    """
    Top-Prob&max-K routing: cumulative probability cutoff with K upper limit.
    logits: (..., n_experts) → gates: (..., n_experts) sparse, renormalized.
    """
    probs = F.softmax(logits, dim=-1)
    sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_p.cumsum(dim=-1)

    arange = torch.arange(logits.shape[-1], device=logits.device)
    mask = (cumsum - sorted_p < threshold) & (arange < max_k)
    mask[..., 0] = True                                # at least 1 expert

    selected = sorted_p * mask.float()

    gates = torch.zeros_like(probs).scatter_(dim=-1, index=sorted_idx, src=selected)
    return gates


# ────────────────── mHC ──────────────────

class MHC(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC, arXiv 2512.24880).
    n-stream residual with doubly stochastic mixing.
    """
    def __init__(self, d_hidden, n=4):
        super().__init__()
        self.n = n
        nd = n * d_hidden
        self.norm = _RMSNorm(nd)
        self.phi_pre  = nn.Linear(nd, n, bias=False)
        self.phi_post = nn.Linear(nd, n, bias=False)
        self.phi_res  = nn.Linear(nd, n * n, bias=False)
        self.b_pre  = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res  = nn.Parameter(torch.zeros(n, n))
        self.alpha_pre  = nn.Parameter(torch.tensor(1.0))
        self.alpha_post = nn.Parameter(torch.tensor(1.0))
        self.alpha_res = nn.Parameter(torch.tensor(1.0))

    def forward(self, stream):
        """stream: (B, n, T, d) → H_res(B,T,n,n), H_pre(B,T,n), H_post(B,T,n)"""
        B, n, T, d = stream.shape
        x = self.norm(stream.transpose(1, 2).reshape(B, T, n * d))

        H_pre = torch.sigmoid(self.alpha_pre * self.phi_pre(x) + self.b_pre)
        H_post = 2 * torch.sigmoid(self.alpha_post * self.phi_post(x) + self.b_post)
        H_res_raw = self.alpha_res * self.phi_res(x).reshape(B, T, self.n, self.n) + self.b_res
        H_res = sinkhorn_knopp(H_res_raw)

        return H_res, H_pre, H_post


class BatchedMHC(nn.Module):
    """
    Batched MHC: N independent MHC instances with stacked parameters.
    Computes MHC for a subset of experts in a single batched pass.
    """
    def __init__(self, n_experts, d_hidden, n=4):
        super().__init__()
        self.n = n
        self.n_experts = n_experts
        nd = n * d_hidden
        self.norm_w     = nn.Parameter(torch.ones(n_experts, nd))
        self.phi_pre_w  = nn.Parameter(torch.randn(n_experts, n, nd) * 0.01)
        self.phi_post_w = nn.Parameter(torch.randn(n_experts, n, nd) * 0.01)
        self.phi_res_w  = nn.Parameter(torch.randn(n_experts, n * n, nd) * 0.01)
        self.b_pre      = nn.Parameter(torch.zeros(n_experts, n))
        self.b_post     = nn.Parameter(torch.zeros(n_experts, n))
        self.b_res      = nn.Parameter(torch.zeros(n_experts, n, n))
        self.alpha_pre  = nn.Parameter(torch.ones(n_experts))
        self.alpha_post = nn.Parameter(torch.ones(n_experts))
        self.alpha_res  = nn.Parameter(torch.ones(n_experts))

    def forward(self, stream, active_idx):
        """
        stream:     (B, n, T, d)
        active_idx: (K,) long — expert indices to compute
        Returns:    H_res (K,B,T,n,n), H_pre (K,B,T,n), H_post (K,B,T,n)
        """
        K = active_idx.shape[0]
        B, n, T, d = stream.shape
        nd = n * d

        # (B, n, T, d) → (B, T, nd) → expand (K, B, T, nd)
        x = stream.transpose(1, 2).reshape(B, T, nd)
        x = x.unsqueeze(0).expand(K, -1, -1, -1)

        # Gather params for active experts
        nw = self.norm_w[active_idx]            # (K, nd)

        # Batched RMSNorm
        rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-8).rsqrt().to(x.dtype)
        x = x * rms * nw.unsqueeze(1).unsqueeze(1)

        # phi_pre: (K, B, T, nd) @ (K, n, nd) → (K, B, T, n)
        H_pre = torch.sigmoid(
            self.alpha_pre[active_idx].view(K, 1, 1, 1)
            * _einsum('kbtd,knd->kbtn', x, self.phi_pre_w[active_idx])
            + self.b_pre[active_idx].unsqueeze(1).unsqueeze(1))

        H_post = 2 * torch.sigmoid(
            self.alpha_post[active_idx].view(K, 1, 1, 1)
            * _einsum('kbtd,knd->kbtn', x, self.phi_post_w[active_idx])
            + self.b_post[active_idx].unsqueeze(1).unsqueeze(1))

        res_einsum = _einsum('kbtd,knd->kbtn', x, self.phi_res_w[active_idx])
        res_flat = res_einsum.reshape(K, B, T, n, n)
        H_res_raw = (
            self.alpha_res[active_idx].view(K, 1, 1, 1, 1)
            * res_flat
            + self.b_res[active_idx].unsqueeze(1).unsqueeze(1))
        H_res = sinkhorn_knopp(H_res_raw)

        return H_res, H_pre, H_post


# ────────────────── Compiled KDA step ──────────────────

@torch.compile
def _kda_step(q_t, k_t, v_t, alpha_t, beta_t, S):
    """Single KDA recursion step — compiled as a small graph."""
    aS = alpha_t.unsqueeze(-1) * S
    kt_aS = _einsum('bd,bde->be', k_t, aS)
    bt = beta_t.unsqueeze(-1)
    S = (aS
         - bt * _bmm(k_t.unsqueeze(2), kt_aS.unsqueeze(1))
         + bt * _bmm(k_t.unsqueeze(2), v_t.unsqueeze(1)))
    out_t = _einsum('bd,bde->be', q_t, S)
    return out_t, S


# ────────────────── MoA Attention ──────────────────

class MoAKDALayer(nn.Module):
    """
    Multi-Head MoA-KDA: H heads, each with unified expert pool.
    Expert 0 = zero/no-op for Q; all experts active for KV.
    Per-head top-1 routing for Q and KV independently.
    Batched expert computation with active-pair filtering.
    """
    def __init__(self, d_input, d_key=16, d_head=48,
                 n_heads=8, n_experts_per_head=6,
                 n_mhc=4):
        super().__init__()
        self.dk = d_key
        self.dv = d_head
        self.dk_pope = d_key * 2
        self.H = n_heads
        self.E = n_experts_per_head
        self.n_mhc = n_mhc
        r = max(d_key // 4, 1)                          # LoRA rank

        # ── Positional encoding ──
        self.register_buffer(
            'freqs', 10000.0 ** (torch.arange(d_key).float() / d_key))
        self.pope_delta_raw = nn.Parameter(torch.zeros(d_key))

        # ── Per-head base projections (stacked) ──
        self.W_q_w = nn.Parameter(torch.randn(n_heads, d_key, d_input) * 0.01)
        self.W_k_w = nn.Parameter(torch.randn(n_heads, d_key, d_input) * 0.01)
        self.W_v_w = nn.Parameter(torch.randn(n_heads, self.dv, d_input) * 0.01)

        # ── Per-head router (unified Q+KV) ──
        self.router_w = nn.Parameter(torch.randn(n_heads, n_experts_per_head, d_input) * 0.01)

        # ── Unified expert params (flat: idx = h*E + e) ──
        HE = n_heads * n_experts_per_head
        self.lora_A_q = nn.Parameter(torch.randn(HE, d_input, r) * 0.01)
        self.lora_B_q = nn.Parameter(torch.zeros(HE, r, d_key))
        self.lora_A_k = nn.Parameter(torch.randn(HE, d_input, r) * 0.01)
        self.lora_B_k = nn.Parameter(torch.zeros(HE, r, d_key))
        self.lora_A_v = nn.Parameter(torch.randn(HE, d_input, r) * 0.01)
        self.lora_B_v = nn.Parameter(torch.zeros(HE, r, self.dv))

        d_alpha = int(d_key * 1.618)
        self.alpha_up_w   = nn.Parameter(torch.randn(HE, d_input, d_alpha) * 0.01)
        self.alpha_down_w = nn.Parameter(torch.randn(HE, d_alpha, self.dk_pope) * 0.01)
        self.beta_up_w    = nn.Parameter(torch.randn(HE, d_input, d_alpha) * 0.01)
        self.beta_down_w  = nn.Parameter(torch.randn(HE, d_alpha, 1) * 0.01)

        # ── Batched MHC + norm ──
        self.batched_mhc = BatchedMHC(HE, d_input, n_mhc)
        self.batched_norm_w = nn.Parameter(torch.ones(HE, d_input))

        # ── Output path: per-head preGate(KV) → per-expert W_o → per-head postGate(Q) ──
        self.W_pre_w = nn.Parameter(torch.randn(HE, d_input, self.dv) * 0.01)
        self.W_o_w   = nn.Parameter(torch.randn(HE, d_input, self.dv) * 0.01)
        d_pg = max(int(d_input * 0.618), 1)
        self.W_pg1_w = nn.Parameter(torch.randn(HE, d_input, d_pg) * 0.01)
        self.W_pg2_w = nn.Parameter(torch.randn(HE, d_pg, d_input) * 0.01)

    def _apply_pope(self, x, positions, is_query):
        mu = F.softplus(x)
        phi = positions.unsqueeze(1) * self.freqs.to(x.device).unsqueeze(0)
        if not is_query:
            phi = phi + (-2 * math.pi * torch.sigmoid(self.pope_delta_raw))
        real = mu * torch.cos(phi)
        imag = mu * torch.sin(phi)
        return torch.cat([real, imag], dim=-1)

    @torch.compiler.disable
    def _kda_recursion(self, q, k, v, alpha, beta, T):
        B = q.shape[0]
        S = q.new_zeros(B, self.dk_pope, self.dv)
        out = q.new_empty(B, T, self.dv)
        for t in range(T):
            attn_out, S = _kda_step(q[:, t], k[:, t], v[:, t],
                                    alpha[:, t], beta[:, t], S)
            out[:, t] = attn_out
        return out

    def _compute_active_pairs(self, stream, active_idx, active_h, active_e,
                              sel, probs, positions,
                              q_h_buf, k_h_buf, v_h_buf, alpha_buf, beta_buf,
                              acc_pre_gate, acc_post_gate):
        """Expert computation for a subset of active (h,e) pairs (e > 0 only).
        Scatter-adds into provided accumulators.
        Returns (H_res_contrib, H_post_contrib) for mHC accumulation.
        """
        B, n, T, d = stream.shape
        nonzero = (active_e > 0)
        if not nonzero.any():
            return stream.new_zeros(B, T, n, n), stream.new_zeros(B, T, n)

        ki = nonzero.nonzero(as_tuple=True)[0]
        hi = active_h[ki]
        ei = active_e[ki]
        fi = active_idx[ki]
        K = ki.shape[0]

        H_res_k, H_pre_k, H_post_k = self.batched_mhc(stream, fi)
        stream_exp = stream.unsqueeze(0).expand(K, -1, -1, -1, -1)
        h_e = _einsum('kbtn,kbntd->kbtd', H_pre_k, stream_exp)

        # Batched RMSNorm
        nw = self.batched_norm_w[fi]
        rms = h_e.float().pow(2).mean(-1, keepdim=True).add(1e-8).rsqrt().to(h_e.dtype)
        h_e = h_e * rms * nw.unsqueeze(1).unsqueeze(1)

        # STE mask (unified)
        mask_k = (sel[hi] == ei.unsqueeze(1).unsqueeze(2))
        prob_k = probs[hi, :, :, ei]
        eff = (prob_k + (mask_k.to(prob_k.dtype) - prob_k).detach()).unsqueeze(-1)

        # ── Q projection ──
        lA_q = self.lora_A_q[fi]; lB_q = self.lora_B_q[fi]
        delta_q = _einsum(
            'kbtr,krd->kbtd',
            F.silu(_einsum('kbtd,kdr->kbtr', h_e, lA_q)), lB_q)
        Wq = self.W_q_w[hi]
        q_proj = _einsum('kbtd,kod->kbto', h_e, Wq)
        q_out = self._apply_pope(
            F.normalize(q_proj + delta_q, dim=-1), positions, True)

        # ── K projection ──
        lA_k = self.lora_A_k[fi]; lB_k = self.lora_B_k[fi]
        delta_k = _einsum(
            'kbtr,krd->kbtd',
            F.silu(_einsum('kbtd,kdr->kbtr', h_e, lA_k)), lB_k)
        Wk = self.W_k_w[hi]
        k_proj = _einsum('kbtd,kod->kbto', h_e, Wk)
        k_out = self._apply_pope(
            F.normalize(k_proj + delta_k, dim=-1), positions, False)

        # ── V projection ──
        lA_v = self.lora_A_v[fi]; lB_v = self.lora_B_v[fi]
        delta_v = _einsum(
            'kbtr,krd->kbtd',
            F.silu(_einsum('kbtd,kdr->kbtr', h_e, lA_v)), lB_v)
        Wv = self.W_v_w[hi]
        v_out = F.silu(
            _einsum('kbtd,kod->kbto', h_e, Wv) + delta_v)

        # ── Alpha, Beta ──
        a_mid = _einsum('kbtd,kde->kbte', h_e, self.alpha_up_w[fi])
        alpha_out = torch.sigmoid(_einsum(
            'kbte,ked->kbtd', F.silu(a_mid), self.alpha_down_w[fi]))
        b_mid = _einsum('kbtd,kde->kbte', h_e, self.beta_up_w[fi])
        beta_out = torch.sigmoid(_einsum(
            'kbte,ked->kbtd', F.silu(b_mid), self.beta_down_w[fi]))

        # ── Gates ──
        pre_gate_out = F.silu(_einsum(
            'kbtd,kde->kbte', h_e, self.W_pre_w[fi]))
        pg_mid = _einsum('kbtd,kde->kbte', h_e, self.W_pg1_w[fi])
        pg_out = torch.sigmoid(_einsum(
            'kbte,ked->kbtd', F.silu(pg_mid), self.W_pg2_w[fi]))

        # ── Scatter-add to per-head buffers ──
        idx_h = hi.view(-1, 1, 1, 1)
        eff_q = eff * q_out
        eff_k = eff * k_out
        eff_v = eff * v_out
        eff_a = eff * alpha_out
        eff_b = eff * beta_out
        eff_pg = eff * pre_gate_out
        eff_post = eff * pg_out
        q_h_buf.scatter_add_(0, idx_h.expand_as(eff_q), eff_q)
        k_h_buf.scatter_add_(0, idx_h.expand_as(eff_k), eff_k)
        v_h_buf.scatter_add_(0, idx_h.expand_as(eff_v), eff_v)
        alpha_buf.scatter_add_(0, idx_h.expand_as(eff_a), eff_a)
        beta_buf.scatter_add_(0, idx_h.expand_as(eff_b), eff_b)
        acc_pre_gate.scatter_add_(0, idx_h.expand_as(eff_pg), eff_pg)
        acc_post_gate.scatter_add_(0, idx_h.expand_as(eff_post), eff_post)

        H_res_contrib = (H_res_k * eff.unsqueeze(-1)).sum(0)
        H_post_contrib = (H_post_k * eff).sum(0)
        return H_res_contrib, H_post_contrib

    def forward(self, stream, head_grad_mask=None):
        """
        stream: (B, n_mhc, T, d)
        head_grad_mask: (H,) bool — True = gradient on this GPU, None = all
        Returns: stream_update (B, n_mhc, T, d), route_lp (B, T)
        """
        B, n, T, d = stream.shape
        H, E = self.H, self.E
        positions = torch.arange(T, device=stream.device)
        route_input = stream.mean(dim=1)                   # (B, T, d)

        # ════════════════ 1. Unified routing (all heads, always with grad) ════════════════
        logits = _einsum('btd,hed->hbte', route_input, self.router_w)
        logits = logits.clamp(-10, 10)                      # (H, B, T, E)

        probs = F.softmax(logits / ROUTE_TEMP, dim=-1)

        if self.training:
            sel = torch.multinomial(probs.reshape(-1, E), 1).reshape(H, B, T)
        else:
            sel = logits.argmax(dim=-1)                      # (H, B, T)

        # Head Dropout: randomly force zero expert → skip computation
        if self.training and HEAD_DROP_PROB > 0:
            head_drop = torch.rand(H, 1, 1, device=stream.device) < HEAD_DROP_PROB
            sel = sel.masked_fill(head_drop.expand_as(sel), 0)

        zero_mask = (sel == 0)                               # (H, B, T)

        route_lp = probs.log().gather(-1, sel.unsqueeze(-1)).squeeze(-1)   # (H, B, T)

        # ════════════════ 2. Identify active (h,e) pairs ════════════════
        e_range = torch.arange(E, device=stream.device)
        active = (sel.unsqueeze(-1) == e_range).any(dim=(1, 2))   # (H, E)
        active_flat = active.reshape(-1)                            # (H*E,)
        active_idx = active_flat.nonzero(as_tuple=True)[0]
        active_h = active_idx // E                                  # (K,)
        active_e = active_idx % E

        # ════════════════ 3. Accumulators ════════════════
        q_h_buf = stream.new_zeros(H, B, T, self.dk_pope)
        k_h_buf = stream.new_zeros(H, B, T, self.dk_pope)
        v_h_buf = stream.new_zeros(H, B, T, self.dv)
        alpha_buf = stream.new_zeros(H, B, T, self.dk_pope)
        beta_buf  = stream.new_zeros(H, B, T, 1)
        acc_pre_gate  = stream.new_zeros(H, B, T, self.dv)
        acc_post_gate = stream.new_zeros(H, B, T, d)
        acc_H_res  = stream.new_zeros(B, T, n, n)
        acc_H_post = stream.new_zeros(B, T, n)

        # ════════════════ 4. Expert computation (split by head mask) ════════════════
        if head_grad_mask is not None and active_idx.numel() > 0:
            assigned = head_grad_mask[active_h]
            if assigned.any():
                hr, hp = self._compute_active_pairs(
                    stream, active_idx[assigned], active_h[assigned], active_e[assigned],
                    sel, probs, positions,
                    q_h_buf, k_h_buf, v_h_buf, alpha_buf, beta_buf,
                    acc_pre_gate, acc_post_gate)
                acc_H_res = acc_H_res + hr
                acc_H_post = acc_H_post + hp
            if (~assigned).any():
                with torch.no_grad():
                    hr, hp = self._compute_active_pairs(
                        stream, active_idx[~assigned], active_h[~assigned], active_e[~assigned],
                        sel, probs, positions,
                        q_h_buf, k_h_buf, v_h_buf, alpha_buf, beta_buf,
                        acc_pre_gate, acc_post_gate)
                acc_H_res = acc_H_res + hr
                acc_H_post = acc_H_post + hp
        elif active_idx.numel() > 0:
            hr, hp = self._compute_active_pairs(
                stream, active_idx, active_h, active_e,
                sel, probs, positions,
                q_h_buf, k_h_buf, v_h_buf, alpha_buf, beta_buf,
                acc_pre_gate, acc_post_gate)
            acc_H_res = acc_H_res + hr
            acc_H_post = acc_H_post + hp

        # Route log probs: only assigned heads (avoid double-counting across GPUs)
        if head_grad_mask is not None:
            mask_h = head_grad_mask.float().view(H, 1, 1)
            route_lp = (route_lp * mask_h).sum(0)
        else:
            route_lp = route_lp.sum(0)

        # ════════════════ 5. Batched KDA (all heads at once) ════════════════
        HB = H * B
        out_flat = self._kda_recursion(
            q_h_buf.reshape(HB, T, self.dk_pope),
            k_h_buf.reshape(HB, T, self.dk_pope),
            v_h_buf.reshape(HB, T, self.dv),
            alpha_buf.reshape(HB, T, self.dk_pope),
            beta_buf.reshape(HB, T, 1), T)
        out_he = out_flat.reshape(H, B, T, self.dv)

        # Zero out where expert 0 selected
        out_he[zero_mask] = 0.0

        # ════════════════ 6. Output assembly ════════════════
        gated_he = out_he * acc_pre_gate                    # (H, B, T, dv)
        prob_h = probs.gather(-1, sel.unsqueeze(-1)).squeeze(-1)
        gated_he = gated_he * prob_h.unsqueeze(-1)         # × routing prob
        h_idx = torch.arange(H, device=stream.device).view(H, 1, 1)
        flat_idx = h_idx * E + sel                          # (H, B, T)
        W_o_token = self.W_o_w[flat_idx]                   # (H, B, T, d, dv)
        proj_he = _einsum('hbtv,hbtdv->hbtd', gated_he, W_o_token)
        result_he = proj_he * acc_post_gate                # (H, B, T, d)
        result = result_he.mean(0)                          # (B, T, d)

        H_res_avg  = acc_H_res / H
        H_post_avg = acc_H_post / H
        res = _einsum('btij,bjtd->bitd', H_res_avg, stream)
        post = _einsum('btn,btd->bntd', H_post_avg, result)
        stream_update = res + post

        return stream_update, route_lp


# ────────────────── SwiGLU (building block) ──────────────────

class SwiGLU(nn.Module):
    """SwiGLU with bottleneck gate and RMSNorm."""
    def __init__(self, d_input):
        super().__init__()
        d_ffn = int(d_input * 1.618)
        self.norm = _RMSNorm(d_input)
        self.wd = nn.Linear(d_input, d_input, bias=False)
        self.wu = nn.Linear(d_input, d_input, bias=False)
        self.gate = nn.Linear(d_input, d_ffn, bias=False)
        self.up   = nn.Linear(d_input, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_input, bias=False)

    def forward(self, x):
        h = self.norm(x)
        g = torch.sigmoid(self.wu(F.silu(self.wd(h))))
        return g * self.down(F.silu(self.gate(h)) * self.up(h))


# ────────────────── MoE Expert ──────────────────

class MoESwiGLU(nn.Module):
    """
    MoE-SwiGLU with per-head routing, mirroring MoA-KDA structure.
    F heads, each with Ef experts (including zero expert 0).
    Per-head top-1 routing. Output: per-head preGate → per-expert W_o → per-head postGate.
    """
    def __init__(self, d_hidden, n_heads=6, n_experts_per_head=8, n_mhc=4, d_head=None):
        super().__init__()
        self.nf = n_heads
        self.ne = n_experts_per_head
        FE = n_heads * n_experts_per_head
        self.FE = FE
        self.d = d_hidden
        self.dh = d_head if d_head is not None else d_hidden
        self.n_mhc = n_mhc

        # Routing: per-head stacked router
        self.router_w = nn.Parameter(torch.randn(n_heads, n_experts_per_head, d_hidden) * 0.01)

        # MHC + norm
        self.batched_mhc = BatchedMHC(FE, d_hidden, n_mhc)
        self.batched_norm_w = nn.Parameter(torch.ones(FE, d_hidden))

        # SwiGLU
        d_ffn = int(d_hidden * 1.618)
        self.swiglu_wd_w   = nn.Parameter(torch.randn(FE, d_hidden, d_hidden) * 0.01)
        self.swiglu_wu_w   = nn.Parameter(torch.randn(FE, d_hidden, d_hidden) * 0.01)
        self.swiglu_gate_w = nn.Parameter(torch.randn(FE, d_ffn, d_hidden) * 0.01)
        self.swiglu_up_w   = nn.Parameter(torch.randn(FE, d_ffn, d_hidden) * 0.01)
        self.swiglu_down_w = nn.Parameter(torch.randn(FE, self.dh, d_ffn) * 0.01)

        # Output: per-head preGate(d_hidden→d_head) → per-expert W_o(d_head→d_hidden) → per-head postGate
        self.W_pre_w = nn.Parameter(torch.randn(FE, self.dh, d_hidden) * 0.01)
        self.W_o_w   = nn.Parameter(torch.randn(FE, d_hidden, self.dh) * 0.01)
        d_pg = max(int(d_hidden * 0.618), 1)
        self.W_pg1_w = nn.Parameter(torch.randn(FE, d_hidden, d_pg) * 0.01)
        self.W_pg2_w = nn.Parameter(torch.randn(FE, d_pg, d_hidden) * 0.01)

        # Expert embeddings for AttnRes query
        self.w_experts = nn.Parameter(torch.randn(FE, d_hidden) * 0.01)

    def _compute_active_pairs(self, stream, active_idx, active_f, active_ef,
                              sel, probs,
                              out_buf, acc_pre_gate, acc_post_gate):
        """Expert computation for a subset of active (f,e) pairs (non-zero only).
        Scatter-adds into provided accumulators.
        Returns (H_res_contrib, H_post_contrib) for mHC accumulation.
        """
        B, n, T, d = stream.shape
        # Filter to non-zero experts
        nonzero = (active_ef > 0)
        if not nonzero.any():
            return stream.new_zeros(B, T, n, n), stream.new_zeros(B, T, n)

        ki = nonzero.nonzero(as_tuple=True)[0]
        f_hi = active_f[ki]
        ef_ei = active_ef[ki]
        fi = active_idx[ki]

        H_res_k, H_pre_k, H_post_k = self.batched_mhc(stream, fi)
        stream_exp = stream.unsqueeze(0).expand(ki.shape[0], -1, -1, -1, -1)
        h_e = _einsum('kbtn,kbntd->kbtd', H_pre_k, stream_exp)

        nw = self.batched_norm_w[fi]
        rms = h_e.float().pow(2).mean(-1, keepdim=True).add(1e-8).rsqrt().to(h_e.dtype)
        h_e = h_e * rms * nw.unsqueeze(1).unsqueeze(1)

        # STE mask
        mask_k = (sel[f_hi] == ef_ei.unsqueeze(1).unsqueeze(2))
        prob_k = probs[f_hi, :, :, ef_ei]
        eff = (prob_k + (mask_k.to(prob_k.dtype) - prob_k).detach()).unsqueeze(-1)

        # SwiGLU
        wd_out = _einsum('kbtd,kod->kbto', h_e, self.swiglu_wd_w[fi])
        g = torch.sigmoid(_einsum('kbtd,kod->kbto', F.silu(wd_out), self.swiglu_wu_w[fi]))
        gate_out = _einsum('kbtd,kod->kbto', h_e, self.swiglu_gate_w[fi])
        up_out = _einsum('kbtd,kod->kbto', h_e, self.swiglu_up_w[fi])
        swiglu_out = g * _einsum('kbtf,kof->kbto', F.silu(gate_out) * up_out, self.swiglu_down_w[fi])

        eff_out = eff * swiglu_out

        # PreGate
        pre_gate_out = F.silu(_einsum('kbtd,kde->kbte', h_e, self.W_pre_w[fi]))
        eff_pg = eff * pre_gate_out

        # PostGate
        pg_mid = _einsum('kbtd,kde->kbte', h_e, self.W_pg1_w[fi])
        pg_out = torch.sigmoid(_einsum('kbte,ked->kbtd', F.silu(pg_mid), self.W_pg2_w[fi]))
        eff_post = eff * pg_out

        # Scatter-accumulate to per-head buffers
        idx_f = f_hi.view(-1, 1, 1, 1)
        out_buf.scatter_add_(0, idx_f.expand_as(eff_out), eff_out)
        acc_pre_gate.scatter_add_(0, idx_f.expand_as(eff_pg), eff_pg)
        acc_post_gate.scatter_add_(0, idx_f.expand_as(eff_post), eff_post)

        H_res_contrib = (H_res_k * eff.unsqueeze(-1)).sum(0)
        H_post_contrib = (H_post_k * eff).sum(0)
        return H_res_contrib, H_post_contrib

    @torch.compiler.disable
    def forward(self, stream, head_grad_mask=None):
        """
        stream: (B, n_mhc, T, d)
        head_grad_mask: (nf,) bool — True = gradient on this GPU, None = all
        Returns: stream_update (B, n_mhc, T, d), gate_moe (B, T, FE), route_lp (B, T)
        """
        B, n, T, d = stream.shape
        nf, ne, FE = self.nf, self.ne, self.FE
        route_input = stream.mean(dim=1)                   # (B, T, d)

        # ════════════════ 1. Routing (all heads, always with grad) ════════════════
        logits = _einsum('btd,fed->fbte', route_input, self.router_w).clamp(-10, 10)
        probs = F.softmax(logits / ROUTE_TEMP, dim=-1)     # (nf, B, T, ne)

        if self.training:
            sel = torch.multinomial(probs.reshape(-1, ne), 1).reshape(nf, B, T)
        else:
            sel = logits.argmax(dim=-1)                     # (nf, B, T)

        # Head Dropout: randomly force zero expert → skip computation
        if self.training and HEAD_DROP_PROB > 0:
            head_drop = torch.rand(nf, 1, 1, device=stream.device) < HEAD_DROP_PROB
            sel = sel.masked_fill(head_drop.expand_as(sel), 0)

        zero_mask = (sel == 0)                              # (nf, B, T)

        # Routing log probs (per-head, masked later)
        route_lp_full = probs.log().gather(-1, sel.unsqueeze(-1)).squeeze(-1)  # (nf, B, T)

        # gate_moe for AttnRes: (B, T, FE)
        f_arange = torch.arange(nf, device=stream.device).view(nf, 1, 1)
        flat_sel = f_arange * ne + sel                     # (nf, B, T)
        sel_probs = probs.gather(-1, sel.unsqueeze(-1)).squeeze(-1)  # (nf, B, T)
        gate_moe = stream.new_zeros(B * T, FE)
        gate_moe.scatter_(-1,
                          flat_sel.permute(1, 2, 0).reshape(B * T, nf),
                          sel_probs.permute(1, 2, 0).reshape(B * T, nf))
        gate_moe = gate_moe.reshape(B, T, FE)

        # ════════════════ 2. Active pairs ════════════════
        e_range = torch.arange(ne, device=stream.device)
        active = (sel.unsqueeze(-1) == e_range).any(dim=(1, 2))  # (nf, ne)
        active_flat = active.reshape(-1)
        active_idx = active_flat.nonzero(as_tuple=True)[0]
        active_f = active_idx // ne
        active_ef = active_idx % ne

        # ════════════════ 3. Accumulators ════════════════
        dh = self.dh
        out_buf = stream.new_zeros(nf, B, T, dh)
        acc_pre_gate = stream.new_zeros(nf, B, T, dh)
        acc_post_gate = stream.new_zeros(nf, B, T, d)
        acc_H_res = stream.new_zeros(B, T, n, n)
        acc_H_post = stream.new_zeros(B, T, n)

        # ════════════════ 4. Expert computation (split by head mask) ════════════════
        if head_grad_mask is not None and active_idx.numel() > 0:
            assigned = head_grad_mask[active_f]
            if assigned.any():
                hr, hp = self._compute_active_pairs(
                    stream, active_idx[assigned], active_f[assigned], active_ef[assigned],
                    sel, probs,
                    out_buf, acc_pre_gate, acc_post_gate)
                acc_H_res = acc_H_res + hr
                acc_H_post = acc_H_post + hp
            if (~assigned).any():
                with torch.no_grad():
                    hr, hp = self._compute_active_pairs(
                        stream, active_idx[~assigned], active_f[~assigned], active_ef[~assigned],
                        sel, probs,
                        out_buf, acc_pre_gate, acc_post_gate)
                # 累加必须在 no_grad 外：保持 assigned heads 的梯度链
                acc_H_res = acc_H_res + hr
                acc_H_post = acc_H_post + hp
        elif active_idx.numel() > 0:
            hr, hp = self._compute_active_pairs(
                stream, active_idx, active_f, active_ef,
                sel, probs,
                out_buf, acc_pre_gate, acc_post_gate)
            acc_H_res = acc_H_res + hr
            acc_H_post = acc_H_post + hp

        # Route log probs: only assigned heads
        if head_grad_mask is not None:
            mask_f = head_grad_mask.float().view(nf, 1, 1)
            route_lp = (route_lp_full * mask_f).sum(0)
        else:
            route_lp = route_lp_full.sum(0)

        # ════════════════ 5. Output assembly ════════════════
        out_buf[zero_mask] = 0.0

        # Per-head preGate → × prob → per-expert W_o → per-head postGate
        gated = out_buf * acc_pre_gate                     # (nf, B, T, d)
        gated = gated * sel_probs.unsqueeze(-1)            # × routing prob
        flat_idx = f_arange * ne + sel                     # (nf, B, T)
        W_o_token = self.W_o_w[flat_idx]                   # (nf, B, T, d, d)
        proj = _einsum('hbtd,hbted->hbte', gated, W_o_token)  # (nf, B, T, d)
        result_he = proj * acc_post_gate                   # (nf, B, T, d)
        result = result_he.mean(0)                          # (B, T, d)

        # mHC
        H_res_avg = acc_H_res / nf
        H_post_avg = acc_H_post / nf
        res = _einsum('btij,bjtd->bitd', H_res_avg, stream)
        post = _einsum('btn,btd->bntd', H_post_avg, result)
        stream_update = res + post

        return stream_update, gate_moe, route_lp


# ────────────────── Halting Router ──────────────────

class HaltingRouter(nn.Module):
    """Per-layer halting router: mHC H_pre aggregation → SwiGLU → Linear → continue/exit."""
    def __init__(self, d_hidden, n_mhc):
        super().__init__()
        self.n = n_mhc
        nd = n_mhc * d_hidden
        self.norm = _RMSNorm(nd)
        self.phi_pre = nn.Linear(nd, n_mhc, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(n_mhc))
        self.swiglu = SwiGLU(d_hidden)
        self.proj = nn.Linear(d_hidden, 2, bias=False)

    def forward(self, stream):
        """
        stream: (B, n, T, d)
        Returns: logits (B, 2) — [continue, exit]
        """
        B, n, T, d = stream.shape
        last = stream[:, :, -1, :]                         # (B, n, d)
        x = self.norm(last.reshape(B, n * d))              # (B, n*d)
        H_pre = torch.sigmoid(self.phi_pre(x) + self.b_pre)  # (B, n)
        x = (last * H_pre.unsqueeze(-1)).sum(dim=1)        # (B, d)
        x = self.swiglu(x)                                 # (B, d)
        logits = self.proj(x)                              # (B, 2)
        return logits


# ────────────────── Main Network ──────────────────

class KDAPolicyNetwork(nn.Module):
    """
    Dynamic-depth MoA-KDA + MoE-SwiGLU with mHC, AttnRes, halting router.

    Shared single MoA+MoE block looped dynamically.  Per-layer router
    decides continue/exit.  Training explores via pre-determined target depths.
    """
    def __init__(self, d_input=14, d_hidden=D_HIDDEN, d_key=D_KEY,
                 d_attn_head=D_ATTN_HEAD, d_ffn_head=D_FFN_HEAD,
                 n_actions=11,
                 max_iterations=MAX_ITERATIONS, inner_steps=INNER_STEPS,
                 n_mhc=4, min_iterations=MIN_ITERATIONS,
                 n_heads=N_ATTN_HEADS,
                 n_experts_per_head=N_EXPERTS_PER_HEAD,
                 n_ffn_heads=N_FFN_HEADS,
                 n_experts_per_ffn_head=N_EXPERTS_PER_FFN_HEAD):
        super().__init__()
        self.max_iterations = max_iterations
        self.inner_steps = inner_steps
        self.min_iterations = min_iterations
        self.n_mhc = n_mhc
        self.n_heads = n_heads
        self.d = d_hidden

        self.inp_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), _RMSNorm(d_hidden))

        # MoA-KDA + MoE-SwiGLU
        self.moa_kda = MoAKDALayer(d_hidden, d_key=d_key, d_head=d_attn_head,
                                    n_heads=n_heads,
                                    n_experts_per_head=n_experts_per_head,
                                    n_mhc=n_mhc)
        self.moe_swiglu = MoESwiGLU(d_hidden, n_heads=n_ffn_heads,
                                     n_experts_per_head=n_experts_per_ffn_head, n_mhc=n_mhc,
                                     d_head=d_ffn_head)

        # Halting router
        self.router = HaltingRouter(d_hidden, n_mhc)

        # AttnRes: shared preH + final w + expert-averaged RMSNorm
        self.attn_res_expert_scales = nn.Parameter(torch.ones(n_ffn_heads * n_experts_per_ffn_head, d_hidden))
        self.preH_iter  = nn.Linear(n_mhc * d_hidden, d_hidden, bias=False)
        self.preH_final = nn.Linear(n_mhc * d_hidden, d_hidden, bias=False)
        self.w_final = nn.Parameter(torch.zeros(d_hidden))

        # Head
        d_head = int(d_hidden * 1.618)
        self.head_gate = nn.Linear(d_hidden, d_head, bias=False)
        self.head_up   = nn.Linear(d_hidden, d_head, bias=False)
        self.head_down = nn.Linear(d_head, n_actions, bias=False)

        self._init_weights()

    def _init_weights(self):
        depth = self.max_iterations * self.inner_steps
        scale = depth ** -0.5
        nn.init.normal_(self.moa_kda.W_o_w,
                        std=scale * (2.0 / self.moa_kda.dv) ** 0.5)
        nn.init.normal_(self.moe_swiglu.swiglu_down_w,
                        std=scale * (2.0 / self.moe_swiglu.swiglu_down_w.shape[-1]) ** 0.5)
        nn.init.normal_(self.moe_swiglu.W_o_w,
                        std=scale * (2.0 / self.moe_swiglu.d) ** 0.5)
        nn.init.normal_(self.head_down.weight,
                        std=scale * (2.0 / self.head_down.in_features) ** 0.5)
        nn.init.normal_(self.preH_iter.weight,
                        std=(2.0 / self.preH_iter.weight.shape[1]) ** 0.5)
        nn.init.normal_(self.preH_final.weight,
                        std=(2.0 / self.preH_final.weight.shape[1]) ** 0.5)

    @torch.compiler.disable
    def _attn_res(self, acc, w_l, pre_h, gate_moe=None):
        """AttnRes: project streams → expert-averaged RMSNorm → softmax weighted sum."""
        B, n, T, d = acc[0].shape
        nd = n * d
        stacked = torch.stack(acc)                          # (L, B, n, T, d)
        L = stacked.shape[0]
        inp = stacked.permute(0, 1, 3, 2, 4).reshape(L, B, T, nd)
        V = pre_h(inp)                                     # (L, B, T, d)
        # Expert-averaged RMSNorm
        rms = V.pow(2).mean(-1, keepdim=True).sqrt()
        if gate_moe is None:
            scale = self.attn_res_expert_scales.mean(0)   # (d,) uniform avg
        else:
            weights = gate_moe.sum(-1, keepdim=True) + 1e-8
            scale = (gate_moe @ self.attn_res_expert_scales) / weights  # (B,T,d)
        K = V / (rms + 1e-8) * scale
        if w_l.dim() == 1:
            logits = _einsum('d,lbtd->lbt', w_l, K)
        else:                                             # (B, T, d) per-token
            logits = _einsum('btd,lbtd->lbt', w_l, K)
        alpha = logits.softmax(0)
        return _einsum('lbt,lbtd->btd', alpha, V)

    @torch.compiler.disable
    def _loop(self, stream, acc, attn_head_mask=None, ffn_head_mask=None):
        """
        Outer loop: MAX_ITERATIONS halting points.
        Inner loop: INNER_STEPS fixed MoA+MoE per outer step.
        KDA state S 只在单次 _kda_recursion 内跨 timestep 传递（50 步），
        不在迭代间传递。

        Training: STE with random target_depths for exploration.
        Returns (stream, acc, exit_iter, route_lp, expected_depth).
        """
        B, n, T_total, d = stream.shape

        if self.training:
            target_depths = torch.randint(
                self.min_iterations, self.max_iterations + 1,
                (B,), device=stream.device)

        active = torch.ones(B, dtype=torch.bool, device=stream.device)
        exit_iter = torch.full((B,), float(self.max_iterations), device=stream.device)
        route_lp_total = torch.zeros(B, device=stream.device)
        route_lp_count = 0
        expected_depth = torch.zeros(B, device=stream.device)     # Σ P(continue) per step

        for i in range(self.max_iterations):
            # ── Halting decision ──
            logits = self.router(stream).clamp(-10, 10)           # (B, 2)
            soft = F.softmax(logits / HALT_TEMP, dim=-1)  # (B, 2) [continue, exit]

            # Accumulate expected depth (for compute cost penalty)
            expected_depth = expected_depth + soft[:, 0]

            if self.training:
                should_continue = (i < target_depths).float()
                gate = soft[:, 0] + (should_continue - soft[:, 0]).detach()
            else:
                gate = (logits.argmax(dim=-1) == 0).float()

            if i < self.min_iterations:
                gate = torch.ones(B, device=stream.device)

            gate_4d = gate.view(B, 1, 1, 1)

            stream_pre = stream

            # ── Inner loop: INNER_STEPS × (MoA + MoE) ──
            mean_query = stream.new_zeros(B, T_total, d)
            gate_moe_sum = stream.new_zeros(B, T_total, self.moe_swiglu.FE)
            for j in range(self.inner_steps):
                if GRADIENT_CHECKPOINTING:
                    attn_update, moa_lp = cp.checkpoint(
                        self.moa_kda, stream, attn_head_mask, use_reentrant=False)
                else:
                    attn_update, moa_lp = self.moa_kda(stream, head_grad_mask=attn_head_mask)
                stream = attn_update
                route_lp_total = route_lp_total + moa_lp.sum(-1)
                route_lp_count += 1

                if GRADIENT_CHECKPOINTING:
                    ffn_update, gate_moe, moe_lp = cp.checkpoint(
                        self.moe_swiglu, stream, ffn_head_mask, use_reentrant=False)
                else:
                    ffn_update, gate_moe, moe_lp = self.moe_swiglu(stream, head_grad_mask=ffn_head_mask)
                stream = ffn_update
                route_lp_total = route_lp_total + moe_lp.sum(-1)
                route_lp_count += 1

                mean_query = mean_query + _einsum(
                    'bte,ed->btd', gate_moe, self.moe_swiglu.w_experts)
                gate_moe_sum = gate_moe_sum + gate_moe
            mean_query = mean_query / self.inner_steps
            gate_moe_avg = gate_moe_sum / self.inner_steps

            # AttnRes (once per outer step)
            stream = stream + self._attn_res(
                acc, mean_query, self.preH_iter, gate_moe_avg).unsqueeze(1)

            # Gate: continue → updated, exit → previous
            stream = gate_4d * stream + (1 - gate_4d) * stream_pre

            acc.append(stream)

            # Track exit
            if self.training:
                newly_exited = active & (i + 1 >= target_depths)
            else:
                newly_exited = active & (gate < 0.5)
            exit_iter = torch.where(newly_exited,
                                    torch.tensor(float(i + 1), device=stream.device),
                                    exit_iter)

            if self.training:
                active = active & (i + 1 < target_depths)
            else:
                active = active & (gate > 0.5)

            if not active.any():
                break

        # Normalize route_lp: mean per inner-step call instead of sum
        if route_lp_count > 0:
            route_lp_total = route_lp_total / route_lp_count
        return stream, acc, exit_iter, route_lp_total, expected_depth

    def _head(self, h_out):
        """Output head — compiled separately for speed."""
        return self.head_down(F.silu(self.head_gate(h_out)) * self.head_up(h_out))

    def forward(self, x, attn_head_mask=None, ffn_head_mask=None):
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(0)
        B, T, _ = x.shape
        n = self.n_mhc
        d = self.d

        v0 = self.inp_proj(x)                               # (B, T, d)

        stream = torch.zeros(B, n, T, d, device=x.device)
        stream[:, 0] = v0

        acc = [stream]

        # Dynamic loop (not compiled)
        stream, acc, exit_iter, route_lp, expected_depth = self._loop(
            stream, acc, attn_head_mask, ffn_head_mask)

        h_out = self._attn_res(acc, self.w_final, self.preH_final)  # (B, T, d)
        h_out = h_out.mean(dim=1)                                     # (B, d)
        out = self._head(h_out)                                       # (B, n_actions)
        if squeeze:
            out = out.squeeze(0)
        return out, exit_iter, route_lp, expected_depth


# ────────────────── Checkpoint Migration ──────────────────

def migrate_checkpoint(state_dict):
    """
    Convert old ModuleList-based state_dict to stacked parameter format.
    Maps old keys like 'moa_kda.mhc.{i}.norm.weight' → 'moa_kda.batched_mhc.norm_w'.
    """
    from config import (D_HIDDEN, N_ATTN_HEADS, N_EXPERTS_PER_HEAD,
                        N_FFN_EXPERTS)
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k] = v                                   # pass through by default

    # ── MoA MHC: 'moa_kda.mhc.{i}.{param}' → 'moa_kda.batched_mhc.{param}_w' etc. ──
    HE = N_ATTN_HEADS * N_EXPERTS_PER_HEAD
    mhc_params = {
        'norm.weight': ('batched_mhc.norm_w', lambda vs: torch.stack(vs)),
        'phi_pre.weight': ('batched_mhc.phi_pre_w', lambda vs: torch.stack(vs)),
        'phi_post.weight': ('batched_mhc.phi_post_w', lambda vs: torch.stack(vs)),
        'phi_res.weight': ('batched_mhc.phi_res_w', lambda vs: torch.stack(vs)),
        'b_pre': ('batched_mhc.b_pre', lambda vs: torch.stack(vs)),
        'b_post': ('batched_mhc.b_post', lambda vs: torch.stack(vs)),
        'b_res': ('batched_mhc.b_res', lambda vs: torch.stack(vs)),
        'alpha_pre': ('batched_mhc.alpha_pre', lambda vs: torch.stack(vs)),
        'alpha_post': ('batched_mhc.alpha_post', lambda vs: torch.stack(vs)),
        'alpha_res': ('batched_mhc.alpha_res', lambda vs: torch.stack(vs)),
    }
    for suffix, (new_prefix, stack_fn) in mhc_params.items():
        vals = [state_dict.get(f'moa_kda.mhc.{i}.{suffix}') for i in range(HE)]
        if all(v is not None for v in vals):
            new_sd[f'moa_kda.{new_prefix}'] = stack_fn(vals)
            for i in range(HE):
                new_sd.pop(f'moa_kda.mhc.{i}.{suffix}', None)

    # ── MoA norm: 'moa_kda.norm.{i}.weight' → 'moa_kda.batched_norm_w' ──
    vals = [state_dict.get(f'moa_kda.norm.{i}.weight') for i in range(HE)]
    if all(v is not None for v in vals):
        new_sd['moa_kda.batched_norm_w'] = torch.stack(vals)
        for i in range(HE):
            new_sd.pop(f'moa_kda.norm.{i}.weight', None)

    # ── MoA W_q/k/v: 'moa_kda.W_q.{i}.weight' → 'moa_kda.W_q_w' ──
    H = N_ATTN_HEADS
    for prefix_old, prefix_new in [('W_q', 'W_q_w'), ('W_k', 'W_k_w'), ('W_v', 'W_v_w')]:
        vals = [state_dict.get(f'moa_kda.{prefix_old}.{i}.weight') for i in range(H)]
        if all(v is not None for v in vals):
            new_sd[f'moa_kda.{prefix_new}'] = torch.stack(vals)
            for i in range(H):
                new_sd.pop(f'moa_kda.{prefix_old}.{i}.weight', None)

    # ── MoE MHC: 'moe_swiglu.mhc_swiglu.{i}.{param}' → 'moe_swiglu.batched_mhc.{param}' ──
    E = N_FFN_EXPERTS
    for suffix, (new_prefix, stack_fn) in mhc_params.items():
        vals = [state_dict.get(f'moe_swiglu.mhc_swiglu.{i}.{suffix}') for i in range(E)]
        if all(v is not None for v in vals):
            new_sd[f'moe_swiglu.{new_prefix}'] = stack_fn(vals)
            for i in range(E):
                new_sd.pop(f'moe_swiglu.mhc_swiglu.{i}.{suffix}', None)

    # ── MoE SwiGLU: 'moe_swiglu.experts.{i}.{param}' → 'moe_swiglu.swiglu_{param}' ──
    swiglu_params = {
        'norm.weight': 'swiglu_norm_w',
        'wd.weight': 'swiglu_wd_w',
        'wu.weight': 'swiglu_wu_w',
        'gate.weight': 'swiglu_gate_w',
        'up.weight': 'swiglu_up_w',
        'down.weight': 'swiglu_down_w',
    }
    for suffix_old, suffix_new in swiglu_params.items():
        vals = [state_dict.get(f'moe_swiglu.experts.{i}.{suffix_old}') for i in range(E)]
        if all(v is not None for v in vals):
            new_sd[f'moe_swiglu.{suffix_new}'] = torch.stack(vals)
            for i in range(E):
                new_sd.pop(f'moe_swiglu.experts.{i}.{suffix_old}', None)

    return new_sd


if __name__ == "__main__":
    model = KDAPolicyNetwork()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    for name, mod in [("MoA-KDA", model.moa_kda), ("MoE-SwiGLU", model.moe_swiglu),
                      ("Router", model.router), ("AttnRes", None)]:
        if mod is not None:
            n = sum(p.numel() for p in mod.parameters())
        else:
            n = sum(p.numel() for n, p in model.named_parameters()
                    if any(n.startswith(pfx) for pfx in ("preH_", "preH_iter", "preH_final",
                                                          "w_final", "attn_res_expert")))
        print(f"  {name}: {n:,} ({n/total*100:.1f}%)")