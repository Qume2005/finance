"""Mini-KDA Policy Network with PoPE, mHC, AttnRes, MoA, MoE, dynamic loop."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (D_HIDDEN, D_KEY, D_ATTN, N_ATTN_HEADS, N_EXPERTS_PER_HEAD,
                    N_FFN_EXPERTS, FFN_TOP_PROB, FFN_MAX_K,
                    MAX_ITERATIONS, MIN_ITERATIONS, INNER_STEPS,
                    ROUTE_TEMP, FFN_GUMBEL_TAU, HALT_TEMP)


# ────────────────── Utilities ──────────────────

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
        self.norm = nn.RMSNorm(nd)
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
            * torch.einsum('kbtd,knd->kbtn', x, self.phi_pre_w[active_idx])
            + self.b_pre[active_idx].unsqueeze(1).unsqueeze(1))

        H_post = 2 * torch.sigmoid(
            self.alpha_post[active_idx].view(K, 1, 1, 1)
            * torch.einsum('kbtd,knd->kbtn', x, self.phi_post_w[active_idx])
            + self.b_post[active_idx].unsqueeze(1).unsqueeze(1))

        res_einsum = torch.einsum('kbtd,knd->kbtn', x, self.phi_res_w[active_idx])
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
    kt_aS = torch.einsum('bd,bde->be', k_t, aS)
    bt = beta_t.unsqueeze(-1)
    S = (aS
         - bt * torch.bmm(k_t.unsqueeze(2), kt_aS.unsqueeze(1))
         + bt * torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1)))
    out_t = torch.einsum('bd,bde->be', q_t, S)
    return out_t, S


# ────────────────── MoA Attention ──────────────────

class MoAKDALayer(nn.Module):
    """
    Multi-Head MoA-KDA: H heads, each with unified expert pool.
    Expert 0 = zero/no-op for Q; all experts active for KV.
    Per-head top-1 routing for Q and KV independently.
    Batched expert computation with active-pair filtering.
    """
    def __init__(self, d_input, d_key=16, d_attn=48,
                 n_heads=8, n_experts_per_head=6,
                 n_mhc=4):
        super().__init__()
        self.dk = d_key
        self.dv = d_attn // n_heads
        assert d_attn % n_heads == 0, f"d_attn={d_attn} must be divisible by n_heads={n_heads}"
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

        # ── Per-head routers: Q + KV ──
        self.q_routers = nn.ModuleList([
            nn.Linear(d_input, n_experts_per_head, bias=False)
            for _ in range(n_heads)])
        self.kv_routers = nn.ModuleList([
            nn.Linear(d_input, n_experts_per_head, bias=False)
            for _ in range(n_heads)])

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

        # ── Output path: preGate(KV) → W_o → postGate(Q) ──
        concat_dim = d_attn                                # H * dv
        self.W_pre_w = nn.Parameter(torch.randn(HE, d_input, concat_dim) * 0.01)
        self.W_o   = nn.Linear(concat_dim, d_input, bias=False)
        d_pg = max(int(concat_dim * 0.618), 1)
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

    @torch.compiler.disable
    def forward(self, stream):
        """
        stream: (B, n_mhc, T, d)
        Returns: stream_update (B, n_mhc, T, d)
        """
        B, n, T, d = stream.shape
        H, E = self.H, self.E
        HE = H * E
        positions = torch.arange(T, device=stream.device)
        route_input = stream.mean(dim=1)                   # (B, T, d)

        # ════════════════ 1. Batched routing ════════════════
        q_logits = torch.stack([self.q_routers[h](route_input) for h in range(H)])
        kv_logits = torch.stack([self.kv_routers[h](route_input) for h in range(H)])
        q_logits = q_logits.clamp(-10, 10)                 # (H, B, T, E)
        kv_logits = kv_logits.clamp(-10, 10)

        q_probs = F.softmax(q_logits / ROUTE_TEMP, dim=-1)
        kv_probs = F.softmax(kv_logits / ROUTE_TEMP, dim=-1)

        if self.training:
            q_sel = torch.multinomial(q_probs.reshape(-1, E), 1).reshape(H, B, T)
            kv_sel = torch.multinomial(kv_probs.reshape(-1, E), 1).reshape(H, B, T)
        else:
            q_sel = q_logits.argmax(dim=-1)                 # (H, B, T)
            kv_sel = kv_logits.argmax(dim=-1)

        q_zero = (q_sel == 0)                               # (H, B, T)
        kv_zero = (kv_sel == 0)
        zero_mask = q_zero | kv_zero                        # 任一选零专家 → 输出零

        # Routing log_probs for GRPO
        q_route_lp = q_probs.log().gather(-1, q_sel.unsqueeze(-1)).squeeze(-1)   # (H, B, T)
        kv_route_lp = kv_probs.log().gather(-1, kv_sel.unsqueeze(-1)).squeeze(-1)
        route_lp = q_route_lp.sum(0) + kv_route_lp.sum(0)                        # (B, T)

        # ════════════════ 2. Identify active (h,e) pairs ════════════════
        e_range = torch.arange(E, device=stream.device)
        q_active = (q_sel.unsqueeze(-1) == e_range).any(dim=(1, 2))   # (H, E)
        kv_active = (kv_sel.unsqueeze(-1) == e_range).any(dim=(1, 2))
        active_flat = (q_active | kv_active).reshape(-1)               # (H*E,)
        active_idx = active_flat.nonzero(as_tuple=True)[0]
        K = active_idx.shape[0]
        active_h = active_idx // E                                     # (K,)
        active_e = active_idx % E

        # ════════════════ 3. Batched MHC + norm ════════════════
        # Accumulators
        q_h_buf = stream.new_zeros(H, B, T, self.dk_pope)
        k_h_buf = stream.new_zeros(H, B, T, self.dk_pope)
        v_h_buf = stream.new_zeros(H, B, T, self.dv)
        alpha_buf = stream.new_zeros(H, B, T, self.dk_pope)
        beta_buf  = stream.new_zeros(H, B, T, 1)
        acc_pre_gate  = stream.new_zeros(B, T, H * self.dv)
        acc_post_gate = stream.new_zeros(B, T, d)
        acc_H_res  = stream.new_zeros(B, T, n, n)
        acc_H_post = stream.new_zeros(B, T, n)

        if K > 0:
            H_res_k, H_pre_k, H_post_k = self.batched_mhc(stream, active_idx)
            stream_exp = stream.unsqueeze(0).expand(K, -1, -1, -1, -1)
            h_e = torch.einsum('kbtn,kbntd->kbtd', H_pre_k, stream_exp)

            # Batched RMSNorm
            nw = self.batched_norm_w[active_idx]
            rms = h_e.float().pow(2).mean(-1, keepdim=True).add(1e-8).rsqrt().to(h_e.dtype)
            h_e = h_e * rms * nw.unsqueeze(1).unsqueeze(1)

            # ════════════════ 4. Batched Q path (e > 0) ════════════════
            q_path = (active_e > 0) & q_active[active_h, active_e]
            if q_path.any():
                qi = q_path.nonzero(as_tuple=True)[0]
                q_hi, q_ei, q_fi = active_h[qi], active_e[qi], active_idx[qi]
                h_q = h_e[qi]

                # LoRA Q + W_q projection
                lA = self.lora_A_q[q_fi]; lB = self.lora_B_q[q_fi]
                delta_q = torch.einsum(
                    'kbtr,krd->kbtd',
                    F.silu(torch.einsum('kbtd,kdr->kbtr', h_q, lA)), lB)
                Wq = self.W_q_w[q_hi]
                q_proj = torch.einsum('kbtd,kod->kbto', h_q, Wq)
                q_out = self._apply_pope(
                    F.normalize(q_proj + delta_q, dim=-1), positions, True)

                # STE mask
                q_mask_k = (q_sel[q_hi] == q_ei.unsqueeze(1).unsqueeze(2))
                q_prob_k = q_probs[q_hi, :, :, q_ei]
                eff_q = (q_prob_k + (q_mask_k.float() - q_prob_k).detach()
                         ).unsqueeze(-1)                    # (K_q, B, T, 1)

                # Scatter-accumulate q_h + postGate
                eff_q_out = eff_q * q_out
                pg_mid = torch.einsum('kbtd,kde->kbte', h_q, self.W_pg1_w[q_fi])
                pg_out = torch.sigmoid(torch.einsum(
                    'kbte,ked->kbtd', F.silu(pg_mid), self.W_pg2_w[q_fi]))
                eff_q_pg = eff_q * pg_out
                for h in range(H):
                    hm = (q_hi == h)
                    if hm.any():
                        q_h_buf[h] += eff_q_out[hm].sum(0)
                        acc_post_gate += eff_q_pg[hm].sum(0) / H

            # ════════════════ 5. Batched KV path ════════════════
            kv_path = kv_active[active_h, active_e]
            if kv_path.any():
                ki = kv_path.nonzero(as_tuple=True)[0]
                kv_hi, kv_ei, kv_fi = active_h[ki], active_e[ki], active_idx[ki]
                h_kv = h_e[ki]

                # STE mask
                kv_mask_k = (kv_sel[kv_hi] == kv_ei.unsqueeze(1).unsqueeze(2))
                kv_prob_k = kv_probs[kv_hi, :, :, kv_ei]
                eff_kv = (kv_prob_k + (kv_mask_k.float() - kv_prob_k).detach()
                          ).unsqueeze(-1)                   # (K_kv, B, T, 1)

                # LoRA K + W_k
                lAk = self.lora_A_k[kv_fi]; lBk = self.lora_B_k[kv_fi]
                delta_k = torch.einsum(
                    'kbtr,krd->kbtd',
                    F.silu(torch.einsum('kbtd,kdr->kbtr', h_kv, lAk)), lBk)
                Wk = self.W_k_w[kv_hi]
                k_proj = torch.einsum('kbtd,kod->kbto', h_kv, Wk)
                k_out = self._apply_pope(
                    F.normalize(k_proj + delta_k, dim=-1), positions, False)

                # LoRA V + W_v
                lAv = self.lora_A_v[kv_fi]; lBv = self.lora_B_v[kv_fi]
                delta_v = torch.einsum(
                    'kbtr,krd->kbtd',
                    F.silu(torch.einsum('kbtd,kdr->kbtr', h_kv, lAv)), lBv)
                Wv = self.W_v_w[kv_hi]
                v_out = F.silu(
                    torch.einsum('kbtd,kod->kbto', h_kv, Wv) + delta_v)

                # Alpha/beta MLPs
                a_mid = torch.einsum('kbtd,kde->kbte', h_kv, self.alpha_up_w[kv_fi])
                alpha_out = torch.sigmoid(torch.einsum(
                    'kbte,ked->kbtd', F.silu(a_mid), self.alpha_down_w[kv_fi]))
                b_mid = torch.einsum('kbtd,kde->kbte', h_kv, self.beta_up_w[kv_fi])
                beta_out = torch.sigmoid(torch.einsum(
                    'kbte,ked->kbtd', F.silu(b_mid), self.beta_down_w[kv_fi]))

                # PreGate
                pre_gate_out = F.silu(torch.einsum(
                    'kbtd,kde->kbte', h_kv, self.W_pre_w[kv_fi]))

                # Scatter-accumulate
                eff_kv_k = eff_kv * k_out
                eff_kv_v = eff_kv * v_out
                eff_kv_a = eff_kv * alpha_out
                eff_kv_b = eff_kv * beta_out
                eff_kv_pg = eff_kv * pre_gate_out
                for h in range(H):
                    hm = (kv_hi == h)
                    if hm.any():
                        k_h_buf[h] += eff_kv_k[hm].sum(0)
                        v_h_buf[h] += eff_kv_v[hm].sum(0)
                        alpha_buf[h] += eff_kv_a[hm].sum(0)
                        beta_buf[h] += eff_kv_b[hm].sum(0)
                        acc_pre_gate += eff_kv_pg[hm].sum(0) / H
                        # mHC outputs
                        acc_H_res += (H_res_k[ki[hm]] * eff_kv[hm].unsqueeze(-1)).sum(0)
                        acc_H_post += (H_post_k[ki[hm]] * eff_kv[hm]).sum(0)

        # ════════════════ 6. Batched KDA (all heads at once) ════════════════
        HB = H * B
        out_flat = self._kda_recursion(
            q_h_buf.reshape(HB, T, self.dk_pope),
            k_h_buf.reshape(HB, T, self.dk_pope),
            v_h_buf.reshape(HB, T, self.dv),
            alpha_buf.reshape(HB, T, self.dk_pope),
            beta_buf.reshape(HB, T, 1), T)
        out_he = out_flat.reshape(H, B, T, self.dv)

        # Zero out where expert 0 selected
        for h in range(H):
            out_he[h][zero_mask[h]] = 0.0

        all_outs = [out_he[h] for h in range(H)]

        # ════════════════ 7. Output assembly ════════════════
        concat = torch.cat(all_outs, dim=-1) / math.sqrt(self.H * self.dv)

        gated = concat * acc_pre_gate
        proj = self.W_o(gated)                              # (B, T, d)
        result = proj * acc_post_gate                       # (B, T, d)

        H_res_avg  = acc_H_res / H
        H_post_avg = acc_H_post / H
        res = torch.einsum('btij,bjtd->bitd', H_res_avg, stream)
        post = torch.einsum('btn,btd->bntd', H_post_avg, result)
        stream_update = res + post

        return stream_update, route_lp


# ────────────────── SwiGLU (building block) ──────────────────

class SwiGLU(nn.Module):
    """SwiGLU with bottleneck gate and RMSNorm."""
    def __init__(self, d_input):
        super().__init__()
        d_ffn = int(d_input * 1.618)
        self.norm = nn.RMSNorm(d_input)
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
    MoE-SwiGLU with batched expert computation.
    Only active experts (gate_moe > 0) participate in computation.
    Returns stream update and gate_moe for external AttnRes query combination.
    """
    def __init__(self, d_hidden, n_experts=16, n_mhc=4,
                 top_prob=0.8, max_k=4):
        super().__init__()
        self.n_experts = n_experts
        self.top_prob = top_prob
        self.max_k = max_k
        self.batched_mhc = BatchedMHC(n_experts, d_hidden, n_mhc)

        d_ffn = int(d_hidden * 1.618)
        self.swiglu_norm_w = nn.Parameter(torch.ones(n_experts, d_hidden))
        self.swiglu_wd_w   = nn.Parameter(torch.randn(n_experts, d_hidden, d_hidden) * 0.01)
        self.swiglu_wu_w   = nn.Parameter(torch.randn(n_experts, d_hidden, d_hidden) * 0.01)
        self.swiglu_gate_w = nn.Parameter(torch.randn(n_experts, d_ffn, d_hidden) * 0.01)
        self.swiglu_up_w   = nn.Parameter(torch.randn(n_experts, d_ffn, d_hidden) * 0.01)
        self.swiglu_down_w = nn.Parameter(torch.randn(n_experts, d_hidden, d_ffn) * 0.01)

        self.expert_router = nn.Linear(d_hidden, n_experts, bias=False)
        self.w_experts = nn.Parameter(torch.randn(n_experts, d_hidden) * 0.01)

    @torch.compiler.disable
    def forward(self, stream):
        """
        stream: (B, n_mhc, T, d)
        Returns: stream_update (B, n_mhc, T, d), gate_moe (B, T, E), route_lp (B, T)
        """
        B, n, T, d = stream.shape
        E = self.n_experts

        # ── Routing ──
        route_input = stream.mean(dim=1)                    # (B, T, d)
        router_logits = self.expert_router(route_input)
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-8) + 1e-8)
            router_logits = (router_logits + gumbel * FFN_GUMBEL_TAU).clamp(-10, 10)
        gate_moe = top_prob_max_k(router_logits,
                                   self.top_prob, self.max_k)  # (B, T, E)

        # Routing log_probs for GRPO
        router_probs = F.softmax(router_logits, dim=-1)     # (B, T, E)
        moe_mask = (gate_moe > 0)
        route_lp = (router_probs.log().clamp(min=-10) * moe_mask.float()).sum(-1)  # (B, T)

        # ── Identify active experts (skip zero expert) ──
        active_idx = (gate_moe > 0).any(dim=(0, 1)).nonzero(as_tuple=True)[0]
        active_idx = active_idx[active_idx > 0]                     # 零专家不参与计算
        K = active_idx.shape[0]
        if K == 0:
            return stream.new_zeros(B, n, T, d), gate_moe, route_lp

        # ── Batched MHC for active experts ──
        stream_exp = stream.unsqueeze(0).expand(K, -1, -1, -1, -1)  # (K, B, n, T, d)
        H_res, H_pre, H_post = self.batched_mhc(stream, active_idx)
        # H_res: (K, B, T, n, n), H_pre: (K, B, T, n), H_post: (K, B, T, n)

        # h_e = H_pre weighted mix of streams
        h_e = torch.einsum('kbtn,kbntd->kbtd', H_pre, stream_exp)   # (K, B, T, d)

        # ── Batched SwiGLU for active experts ──
        # Gather params
        nw   = self.swiglu_norm_w[active_idx]   # (K, d)
        wd_w = self.swiglu_wd_w[active_idx]     # (K, d, d)
        wu_w = self.swiglu_wu_w[active_idx]     # (K, d, d)
        gw   = self.swiglu_gate_w[active_idx]   # (K, d_ffn, d)
        uw   = self.swiglu_up_w[active_idx]     # (K, d_ffn, d)
        dw   = self.swiglu_down_w[active_idx]   # (K, d, d_ffn)

        # Batched RMSNorm
        rms = h_e.float().pow(2).mean(-1, keepdim=True).add(1e-8).rsqrt().to(h_e.dtype)
        h = h_e * rms * nw.unsqueeze(1).unsqueeze(1)                   # (K, B, T, d)

        # Bottleneck gate: silu(x @ wd) @ wu → sigmoid
        wd_out = torch.einsum('kbtd,kod->kbto', h, wd_w)               # (K, B, T, d)
        g = torch.sigmoid(torch.einsum('kbtd,kod->kbto', F.silu(wd_out), wu_w))

        # SwiGLU core: silu(x @ gate) * (x @ up) → down
        gate_out = torch.einsum('kbtd,kod->kbto', h, gw)               # (K, B, T, d_ffn)
        up_out   = torch.einsum('kbtd,kod->kbto', h, uw)               # (K, B, T, d_ffn)
        out_e = g * torch.einsum('kbtf,kof->kbto', F.silu(gate_out) * up_out, dw)

        # ── Gate and accumulate ──
        gate_e = gate_moe[:, :, active_idx]                             # (B, T, K)
        gate_4d = gate_e.permute(2, 0, 1).unsqueeze(2).unsqueeze(-1)   # (K, B, 1, T, 1)

        res  = torch.einsum('kbtij,kbjtd->kbitd', H_res, stream_exp)   # (K, B, n, T, d)
        post = torch.einsum('kbtn,kbtd->kbntd', H_post, out_e)         # (K, B, n, T, d)

        stream_update = (gate_4d * (res + post)).sum(dim=0)             # (B, n, T, d)

        return stream_update, gate_moe, route_lp


# ────────────────── Halting Router ──────────────────

class HaltingRouter(nn.Module):
    """Per-layer halting router: mHC H_pre aggregation → SwiGLU → Linear → continue/exit."""
    def __init__(self, d_hidden, n_mhc):
        super().__init__()
        self.n = n_mhc
        nd = n_mhc * d_hidden
        self.norm = nn.RMSNorm(nd)
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
    def __init__(self, d_input=14, d_hidden=D_HIDDEN, d_key=D_KEY, d_attn=D_ATTN,
                 n_actions=11,
                 max_iterations=MAX_ITERATIONS, inner_steps=INNER_STEPS,
                 n_mhc=4, min_iterations=MIN_ITERATIONS,
                 n_heads=N_ATTN_HEADS,
                 n_experts_per_head=N_EXPERTS_PER_HEAD,
                 n_ffn_experts=N_FFN_EXPERTS,
                 ffn_top_prob=FFN_TOP_PROB, ffn_max_k=FFN_MAX_K):
        super().__init__()
        self.max_iterations = max_iterations
        self.inner_steps = inner_steps
        self.min_iterations = min_iterations
        self.n_mhc = n_mhc
        self.n_heads = n_heads
        self.d = d_hidden

        self.inp_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.RMSNorm(d_hidden))

        # MoA-KDA + MoE-SwiGLU
        self.moa_kda = MoAKDALayer(d_hidden, d_key=d_key, d_attn=d_attn,
                                    n_heads=n_heads,
                                    n_experts_per_head=n_experts_per_head,
                                    n_mhc=n_mhc)
        self.moe_swiglu = MoESwiGLU(d_hidden, n_experts=n_ffn_experts, n_mhc=n_mhc,
                                     top_prob=ffn_top_prob, max_k=ffn_max_k)

        # Halting router
        self.router = HaltingRouter(d_hidden, n_mhc)

        # AttnRes: shared preH + final w + expert-averaged RMSNorm
        self.attn_res_expert_scales = nn.Parameter(torch.ones(n_ffn_experts, d_hidden))
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
        nn.init.normal_(self.moa_kda.W_o.weight,
                        std=scale * (2.0 / self.moa_kda.W_o.in_features) ** 0.5)
        nn.init.normal_(self.moe_swiglu.swiglu_down_w,
                        std=scale * (2.0 / self.moe_swiglu.swiglu_down_w.shape[-1]) ** 0.5)
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
            logits = torch.einsum('d,lbtd->lbt', w_l, K)
        else:                                             # (B, T, d) per-token
            logits = torch.einsum('btd,lbtd->lbt', w_l, K)
        alpha = logits.softmax(0)
        return torch.einsum('lbt,lbtd->btd', alpha, V)

    @torch.compiler.disable
    def _loop(self, stream, acc):
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
            soft = F.softmax(logits / HALT_TEMP, dim=-1)          # (B, 2) [continue, exit]

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
            gate_moe_sum = stream.new_zeros(B, T_total, self.moe_swiglu.n_experts)
            for j in range(self.inner_steps):
                attn_update, moa_lp = self.moa_kda(stream)
                stream = attn_update
                route_lp_total = route_lp_total + moa_lp.sum(-1)
                route_lp_count += 1

                ffn_update, gate_moe, moe_lp = self.moe_swiglu(stream)
                stream = ffn_update
                route_lp_total = route_lp_total + moe_lp.sum(-1)
                route_lp_count += 1

                mean_query = mean_query + torch.einsum(
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

    def forward(self, x):
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
        stream, acc, exit_iter, route_lp, expected_depth = self._loop(stream, acc)

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