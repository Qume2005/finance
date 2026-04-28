"""Mini-KDA Policy Network with PoPE, mHC, AttnRes, MoA, MoE, dynamic loop."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (N_ATTN_HEADS, N_Q_EXPERTS_PER_HEAD, N_KV_EXPERTS_PER_HEAD,
                    N_FFN_EXPERTS, FFN_TOP_PROB, FFN_MAX_K,
                    MAX_ITERATIONS, MIN_ITERATIONS)


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
        self.alpha_pre  = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

    def forward(self, stream):
        """stream: (B, n, T, d) → H_res(B,T,n,n), H_pre(B,T,n), H_post(B,T,n)"""
        B, n, T, d = stream.shape
        x = self.norm(stream.transpose(1, 2).reshape(B, T, n * d))

        H_pre = torch.sigmoid(self.alpha_pre * self.phi_pre(x) + self.b_pre)
        H_post = 2 * torch.sigmoid(self.alpha_post * self.phi_post(x) + self.b_post)
        H_res_raw = self.alpha_res * self.phi_res(x).reshape(B, T, self.n, self.n) + self.b_res
        H_res = sinkhorn_knopp(H_res_raw)

        return H_res, H_pre, H_post


# ────────────────── S-MHC (state mixing) ──────────────────

class S_MHC(nn.Module):
    """mHC for KDA state matrix S — operates on S magnitude (phase-free)."""
    def __init__(self, d_key, d_value, n=4):
        super().__init__()
        self.n = n
        self.dk = d_key
        nd = n * d_value
        self.norm = nn.RMSNorm(nd)
        self.phi_pre  = nn.Linear(nd, n, bias=False)
        self.phi_post = nn.Linear(nd, n, bias=False)
        self.phi_res  = nn.Linear(nd, n * n, bias=False)
        self.b_pre  = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res  = nn.Parameter(torch.zeros(n, n))
        self.alpha_pre  = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res  = nn.Parameter(torch.tensor(0.01))

    def forward(self, S_stack):
        """
        S_stack: (B, n, dk_pope, dv) — full state with PoPE phase
        Returns: H_res(B,n,n), H_pre(B,n), H_post(B,n)
        """
        B, n, dk2, dv = S_stack.shape
        dk = dk2 // 2
        # 幅度提取（消除相位）
        mag = (S_stack[:, :, :dk, :]**2 + S_stack[:, :, dk:, :]**2).mean(dim=2)  # (B, n, dv)
        x = self.norm(mag.reshape(B, n * dv))              # (B, n*dv)

        H_pre  = torch.sigmoid(self.alpha_pre * self.phi_pre(x) + self.b_pre)
        H_post = 2 * torch.sigmoid(self.alpha_post * self.phi_post(x) + self.b_post)
        H_res_raw = self.alpha_res * self.phi_res(x).reshape(B, self.n, self.n) + self.b_res
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
    Multi-Head MoA-KDA: H heads, each with separate Q expert pool and KV expert pool.
    Q expert 0 = zero/no-op; all KV experts are active. Per-head top-1 routing.
    Per-head KDA, concat heads → preGate(KV) → W_o → postGate(Q).
    """
    def __init__(self, d_input, d_key=16, d_value=16,
                 n_heads=8, n_q_experts_per_head=32, n_kv_experts_per_head=24,
                 n_mhc=4):
        super().__init__()
        self.dk, self.dv = d_key, d_value
        self.dk_pope = d_key * 2
        self.H = n_heads
        self.Eq = n_q_experts_per_head
        self.Ek = n_kv_experts_per_head
        self.n_mhc = n_mhc
        r = max(d_key // 4, 1)                          # LoRA rank

        # ── Shared base projections ──
        self.register_buffer(
            'freqs', 10000.0 ** (torch.arange(d_key).float() / d_key))
        self.W_q = nn.Linear(d_input, d_key, bias=False)
        self.W_k = nn.Linear(d_input, d_key, bias=False)
        self.W_v = nn.Linear(d_input, d_value, bias=False)
        self.pope_delta_raw = nn.Parameter(torch.zeros(d_key))

        # ── Per-head routers: Q routers + KV routers ──
        self.q_routers = nn.ModuleList([
            nn.Linear(d_input, n_q_experts_per_head, bias=False)
            for _ in range(n_heads)])
        self.kv_routers = nn.ModuleList([
            nn.Linear(d_input, n_kv_experts_per_head, bias=False)
            for _ in range(n_heads)])

        # ── Q expert params (flat: idx = h*Eq + e) — expert 0 = zero ──
        HQ = n_heads * n_q_experts_per_head
        self.lora_A_q = nn.Parameter(torch.randn(HQ, d_input, r) * 0.01)
        self.lora_B_q = nn.Parameter(torch.zeros(HQ, r, d_key))
        self.mhc_q  = nn.ModuleList([MHC(d_input, n_mhc) for _ in range(HQ)])
        self.norm_q = nn.ModuleList([nn.RMSNorm(d_input) for _ in range(HQ)])

        # ── KV expert params (flat: idx = h*Ek + e) — all active ──
        HK = n_heads * n_kv_experts_per_head
        self.lora_A_k = nn.Parameter(torch.randn(HK, d_input, r) * 0.01)
        self.lora_B_k = nn.Parameter(torch.zeros(HK, r, d_key))
        self.lora_A_v = nn.Parameter(torch.randn(HK, d_input, r) * 0.01)
        self.lora_B_v = nn.Parameter(torch.zeros(HK, r, d_value))

        d_alpha = int(d_key * 1.618)
        self.alpha_up_w   = nn.Parameter(torch.randn(HK, d_input, d_alpha) * 0.01)
        self.alpha_down_w = nn.Parameter(torch.randn(HK, d_alpha, self.dk_pope) * 0.01)
        self.beta_up_w    = nn.Parameter(torch.randn(HK, d_input, d_alpha) * 0.01)
        self.beta_down_w  = nn.Parameter(torch.randn(HK, d_alpha, 1) * 0.01)

        self.mhc_kv  = nn.ModuleList([MHC(d_input, n_mhc) for _ in range(HK)])
        self.norm_kv = nn.ModuleList([nn.RMSNorm(d_input) for _ in range(HK)])

        # ── S-MHC: per-head state stream mixing ──
        self.s_mhc = nn.ModuleList([S_MHC(d_key, d_value, n_mhc) for _ in range(n_heads)])

        # ── Output path: preGate(KV) → W_o → postGate(Q) ──
        concat_dim = n_heads * d_value                    # H * dv
        self.W_pre = nn.Linear(d_input, concat_dim, bias=False)
        self.W_o   = nn.Linear(concat_dim, d_input, bias=False)
        d_pg = max(int(concat_dim * 0.618), 1)
        self.W_pg1 = nn.Linear(d_input, d_pg, bias=False)
        self.W_pg2 = nn.Linear(d_pg, d_input, bias=False)

    def _apply_pope(self, x, positions, is_query):
        mu = F.softplus(x)
        phi = positions.unsqueeze(1) * self.freqs.to(x.device).unsqueeze(0)
        if not is_query:
            phi = phi + (-2 * math.pi * torch.sigmoid(self.pope_delta_raw))
        real = mu * torch.cos(phi)
        imag = mu * torch.sin(phi)
        return torch.cat([real, imag], dim=-1)

    @torch.compiler.disable
    def _kda_recursion(self, q, k, v, alpha, beta, T, S_init=None, s_mhc=None):
        B = q.shape[0]
        n = self.n_mhc

        # S_init: (B, n, dk_pope, dv) or None
        if S_init is not None:
            S_stack = S_init                                  # (B, n, dk_pope, dv)
        else:
            S_stack = q.new_zeros(B, n, self.dk_pope, self.dv)

        # mHC 混合矩阵（每窗口算一次）
        H_res, H_pre, H_post = s_mhc(S_stack)                # (B,n,n), (B,n), (B,n)

        out = q.new_empty(B, T, self.dv)
        for t in range(T):
            # 聚合 n 流 → 单个 s
            s = torch.einsum('bi,bide->bde', H_pre, S_stack)  # (B, dk_pope, dv)

            # Delta Rule（_kda_step 不变）
            attn_out, s_new = _kda_step(q[:, t], k[:, t], v[:, t],
                                        alpha[:, t], beta[:, t], s)

            # 残差 + 分发
            delta = s_new - s                                  # (B, dk_pope, dv)
            S_stack = (torch.einsum('bij,bjde->bide', H_res, S_stack)
                       + torch.einsum('bi,bde->bide', H_post, delta))

            out[:, t] = attn_out

        return out, S_stack

    @torch.compiler.disable
    def forward(self, stream, S_init=None):
        """
        stream: (B, n_mhc, T, d)
        S_init: list[H] of (B, n, dk_pope, dv) or None
        Returns: stream_update (B, n_mhc, T, d), S_new list[H] of (B, n, dk_pope, dv)
        """
        B, n, T, d = stream.shape
        H, Eq, Ek = self.H, self.Eq, self.Ek
        positions = torch.arange(T, device=stream.device)
        route_input = stream.mean(dim=1)                   # (B, T, d)

        all_outs = []
        S_new_list = []
        h_q_routed  = stream.new_zeros(B, T, d)           # for postGate (Q-bound)
        h_kv_routed = stream.new_zeros(B, T, d)           # for preGate (KV-bound)

        # Stream mixing accumulators
        q_stream_mix = []                                  # per head: list of (H_res, H_post, mask)
        kv_stream_mix = []

        for h in range(H):
            # ── Q routing (top-1) ──
            q_logits = self.q_routers[h](route_input)      # (B, T, Eq)
            q_probs = F.softmax(q_logits, dim=-1)
            q_sel = q_logits.argmax(dim=-1)                 # (B, T)
            q_coeff = q_probs.gather(-1, q_sel.unsqueeze(-1)).squeeze(-1)

            # ── KV routing (top-1) ──
            kv_logits = self.kv_routers[h](route_input)    # (B, T, Ek)
            kv_probs = F.softmax(kv_logits, dim=-1)
            kv_sel = kv_logits.argmax(dim=-1)               # (B, T)
            kv_coeff = kv_probs.gather(-1, kv_sel.unsqueeze(-1)).squeeze(-1)

            q_zero = (q_sel == 0)

            # ── Phase 1a: Q experts → accumulate q ──
            q_h = stream.new_zeros(B, T, self.dk_pope)
            head_q_Hres = []
            head_q_Hpost = []
            head_q_mask = []

            for e in range(1, Eq):
                mask_e = (q_sel == e)
                if not mask_e.any():
                    head_q_Hres.append(None)
                    head_q_Hpost.append(None)
                    head_q_mask.append(None)
                    continue

                idx = h * Eq + e
                H_res, H_pre, H_post = self.mhc_q[idx](stream)
                h_e = torch.einsum('btn,bntd->btd', H_pre, stream)
                h_e = self.norm_q[idx](h_e)

                mask_f = mask_e.unsqueeze(-1).float()
                delta_q = F.silu(h_e @ self.lora_A_q[idx]) @ self.lora_B_q[idx]
                q_e = self._apply_pope(
                    F.normalize(self.W_q(h_e) + delta_q, dim=-1), positions, True)
                q_h += mask_f * q_e

                h_q_routed += (mask_f * h_e) / H

                head_q_Hres.append(H_res)
                head_q_Hpost.append(H_post)
                head_q_mask.append(mask_e)

            # ── Phase 1b: KV experts → accumulate k/v/α/β ──
            k_h = stream.new_zeros(B, T, self.dk_pope)
            v_h = stream.new_zeros(B, T, self.dv)
            alpha_h = stream.new_zeros(B, T, self.dk_pope)
            beta_h  = stream.new_zeros(B, T, 1)
            head_kv_Hres = []
            head_kv_Hpost = []
            head_kv_mask = []

            for e in range(Ek):
                mask_e = (kv_sel == e)
                if not mask_e.any():
                    head_kv_Hres.append(None)
                    head_kv_Hpost.append(None)
                    head_kv_mask.append(None)
                    continue

                idx = h * Ek + e
                H_res, H_pre, H_post = self.mhc_kv[idx](stream)
                h_e = torch.einsum('btn,bntd->btd', H_pre, stream)
                h_e = self.norm_kv[idx](h_e)

                mask_f = mask_e.unsqueeze(-1).float()

                delta_k = F.silu(h_e @ self.lora_A_k[idx]) @ self.lora_B_k[idx]
                k_e = self._apply_pope(
                    F.normalize(self.W_k(h_e) + delta_k, dim=-1), positions, False)
                k_h += mask_f * k_e

                delta_v = F.silu(h_e @ self.lora_A_v[idx]) @ self.lora_B_v[idx]
                v_e = F.silu(self.W_v(h_e) + delta_v)
                v_h += mask_f * v_e

                alpha_e = torch.sigmoid(
                    F.silu(h_e @ self.alpha_up_w[idx]) @ self.alpha_down_w[idx])
                beta_e = torch.sigmoid(
                    F.silu(h_e @ self.beta_up_w[idx]) @ self.beta_down_w[idx])
                alpha_h += mask_f * alpha_e
                beta_h  += mask_f * beta_e

                h_kv_routed += (mask_f * h_e) / H

                head_kv_Hres.append(H_res)
                head_kv_Hpost.append(H_post)
                head_kv_mask.append(mask_e)

            # ── Q zero expert → output 0 ──

            # ── Per-head KDA ──
            S_h_init = S_init[h] if S_init is not None else None
            out_h, S_h_new = self._kda_recursion(
                q_h, k_h, v_h, alpha_h, beta_h, T, S_h_init, s_mhc=self.s_mhc[h])

            # Scale by Q coefficient; zero out Q-zero tokens
            out_h = out_h * q_coeff.unsqueeze(-1)
            out_h[q_zero] = 0.0

            all_outs.append(out_h)
            S_new_list.append(S_h_new)
            q_stream_mix.append((head_q_Hres, head_q_Hpost, head_q_mask))
            kv_stream_mix.append((head_kv_Hres, head_kv_Hpost, head_kv_mask))

        # ── Concatenate all heads ──
        concat = torch.cat(all_outs, dim=-1)               # (B, T, H*dv)

        # ── preGate(KV) → W_o → postGate(Q) ──
        pre_gate = F.silu(self.W_pre(h_kv_routed))         # (B, T, H*dv)
        gated = concat * pre_gate
        proj = self.W_o(gated)                              # (B, T, d)
        post_gate = torch.sigmoid(
            self.W_pg2(F.silu(self.W_pg1(h_q_routed))))    # (B, T, d)
        result = proj * post_gate                           # (B, T, d)

        # ── mHC stream mixing ──
        stream_update = stream.new_zeros(B, n, T, d)

        for h in range(H):
            # Q experts
            q_Hres, q_Hpost, q_masks = q_stream_mix[h]
            for ei, e in enumerate(range(1, Eq)):
                if q_masks[ei] is None:
                    continue
                m4 = q_masks[ei].view(B, 1, T, 1).float()
                res = torch.einsum('btij,bjtd->bitd', q_Hres[ei], stream)
                post = torch.einsum('btn,btd->bntd', q_Hpost[ei], result)
                stream_update += m4 * (res + post)

            # KV experts
            kv_Hres, kv_Hpost, kv_masks = kv_stream_mix[h]
            for ei, e in enumerate(range(1, Ek)):
                if kv_masks[ei] is None:
                    continue
                m4 = kv_masks[ei].view(B, 1, T, 1).float()
                res = torch.einsum('btij,bjtd->bitd', kv_Hres[ei], stream)
                post = torch.einsum('btn,btd->bntd', kv_Hpost[ei], result)
                stream_update += m4 * (res + post)

        return stream_update, S_new_list


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
    MoE-SwiGLU: per-expert SwiGLU + per-expert mHC + per-expert AttnRes w.
    Returns stream update and gate_moe for external AttnRes query combination.
    """
    def __init__(self, d_hidden, n_experts=16, n_mhc=4,
                 top_prob=0.8, max_k=4):
        super().__init__()
        self.n_experts = n_experts
        self.top_prob = top_prob
        self.max_k = max_k
        self.experts = nn.ModuleList([SwiGLU(d_hidden) for _ in range(n_experts)])
        self.mhc_swiglu = nn.ModuleList([MHC(d_hidden, n_mhc) for _ in range(n_experts)])
        self.expert_router = nn.Linear(d_hidden, n_experts, bias=False)
        self.w_experts = nn.Parameter(torch.randn(n_experts, d_hidden) * 0.01)

    @torch.compiler.disable
    def forward(self, stream):
        """
        stream: (B, n_mhc, T, d)
        Returns: stream_update (B, n_mhc, T, d), gate_moe (B, T, E)
        """
        B, n, T, d = stream.shape
        E = self.n_experts

        route_input = stream.mean(dim=1)                    # (B, T, d)
        gate_moe = top_prob_max_k(self.expert_router(route_input),
                                   self.top_prob, self.max_k)  # (B, T, E)

        stream_update = stream.new_zeros(B, n, T, d)

        for e in range(E):
            H_res, H_pre, H_post = self.mhc_swiglu[e](stream)
            h_e = torch.einsum('btn,bntd->btd', H_pre, stream)
            out_e = self.experts[e](h_e)                  # SwiGLU has own preNorm

            g = gate_moe[:, :, e].view(B, 1, T, 1)
            res = torch.einsum('btij,bjtd->bitd', H_res, stream)
            post = torch.einsum('btn,btd->bntd', H_post, out_e)
            stream_update += g * (res + post)

        return stream_update, gate_moe


# ────────────────── Halting (dynamic loop) ──────────────────

class HaltingRouter(nn.Module):
    """Per-sample halting router with linear exit bias α·T."""
    def __init__(self, d_hidden, n_mhc, max_iterations=12):
        super().__init__()
        nd = n_mhc * d_hidden
        self.n_mhc = n_mhc
        self.d = d_hidden
        self.norm = nn.RMSNorm(nd)
        self.mix = nn.Linear(nd, nd, bias=False)
        self.proj = nn.Linear(d_hidden, 2, bias=False)

    def forward(self, stream):
        """
        stream: (B, n, T, d)
        """
        B, n, T, d = stream.shape
        last = stream[:, :, -1, :]
        x = self.norm(last.reshape(B, n * d))
        x = self.mix(x)
        x = x.reshape(B, n, d).sum(dim=1)               # (B, d)

        logits = self.proj(x)                             # (B, 2)
        return logits


def _ste_route(logits):
    """STE 2-way gate: forward=argmax, backward=straight-through softmax."""
    soft = F.softmax(logits, dim=-1)                      # (B, 2)
    hard_idx = logits.argmax(dim=-1)
    hard = (hard_idx == 0).float()                        # 1=continue, 0=exit
    ste = soft[:, 0] + (hard - soft[:, 0]).detach()
    return ste, hard


# ────────────────── Main Network ──────────────────

class KDAPolicyNetwork(nn.Module):
    """
    Dynamic-depth MoA-KDA + MoE-SwiGLU with mHC, AttnRes, STE halting router.

    Shared single MoA+MoE block looped dynamically.  STE router decides
    per-sample continue/exit.  Per-expert mHC, post_norm, gates, AttnRes w.
    """
    def __init__(self, d_input=14, d_hidden=48, n_actions=11,
                 max_iterations=MAX_ITERATIONS, n_mhc=4,
                 min_iterations=MIN_ITERATIONS,
                 n_heads=N_ATTN_HEADS,
                 n_q_experts_per_head=N_Q_EXPERTS_PER_HEAD,
                 n_kv_experts_per_head=N_KV_EXPERTS_PER_HEAD,
                 n_ffn_experts=N_FFN_EXPERTS,
                 ffn_top_prob=FFN_TOP_PROB, ffn_max_k=FFN_MAX_K):
        super().__init__()
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.n_mhc = n_mhc
        self.n_heads = n_heads
        self.d = d_hidden

        self.inp_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.RMSNorm(d_hidden))

        # Learned [EOS] embedding — appended after the sequence
        self.eos = nn.Parameter(torch.randn(d_hidden) * 0.02)

        # MoA-KDA + MoE-SwiGLU
        self.moa_kda = MoAKDALayer(d_hidden, n_heads=n_heads,
                                    n_q_experts_per_head=n_q_experts_per_head,
                                    n_kv_experts_per_head=n_kv_experts_per_head,
                                    n_mhc=n_mhc)
        self.moe_swiglu = MoESwiGLU(d_hidden, n_experts=n_ffn_experts, n_mhc=n_mhc,
                                     top_prob=ffn_top_prob, max_k=ffn_max_k)

        # STE halting
        self.router = HaltingRouter(d_hidden, n_mhc, max_iterations)

        # AttnRes: shared preH + final w
        self.attn_res_norm = nn.RMSNorm(d_hidden)
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
        depth = self.max_iterations * 2
        scale = depth ** -0.5
        nn.init.normal_(self.moa_kda.W_o.weight,
                        std=scale * (2.0 / self.moa_kda.W_o.in_features) ** 0.5)
        for expert in self.moe_swiglu.experts:
            nn.init.normal_(expert.down.weight,
                            std=scale * (2.0 / expert.down.in_features) ** 0.5)
        nn.init.normal_(self.head_down.weight,
                        std=scale * (2.0 / self.head_down.in_features) ** 0.5)
        nn.init.normal_(self.preH_iter.weight, std=scale)
        nn.init.normal_(self.preH_final.weight, std=scale)

    @torch.compiler.disable
    def _attn_res(self, acc, w_l, pre_h):
        """AttnRes: project streams → softmax weighted sum. w_l: (d,) or (B,T,d)."""
        projected = []
        for a in acc:
            B, n, T, d = a.shape
            projected.append(pre_h(a.permute(0, 2, 1, 3).reshape(B, T, n * d)))
        V = torch.stack(projected)                        # (L, B, T, d)
        K = self.attn_res_norm(V)
        if w_l.dim() == 1:
            logits = torch.einsum('d,lbtd->lbt', w_l, K)
        else:                                             # (B, T, d) per-token
            logits = torch.einsum('btd,lbtd->lbt', w_l, K)
        alpha = logits.softmax(0)
        return torch.einsum('lbt,lbtd->btd', alpha, V)

    @torch.compiler.disable
    def _loop(self, stream, kda_states, acc):
        """Dynamic loop body — not compiled (many experts, variable iterations)."""
        B, n, T_total, d = stream.shape
        H = self.n_heads
        active = torch.ones(B, dtype=torch.bool, device=stream.device)
        exit_iter = torch.full((B,), float(self.max_iterations), device=stream.device)

        for i in range(self.max_iterations):
            logits = self.router(stream)
            ste, hard = _ste_route(logits)

            if i < self.min_iterations:
                ste = active.float()
                hard = torch.ones_like(hard)

            gate = ste * active.float()
            gate_4d = gate.view(B, 1, 1, 1)
            gate_kda = gate.view(B, 1, 1, 1)

            # MoA Attention (returns list[H] of S matrices)
            attn_update, S_new_list = self.moa_kda(stream, S_init=kda_states)
            updated = attn_update

            # MoE SwiGLU
            ffn_update, gate_moe = self.moe_swiglu(updated)
            updated = ffn_update

            # Per-expert AttnRes
            w_combined = torch.einsum('bte,ed->btd',
                                      gate_moe, self.moe_swiglu.w_experts)
            updated = updated + self._attn_res(
                acc, w_combined, self.preH_iter).unsqueeze(1)

            # Gate: active update, exited spin
            stream = gate_4d * updated + (1 - gate_4d) * stream

            # Per-head KDA state gating
            if kda_states is not None:
                for h in range(H):
                    kda_states[h] = gate_kda * S_new_list[h] + (1 - gate_kda) * kda_states[h]
            else:
                kda_states = S_new_list

            acc.append(stream)

            hard_cont = (hard > 0.5)
            newly_exited = active & (~hard_cont)
            exit_iter = torch.where(newly_exited,
                                    torch.tensor(float(i + 1), device=stream.device),
                                    exit_iter)
            active = active & hard_cont
            if not active.any():
                break

        return stream, kda_states, acc, exit_iter

    def _head(self, h_out):
        """Output head — compiled separately for speed."""
        return self.head_down(F.silu(self.head_gate(h_out)) * self.head_up(h_out))

    def forward(self, x, kda_states=None):
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(0)
        B, T, _ = x.shape
        n = self.n_mhc
        d = self.d

        # kda_states: list[H] of (B, n, dk_pope, dv) or None
        # _loop expects same format

        # Append learned [EOS] at the end
        eos_emb = self.eos.view(1, 1, d).expand(B, 1, d)
        v0 = self.inp_proj(x)
        v0 = torch.cat([v0, eos_emb], dim=1)            # (B, T+1, d)

        stream = torch.zeros(B, n, T + 1, d, device=x.device)
        stream[:, 0] = v0

        acc = [stream]

        # Dynamic loop (not compiled)
        stream, kda_states, acc, exit_iter = self._loop(stream, kda_states, acc)

        h_out = self._attn_res(acc, self.w_final, self.preH_final)
        out = self._head(h_out)
        if squeeze:
            out = out.squeeze(0)
        return out, kda_states, exit_iter