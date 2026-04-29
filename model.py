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
    Per-head KDA, concat heads → preGate(KV) → W_o → postGate(Q).
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

        # ── Per-head base projections ──
        self.register_buffer(
            'freqs', 10000.0 ** (torch.arange(d_key).float() / d_key))
        self.W_q = nn.ModuleList([nn.Linear(d_input, d_key, bias=False) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_input, d_key, bias=False) for _ in range(n_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_input, self.dv, bias=False) for _ in range(n_heads)])
        self.pope_delta_raw = nn.Parameter(torch.zeros(d_key))

        # ── Per-head routers: Q + KV, both route into same expert pool ──
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

        self.mhc  = nn.ModuleList([MHC(d_input, n_mhc) for _ in range(HE)])
        self.norm = nn.ModuleList([nn.RMSNorm(d_input) for _ in range(HE)])

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

        Flow: per-head expert routing → per-head KDA → concat →
              preGate → W_o → postGate → output stream mixing
        """
        B, n, T, d = stream.shape
        H, E = self.H, self.E
        positions = torch.arange(T, device=stream.device)
        route_input = stream.mean(dim=1)                   # (B, T, d)

        all_outs = []

        # Per-expert gate 累积（激活专家输出取平均）
        acc_pre_gate  = stream.new_zeros(B, T, H * self.dv)
        acc_post_gate = stream.new_zeros(B, T, d)

        # KV expert mHC 累积（per token 位置，H 个 head 各选 1 个 KV expert）
        acc_H_res  = stream.new_zeros(B, T, n, n)
        acc_H_post = stream.new_zeros(B, T, n)

        for h in range(H):
            # ── Q routing (top-1) — expert 0 = zero ──
            q_logits = self.q_routers[h](route_input).clamp(-10, 10)
            q_probs = F.softmax(q_logits / ROUTE_TEMP, dim=-1)
            if self.training:
                q_sel = torch.multinomial(q_probs.reshape(-1, E), 1).reshape(B, T)
            else:
                q_sel = q_logits.argmax(dim=-1)
            # ── KV routing (top-1) — all active ──
            kv_logits = self.kv_routers[h](route_input).clamp(-10, 10)
            kv_probs = F.softmax(kv_logits / ROUTE_TEMP, dim=-1)
            if self.training:
                kv_sel = torch.multinomial(kv_probs.reshape(-1, E), 1).reshape(B, T)
            else:
                kv_sel = kv_logits.argmax(dim=-1)
            q_zero = (q_sel == 0)

            # ── 统一专家循环 ──
            q_h = stream.new_zeros(B, T, self.dk_pope)
            k_h = stream.new_zeros(B, T, self.dk_pope)
            v_h = stream.new_zeros(B, T, self.dv)
            alpha_h = stream.new_zeros(B, T, self.dk_pope)
            beta_h  = stream.new_zeros(B, T, 1)

            for e in range(E):
                q_mask = (q_sel == e)
                kv_mask = (kv_sel == e)
                if not (q_mask.any() or kv_mask.any()):
                    continue

                idx = h * E + e

                # mHC — 一次计算，Q/KV 共享 h_e
                need_kv = kv_mask.any()
                if need_kv:
                    H_res_e, H_pre, H_post_e = self.mhc[idx](stream)
                else:
                    _, H_pre, _ = self.mhc[idx](stream)

                h_e = torch.einsum('btn,bntd->btd', H_pre, stream)
                h_e = self.norm[idx](h_e)

                # Q path（expert 0 = zero，跳过）— STE 软近似反向
                if e > 0 and q_mask.any():
                    q_prob_e = q_probs[:, :, e:e+1]              # (B,T,1) 可微
                    mask_f = q_mask.unsqueeze(-1).float()
                    eff_q = q_prob_e + (mask_f - q_prob_e).detach()
                    delta_q = F.silu(h_e @ self.lora_A_q[idx]) @ self.lora_B_q[idx]
                    q_e = self._apply_pope(
                        F.normalize(self.W_q[h](h_e) + delta_q, dim=-1), positions, True)
                    q_h += eff_q * q_e

                    # Per-expert postGate
                    pg_e = torch.sigmoid(
                        F.silu(h_e @ self.W_pg1_w[idx]) @ self.W_pg2_w[idx])
                    acc_post_gate += eff_q * pg_e / H

                # KV path — STE 软近似反向
                if need_kv:
                    kv_prob_e = kv_probs[:, :, e:e+1]            # (B,T,1) 可微
                    mask_f = kv_mask.unsqueeze(-1).float()
                    eff_kv = kv_prob_e + (mask_f - kv_prob_e).detach()

                    delta_k = F.silu(h_e @ self.lora_A_k[idx]) @ self.lora_B_k[idx]
                    k_e = self._apply_pope(
                        F.normalize(self.W_k[h](h_e) + delta_k, dim=-1), positions, False)
                    k_h += eff_kv * k_e

                    delta_v = F.silu(h_e @ self.lora_A_v[idx]) @ self.lora_B_v[idx]
                    v_e = F.silu(self.W_v[h](h_e) + delta_v)
                    v_h += eff_kv * v_e

                    alpha_e = torch.sigmoid(
                        F.silu(h_e @ self.alpha_up_w[idx]) @ self.alpha_down_w[idx])
                    beta_e = torch.sigmoid(
                        F.silu(h_e @ self.beta_up_w[idx]) @ self.beta_down_w[idx])
                    alpha_h += eff_kv * alpha_e
                    beta_h  += eff_kv * beta_e

                    # Per-expert preGate
                    pre_gate_e = F.silu(h_e @ self.W_pre_w[idx])
                    acc_pre_gate += eff_kv * pre_gate_e / H

                    # 累积 KV expert 的 mHC output
                    acc_H_res  += H_res_e * eff_kv.unsqueeze(-1)
                    acc_H_post += H_post_e * eff_kv

            # ── Per-head KDA ──
            out_h = self._kda_recursion(q_h, k_h, v_h, alpha_h, beta_h, T)
            out_h[q_zero] = 0.0
            all_outs.append(out_h)

        # ── Concatenate all heads + scale ──
        concat = torch.cat(all_outs, dim=-1) / math.sqrt(self.H * self.dv)

        # ── preGate(激活专家平均) → W_o → postGate(激活专家平均) ──
        gated = concat * acc_pre_gate
        proj = self.W_o(gated)                              # (B, T, d)
        result = proj * acc_post_gate                       # (B, T, d)

        # ── Output stream mixing: 由 KV expert mHC 动态提供 ──
        H_res_avg  = acc_H_res / H
        H_post_avg = acc_H_post / H
        res = torch.einsum('btij,bjtd->bitd', H_res_avg, stream)
        post = torch.einsum('btn,btd->bntd', H_post_avg, result)
        stream_update = res + post

        return stream_update


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
        router_logits = self.expert_router(route_input)
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-8) + 1e-8)
            router_logits = (router_logits + gumbel * FFN_GUMBEL_TAU).clamp(-10, 10)
        gate_moe = top_prob_max_k(router_logits,
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

        # Learned [EOS] embedding — appended after the sequence
        self.eos = nn.Parameter(torch.randn(d_hidden) * 0.02)

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
        for expert in self.moe_swiglu.experts:
            nn.init.normal_(expert.down.weight,
                            std=scale * (2.0 / expert.down.in_features) ** 0.5)
        nn.init.normal_(self.head_down.weight,
                        std=scale * (2.0 / self.head_down.in_features) ** 0.5)
        nn.init.normal_(self.preH_iter.weight,
                        std=(2.0 / self.preH_iter.weight.shape[1]) ** 0.5)
        nn.init.normal_(self.preH_final.weight,
                        std=(2.0 / self.preH_final.weight.shape[1]) ** 0.5)

    @torch.compiler.disable
    def _attn_res(self, acc, w_l, pre_h, gate_moe=None):
        """AttnRes: project streams → expert-averaged RMSNorm → softmax weighted sum."""
        projected = []
        for a in acc:
            B, n, T, d = a.shape
            projected.append(pre_h(a.permute(0, 2, 1, 3).reshape(B, T, n * d)))
        V = torch.stack(projected)                        # (L, B, T, d)
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
        """
        B, n, T_total, d = stream.shape

        if self.training:
            target_depths = torch.randint(
                self.min_iterations, self.max_iterations + 1,
                (B,), device=stream.device)

        active = torch.ones(B, dtype=torch.bool, device=stream.device)
        exit_iter = torch.full((B,), float(self.max_iterations), device=stream.device)

        for i in range(self.max_iterations):
            # ── Halting decision ──
            logits = self.router(stream).clamp(-10, 10)       # (B, 2)
            soft = F.softmax(logits / HALT_TEMP, dim=-1)      # (B, 2)

            if self.training:
                should_continue = (i < target_depths).float()
                gate = soft[:, 0] + (should_continue - soft[:, 0]).detach()
            else:
                hard = (logits.argmax(dim=-1) == 0).float()
                gate = hard

            if i < self.min_iterations:
                gate = torch.ones(B, device=stream.device)

            gate_4d = gate.view(B, 1, 1, 1)

            stream_pre = stream

            # ── Inner loop: INNER_STEPS × (MoA + MoE) ──
            mean_query = stream.new_zeros(B, T_total, d)
            gate_moe_sum = stream.new_zeros(B, T_total, self.moe_swiglu.n_experts)
            for j in range(self.inner_steps):
                attn_update = self.moa_kda(stream)
                stream = attn_update

                ffn_update, gate_moe = self.moe_swiglu(stream)
                stream = ffn_update

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

        return stream, acc, exit_iter

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

        # Append learned [EOS] at the end
        eos_emb = self.eos.view(1, 1, d).expand(B, 1, d)
        v0 = self.inp_proj(x)
        v0 = torch.cat([v0, eos_emb], dim=1)            # (B, T+1, d)

        stream = torch.zeros(B, n, T + 1, d, device=x.device)
        stream[:, 0] = v0

        acc = [stream]

        # Dynamic loop (not compiled)
        stream, acc, exit_iter = self._loop(stream, acc)

        h_out = self._attn_res(acc, self.w_final, self.preH_final)
        out = self._head(h_out)
        if squeeze:
            out = out.squeeze(0)
        return out, exit_iter