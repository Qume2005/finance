"""Mini-KDA Policy Network with PoPE, mHC, AttnRes, MoA, MoE, dynamic loop."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = self.norm(stream.permute(0, 2, 1, 3).reshape(B, T, n * d))

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
    MoA-KDA: shared base W_q/W_k/W_v + per-expert LoRA + per-expert mHC.
    Per-KV-expert KDA → concat heads → preGate(KV) → W_o → postGate(Q).
    """
    def __init__(self, d_input, d_key=16, d_value=16,
                 n_q_experts=16, n_kv_experts=16, n_mhc=4,
                 q_top_prob=0.8, q_max_k=4, kv_top_prob=0.8, kv_max_k=4):
        super().__init__()
        self.dk, self.dv = d_key, d_value
        self.dk_pope = d_key * 2
        self.q_top_prob = q_top_prob
        self.q_max_k = q_max_k
        self.kv_top_prob = kv_top_prob
        self.kv_max_k = kv_max_k
        self.n_q_experts = n_q_experts
        self.n_kv_experts = n_kv_experts
        r = max(d_key // 4, 1)                          # LoRA rank = d_key / 4

        # ── Shared base projections ──
        self.register_buffer(
            'freqs', 10000.0 ** (torch.arange(d_key).float() / d_key))
        self.W_q = nn.Linear(d_input, d_key, bias=False)
        self.W_k = nn.Linear(d_input, d_key, bias=False)
        self.W_v = nn.Linear(d_input, d_value, bias=False)
        self.pope_delta_raw = nn.Parameter(torch.zeros(d_key))

        # ── Q-expert params (LoRA_q + mHC) ──
        self.lora_A_q = nn.Parameter(torch.randn(n_q_experts, d_input, r) * 0.01)
        self.lora_B_q = nn.Parameter(torch.zeros(n_q_experts, r, d_key))
        self.mhc_q    = nn.ModuleList([MHC(d_input, n_mhc) for _ in range(n_q_experts)])

        # ── KV-expert params (LoRA_k/v + alpha/beta + mHC) ──
        self.lora_A_k = nn.Parameter(torch.randn(n_kv_experts, d_input, r) * 0.01)
        self.lora_B_k = nn.Parameter(torch.zeros(n_kv_experts, r, d_key))
        self.lora_A_v = nn.Parameter(torch.randn(n_kv_experts, d_input, r) * 0.01)
        self.lora_B_v = nn.Parameter(torch.zeros(n_kv_experts, r, d_value))

        d_alpha = int(d_key * 1.618)
        self.alpha_up_w   = nn.Parameter(torch.randn(n_kv_experts, d_input, d_alpha) * 0.01)
        self.alpha_down_w = nn.Parameter(torch.randn(n_kv_experts, d_alpha, self.dk_pope) * 0.01)
        self.beta_up_w    = nn.Parameter(torch.randn(n_kv_experts, d_input, d_alpha) * 0.01)
        self.beta_down_w  = nn.Parameter(torch.randn(n_kv_experts, d_alpha, 1) * 0.01)

        self.mhc_kv      = nn.ModuleList([MHC(d_input, n_mhc) for _ in range(n_kv_experts)])

        # ── Output path: preGate(KV) → W_o → postGate(Q) ──
        concat_dim = n_kv_experts * d_value              # Ek * dv
        self.W_pre = nn.Linear(d_input, concat_dim, bias=False)
        self.W_o   = nn.Linear(concat_dim, d_input, bias=False)
        d_pg = int(concat_dim * 0.618)
        self.W_pg1 = nn.Linear(d_input, d_pg, bias=False)
        self.W_pg2 = nn.Linear(d_pg, d_input, bias=False)

        # ── Routers ──
        self.router_q  = nn.Linear(d_input, n_q_experts, bias=False)
        self.router_kv = nn.Linear(d_input, n_kv_experts, bias=False)

    def _apply_pope(self, x, positions, is_query):
        mu = F.softplus(x)
        phi = positions.unsqueeze(1) * self.freqs.to(x.device).unsqueeze(0)
        if not is_query:
            phi = phi + (-2 * math.pi * torch.sigmoid(self.pope_delta_raw))
        real = mu * torch.cos(phi)
        imag = mu * torch.sin(phi)
        return torch.cat([real, imag], dim=-1)

    @torch.compiler.disable
    def _kda_recursion(self, q, k, v, alpha, beta, T, S_init=None):
        B = q.shape[0]
        S = S_init if S_init is not None else q.new_zeros(B, self.dk_pope, self.dv)
        out = q.new_empty(B, T, self.dv)
        for t in range(T):
            out[:, t], S = _kda_step(q[:, t], k[:, t], v[:, t],
                                     alpha[:, t], beta[:, t], S)
        return out, S

    def forward(self, normed_stream, S_init=None):
        """
        normed_stream: (B, n_mhc, T, d) pre-normed
        Returns: stream_update (B, n_mhc, T, d), S_new
        """
        B, n, T, d = normed_stream.shape
        Eq = self.n_q_experts
        Ek = self.n_kv_experts
        positions = torch.arange(T, device=normed_stream.device)

        # ── Token-wise routing ──
        route_input = normed_stream.mean(dim=1)          # (B, T, d)
        gate_q  = top_prob_max_k(self.router_q(route_input),
                                  self.q_top_prob, self.q_max_k)   # (B, T, Eq)
        gate_kv = top_prob_max_k(self.router_kv(route_input),
                                  self.kv_top_prob, self.kv_max_k)  # (B, T, Ek)

        # ── Phase 1a: Q-experts → accumulate weighted q ──
        q_total  = normed_stream.new_zeros(B, T, self.dk_pope)
        h_q_list = []
        H_res_q  = []
        H_post_q = []

        for e in range(Eq):
            H_res, H_pre, H_post = self.mhc_q[e](normed_stream)
            h_e = torch.einsum('btn,bntd->btd', H_pre, normed_stream)
            h_q_list.append(h_e)
            H_res_q.append(H_res)
            H_post_q.append(H_post)

            delta_q = F.silu(h_e @ self.lora_A_q[e]) @ self.lora_B_q[e]
            q_e = self._apply_pope(
                F.normalize(self.W_q(h_e) + delta_q, dim=-1), positions, True)

            g = gate_q[:, :, e].unsqueeze(-1)            # (B, T, 1)
            q_total += g * q_e

        # h_q_routed → postGate input (Q-bound)
        h_q_routed = (torch.stack(h_q_list)                  # (Eq, B, T, d)
                      * gate_q.permute(2, 0, 1).unsqueeze(-1)
                      ).sum(dim=0)                            # (B, T, d)

        # ── Phase 1b: KV-experts → accumulate weighted k/v/alpha/beta ──
        k_total     = normed_stream.new_zeros(B, T, self.dk_pope)
        v_total     = normed_stream.new_zeros(B, T, self.dv)
        alpha_total = normed_stream.new_zeros(B, T, self.dk_pope)
        beta_total  = normed_stream.new_zeros(B, T, 1)
        h_kv_list   = []
        H_res_kv    = []
        H_post_kv   = []

        for e in range(Ek):
            H_res, H_pre, H_post = self.mhc_kv[e](normed_stream)
            h_e = torch.einsum('btn,bntd->btd', H_pre, normed_stream)
            h_kv_list.append(h_e)
            H_res_kv.append(H_res)
            H_post_kv.append(H_post)

            delta_k = F.silu(h_e @ self.lora_A_k[e]) @ self.lora_B_k[e]
            k_e = self._apply_pope(
                F.normalize(self.W_k(h_e) + delta_k, dim=-1), positions, False)

            delta_v = F.silu(h_e @ self.lora_A_v[e]) @ self.lora_B_v[e]
            v_e = F.silu(self.W_v(h_e) + delta_v)

            alpha_e = torch.sigmoid(
                F.silu(h_e @ self.alpha_up_w[e]) @ self.alpha_down_w[e])
            beta_e = torch.sigmoid(
                F.silu(h_e @ self.beta_up_w[e]) @ self.beta_down_w[e])

            g = gate_kv[:, :, e].unsqueeze(-1)
            k_total     += g * k_e
            v_total     += g * v_e
            alpha_total += g * alpha_e
            beta_total  += g * beta_e

        # h_kv_routed → preGate input (KV-bound)
        h_kv_routed = (torch.stack(h_kv_list)                # (Ek, B, T, d)
                       * gate_kv.permute(2, 0, 1).unsqueeze(-1)
                       ).sum(dim=0)                           # (B, T, d)

        # ── Phase 2: single KDA pass (Delta Rule unchanged) ──
        out, S_new = self._kda_recursion(
            q_total, k_total, v_total, alpha_total, beta_total, T, S_init)

        # ── Phase 3: preGate(KV) → W_o → postGate(Q) → mHC stream mixing ──
        # heads: gate_kv-weighted copies of KDA output, each expert a "head"
        heads = (out.unsqueeze(-2) * gate_kv.unsqueeze(-1)).reshape(
            B, T, Ek * self.dv)                              # (B, T, Ek*dv)

        pre_gate = F.silu(self.W_pre(h_kv_routed))          # (B, T, Ek*dv)
        gated = heads * pre_gate

        proj = self.W_o(gated)                               # (B, T, d)

        post_gate = torch.sigmoid(
            self.W_pg2(F.silu(self.W_pg1(h_q_routed))))     # (B, T, d)
        result = proj * post_gate                            # (B, T, d)

        # mHC stream mixing: H_res residual + H_post output
        stream_update = normed_stream.new_zeros(B, n, T, d)

        for e in range(Eq):
            g = gate_q[:, :, e].view(B, 1, T, 1)
            res = torch.einsum('btij,bjtd->bitd', H_res_q[e], normed_stream)
            post = torch.einsum('btn,btd->bntd', H_post_q[e], result)
            stream_update += g * (res + post)

        for e in range(Ek):
            g = gate_kv[:, :, e].view(B, 1, T, 1)
            res = torch.einsum('btij,bjtd->bitd', H_res_kv[e], normed_stream)
            post = torch.einsum('btn,btd->bntd', H_post_kv[e], result)
            stream_update += g * (res + post)

        return stream_update, S_new


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

    def forward(self, normed_stream):
        """
        normed_stream: (B, n_mhc, T, d)
        Returns: stream_update (B, n_mhc, T, d), gate_moe (B, T, E)
        """
        B, n, T, d = normed_stream.shape
        E = self.n_experts

        route_input = normed_stream.mean(dim=1)           # (B, T, d)
        gate_moe = top_prob_max_k(self.expert_router(route_input),
                                   self.top_prob, self.max_k)  # (B, T, E)

        stream_update = normed_stream.new_zeros(B, n, T, d)

        for e in range(E):
            H_res, H_pre, H_post = self.mhc_swiglu[e](normed_stream)
            h_e = torch.einsum('btn,bntd->btd', H_pre, normed_stream)
            out_e = self.experts[e](h_e)                  # (B, T, d)

            g = gate_moe[:, :, e].view(B, 1, T, 1)
            res = torch.einsum('btij,bjtd->bitd', H_res, normed_stream)
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
        # 可学习的退出线性偏置 α·T
        self.iter_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, stream, iter_idx):
        """
        stream: (B, n, T, d)
        iter_idx: (B,) long tensor — current iteration index
        """
        B, n, T, d = stream.shape
        last = stream[:, :, -1, :]
        x = self.norm(last.reshape(B, n * d))
        x = self.mix(x)
        x = x.reshape(B, n, d).sum(dim=1)               # (B, d)

        logits = self.proj(x)                             # (B, 2)
        logits[:, 1] = logits[:, 1] + F.silu(self.iter_alpha) * iter_idx.float()
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
                 max_iterations=12, n_mhc=4, min_iterations=1,
                 n_q_experts=16, n_kv_experts=16, n_ffn_experts=16,
                 q_top_prob=0.8, q_max_k=4,
                 kv_top_prob=0.8, kv_max_k=4,
                 ffn_top_prob=0.8, ffn_max_k=4):
        super().__init__()
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.n_mhc = n_mhc
        self.d = d_hidden

        self.inp_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.RMSNorm(d_hidden))

        # Learned [EOS] embedding — appended after the sequence
        self.eos = nn.Parameter(torch.randn(d_hidden) * 0.02)

        # Shared pre-norms (applied before MoA/MoE)
        self.stream_norms = nn.ModuleList([
            nn.RMSNorm(d_hidden) for _ in range(2)])     # [attn, swiglu]

        # MoA-KDA + MoE-SwiGLU
        self.moa_kda = MoAKDALayer(d_hidden, n_q_experts=n_q_experts,
                                    n_kv_experts=n_kv_experts, n_mhc=n_mhc,
                                    q_top_prob=q_top_prob, q_max_k=q_max_k,
                                    kv_top_prob=kv_top_prob, kv_max_k=kv_max_k)
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

    def forward(self, x, kda_states=None):
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(0)
        B, T, _ = x.shape
        n = self.n_mhc
        d = self.d

        kda_state = kda_states[0] if kda_states is not None else None

        # Append learned [EOS] at the end
        eos_emb = self.eos.view(1, 1, d).expand(B, 1, d)
        v0 = self.inp_proj(x)
        v0 = torch.cat([v0, eos_emb], dim=1)            # (B, T+1, d)
        T_total = T + 1

        stream = torch.zeros(B, n, T_total, d, device=x.device)
        stream[:, 0] = v0

        acc = [stream]
        active = torch.ones(B, dtype=torch.bool, device=x.device)
        exit_iter = torch.full((B,), float(self.max_iterations), device=x.device)
        sample_depth = torch.zeros(B, dtype=torch.long, device=x.device)

        for i in range(self.max_iterations):
            # ── STE route ── iter_t 冻结已退出样本，仅活跃样本递增
            iter_t = sample_depth
            logits = self.router(stream, iter_idx=iter_t)
            ste, hard = _ste_route(logits)

            if i < self.min_iterations:
                ste = active.float()
                hard = torch.ones_like(hard)

            gate = ste * active.float()
            gate_4d = gate.view(B, 1, 1, 1)
            gate_kda = gate.view(B, 1, 1)

            # ── MoA Attention sub-layer ──
            normed = self.stream_norms[0](stream)
            attn_update, S_new = self.moa_kda(normed, S_init=kda_state)
            updated = attn_update                            # replaces stream

            # ── MoE SwiGLU sub-layer ──
            normed2 = self.stream_norms[1](updated)
            ffn_update, gate_moe = self.moe_swiglu(normed2)
            updated = ffn_update

            # Per-expert AttnRes (additive)
            w_combined = torch.einsum('bte,ed->btd',
                                      gate_moe, self.moe_swiglu.w_experts)
            updated = updated + self._attn_res(
                acc, w_combined, self.preH_iter).unsqueeze(1)

            # ── Gate: active update, exited spin ──
            stream = gate_4d * updated + (1 - gate_4d) * stream
            if kda_state is not None:
                kda_state = gate_kda * S_new + (1 - gate_kda) * kda_state
            else:
                kda_state = S_new

            acc.append(stream)

            # ── Track exit iteration ──
            hard_cont = (hard > 0.5)
            newly_exited = active & (~hard_cont)
            exit_iter = torch.where(newly_exited,
                                    torch.tensor(float(i + 1), device=x.device),
                                    exit_iter)
            active = active & hard_cont
            sample_depth = sample_depth + active.long()
            if not active.any():
                break

        new_kda_states = [kda_state]
        h_out = self._attn_res(acc, self.w_final, self.preH_final)
        out = self.head_down(F.silu(self.head_gate(h_out)) * self.head_up(h_out))
        if squeeze:
            out = out.squeeze(0)
        return out, new_kda_states, exit_iter
