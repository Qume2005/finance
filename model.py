"""Mini-KDA Policy Network with PoPE, mHC, AttnRes, SwiGLU."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(M, n_iters=20):
    """Project (..., n, n) to doubly stochastic matrix."""
    M = torch.exp(M)
    for _ in range(n_iters):
        M = M / M.sum(dim=-1, keepdim=True)
        M = M / M.sum(dim=-2, keepdim=True)
    return M


class MHC(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC, arXiv 2512.24880).
    n-stream residual with doubly stochastic mixing.
    """
    def __init__(self, d_hidden, n=4):
        super().__init__()
        self.n = n
        nd = n * d_hidden
        self.phi_pre  = nn.Linear(nd, n, bias=False)
        self.phi_post = nn.Linear(nd, n, bias=False)
        self.phi_res  = nn.Linear(nd, n * n, bias=False)
        self.b_pre  = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res  = nn.Parameter(torch.zeros(n, n))
        self.alpha_pre  = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res  = nn.Parameter(torch.tensor(0.01))

    def forward(self, stream):
        """stream: (n, T, d) → H_res(T,n,n), H_pre(T,n), H_post(T,n)"""
        n, T, d = stream.shape
        x = stream.permute(1, 0, 2).reshape(T, n * d)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8)

        H_pre = torch.sigmoid(self.alpha_pre * self.phi_pre(x) + self.b_pre)
        H_post = 2 * torch.sigmoid(self.alpha_post * self.phi_post(x) + self.b_post)
        H_res_raw = self.alpha_res * self.phi_res(x).reshape(T, self.n, self.n) + self.b_res
        H_res = sinkhorn_knopp(H_res_raw)

        return H_res, H_pre, H_post


class MiniKDALayer(nn.Module):
    """KDA delta attention with PoPE, SwiGLU alpha gate."""
    def __init__(self, d_input, d_key=16, d_value=16):
        super().__init__()
        self.dk, self.dv = d_key, d_value
        self.dk_pope = d_key * 2
        self.theta_base = 10000.0
        self.W_q = nn.Linear(d_input, d_key, bias=False)
        self.W_k = nn.Linear(d_input, d_key, bias=False)
        self.W_v = nn.Linear(d_input, d_value, bias=False)
        self.pope_delta_raw = nn.Parameter(torch.zeros(d_key))
        d_alpha = int(d_key * 1.618)
        self.alpha_gate = nn.Linear(d_input, d_alpha, bias=False)
        self.alpha_up   = nn.Linear(d_input, d_alpha, bias=False)
        self.alpha_down = nn.Linear(d_alpha, self.dk_pope, bias=False)
        self.post_norm = nn.RMSNorm(d_value)
        d_d = int(d_value / 1.618)
        self.W_d = nn.Linear(d_input, d_d, bias=False)
        self.W_u = nn.Linear(d_d, d_value, bias=False)
        self.W_out = nn.Linear(d_value, d_input, bias=False)

    def _apply_pope(self, x, positions, is_query):
        mu = F.softplus(x)
        freqs = self.theta_base ** (torch.arange(self.dk, device=x.device).float() / self.dk)
        phi = positions.unsqueeze(1) * freqs.unsqueeze(0)
        if not is_query:
            phi = phi + (-2 * math.pi * torch.sigmoid(self.pope_delta_raw))
        real = mu * torch.cos(phi)
        imag = mu * torch.sin(phi)
        return torch.cat([real, imag], dim=-1)

    def forward(self, x_seq):
        T, d = x_seq.shape
        positions = torch.arange(T, device=x_seq.device)
        q = self._apply_pope(F.normalize(self.W_q(x_seq), dim=-1), positions, is_query=True)
        k = self._apply_pope(F.normalize(self.W_k(x_seq), dim=-1), positions, is_query=False)
        v = F.silu(self.W_v(x_seq))
        alpha = F.sigmoid(self.alpha_down(
            F.silu(self.alpha_gate(x_seq)) * self.alpha_up(x_seq)))
        S = x_seq.new_zeros(self.dk_pope, self.dv)
        out = x_seq.new_empty(T, self.dv)
        for t in range(T):
            aS = alpha[t].unsqueeze(1) * S
            S = aS - torch.outer(k[t], k[t] @ aS) + torch.outer(k[t], v[t])
            out[t] = q[t] @ S

        out = self.post_norm(out)
        out = out * torch.sigmoid(self.W_u(F.silu(self.W_d(x_seq))))
        return self.W_out(out)


class SwiGLU(nn.Module):
    """SwiGLU with bottleneck gate and RMSNorm."""
    def __init__(self, d_input):
        super().__init__()
        d_ffn = int(d_input * 1.618)
        self.norm = nn.RMSNorm(d_input)
        self.wd = nn.Linear(d_input, d_input, bias=False)
        self.wu = nn.Linear(d_input, d_ffn, bias=False)
        self.gate = nn.Linear(d_input, d_ffn, bias=False)
        self.up   = nn.Linear(d_input, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_input, bias=False)

    def forward(self, x):
        h = self.norm(x)
        g = torch.sigmoid(self.wu(F.silu(self.wd(h))))
        return self.down(g * (F.silu(self.gate(h)) * self.up(h)))


class KDAPolicyNetwork(nn.Module):
    """
    KDA + SwiGLU with mHC (n=4) and AttnRes (with pre W) depth aggregation.

    mHC: n-stream residual with Sinkhorn-Knopp doubly stochastic mixing
    AttnRes: softmax attention over depth with learned key projection (pre W)

    Input : (T, 14)  MMn diff features
    Output: (T, 11)  logits for positions 0-10
    """
    def __init__(self, d_input=14, d_hidden=32, n_actions=11, n_layers=8, n_mhc=4):
        super().__init__()
        self.n_layers = n_layers
        self.n_mhc = n_mhc
        self.d = d_hidden

        self.inp_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.RMSNorm(d_hidden))

        self.kda_layers = nn.ModuleList([
            MiniKDALayer(d_hidden) for _ in range(n_layers)])
        self.swiglu_layers = nn.ModuleList([
            SwiGLU(d_hidden) for _ in range(n_layers)])

        # mHC per sub-layer (attention + swiglu each have their own)
        self.mhc = nn.ModuleList([
            MHC(d_hidden, n_mhc) for _ in range(n_layers * 2)])

        # AttnRes with pre W key projection
        self.attn_res_norm = nn.RMSNorm(d_hidden)
        self.pre_w = nn.Linear(d_hidden, d_hidden, bias=False)
        self.w = nn.ParameterList([
            nn.Parameter(torch.zeros(d_hidden))
            for _ in range(n_layers * 2 + 1)])

        # Head
        d_head = int(d_hidden * 1.618)
        self.head_gate = nn.Linear(d_hidden, d_head, bias=False)
        self.head_up   = nn.Linear(d_hidden, d_head, bias=False)
        self.head_down = nn.Linear(d_head, n_actions, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Depth-scaled init: residual output projections × 1/√depth."""
        depth = self.n_layers * 2                          # 16 sub-layers
        scale = depth ** -0.5                              # 0.25
        for layer in self.kda_layers:
            nn.init.normal_(layer.W_out.weight, std=scale * (2.0 / layer.W_out.in_features) ** 0.5)
        for layer in self.swiglu_layers:
            nn.init.normal_(layer.down.weight, std=scale * (2.0 / layer.down.in_features) ** 0.5)
        nn.init.normal_(self.head_down.weight,
                        std=scale * (2.0 / self.head_down.in_features) ** 0.5)
        nn.init.normal_(self.pre_w.weight, std=scale)

    def _attn_res(self, acc, w_l):
        """AttnRes aggregation with pre W key projection."""
        V = torch.stack(acc)                          # (L, T, d)
        K = self.pre_w(self.attn_res_norm(V))         # (L, T, d)
        logits = torch.einsum('d,ltd->lt', w_l, K)
        alpha = logits.softmax(0)
        return torch.einsum('lt,ltd->td', alpha, V)

    def forward(self, x):
        n = self.n_mhc
        T = x.shape[0]
        d = self.d

        v0 = self.inp_proj(x)                         # (T, d)
        stream = torch.zeros(n, T, d, device=x.device)
        stream[0] = v0

        acc = [v0]

        for l in range(self.n_layers):
            # ── Attention ──
            H_res, H_pre, H_post = self.mhc[l * 2](stream)
            stream = torch.einsum('tij,jtd->itd', H_res, stream)   # mix
            h = torch.einsum('tn,ntd->td', H_pre, stream)          # read
            if len(acc) > 1:
                h = h + self._attn_res(acc, self.w[l * 2])
            out = self.kda_layers[l](h)
            stream = stream + torch.einsum('tn,td->ntd', H_post, out)  # write
            acc.append(out)

            # ── SwiGLU ──
            H_res, H_pre, H_post = self.mhc[l * 2 + 1](stream)
            stream = torch.einsum('tij,jtd->itd', H_res, stream)
            h = torch.einsum('tn,ntd->td', H_pre, stream)
            h = h + self._attn_res(acc, self.w[l * 2 + 1])
            out = self.swiglu_layers[l](h)
            stream = stream + torch.einsum('tn,td->ntd', H_post, out)
            acc.append(out)

        h_out = self._attn_res(acc, self.w[self.n_layers * 2])
        return self.head_down(F.silu(self.head_gate(h_out)) * self.head_up(h_out))
