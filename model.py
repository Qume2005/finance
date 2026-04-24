"""Mini-KDA Policy Network with PoPE, AttnRes, SwiGLU."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _kda_step(qt, kt, vt, at, bt, S):
    """
    One step of KDA delta rule.
    Shapes are all fixed: qt(32), kt(32), vt(16), at(32), bt(1), S(32,16).
    """
    aS = at.unsqueeze(1) * S                    # (2*dk, dv)
    ktaS = kt.unsqueeze(0) @ aS                 # (1, dv)
    S_new = aS - bt * kt.unsqueeze(1) * ktaS + bt * kt.unsqueeze(1) * vt.unsqueeze(0)
    ot = S_new.T @ qt                           # (dv,)
    return ot, S_new


class MiniKDALayer(nn.Module):
    """
    Mini Kimi Delta Attention (from Kimi Linear, arXiv 2510.26692).

    Recurrence (delta rule + channel-wise gate):
        S_t = (I − β_t k_t k_t^⊤) Diag(α_t) S_{t−1} + β_t k_t v_t^⊤
        o_t = S_t^⊤ q_t

    α_t : per-dimension forgetting (fine-grained gating)
    β_t : update strength
    """
    def __init__(self, d_input, d_key=16, d_value=16):
        super().__init__()
        self.dk, self.dv = d_key, d_value
        self.dk_pope = d_key * 2  # PoPE doubles the effective dimension
        self.theta_base = 10000.0
        self.W_q = nn.Linear(d_input, d_key, bias=False)
        self.W_k = nn.Linear(d_input, d_key, bias=False)
        self.W_v = nn.Linear(d_input, d_value, bias=False)
        # PoPE learnable phase bias δ_c (initialized to 0, constrained via sigmoid in forward)
        self.pope_delta_raw = nn.Parameter(torch.zeros(d_key))
        # channel-wise gate α — output matches dk_pope
        self.W_alpha = nn.Sequential(
            nn.Linear(d_input, d_key * 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_key * 2, self.dk_pope, bias=False),
        )
        # update gate β
        self.W_beta = nn.Linear(d_input, 1, bias=False)
        # post-KDA: Norm → gate → linear (+ residual)
        self.post_norm  = nn.RMSNorm(d_value)
        self.out_gate   = nn.Linear(d_input, d_value, bias=False)
        self.W_out      = nn.Linear(d_value, d_input, bias=False)
        # SwiGLU FFN (+ residual)
        self.ffn_norm   = nn.RMSNorm(d_input)
        self.ffn_gate   = nn.Linear(d_input, d_input, bias=False)
        self.ffn_up     = nn.Linear(d_input, d_input, bias=False)
        self.ffn_down   = nn.Linear(d_input, d_input, bias=False)

    def _apply_pope(self, x, positions, is_query):
        """
        PoPE: magnitude = softplus(content), phase = position × frequency.
        Returns (T, 2*dk) with [real, imag] concatenation.
        """
        mu = F.softplus(x)                                                  # (T, dk) content magnitude
        freqs = self.theta_base ** (torch.arange(self.dk, device=x.device).float() / self.dk)  # (dk,)
        phi = positions.unsqueeze(1) * freqs.unsqueeze(0)                   # (T, dk)
        if not is_query:
            # Learnable bias δ_c ∈ [-2π, 0] via sigmoid parameterization
            phi = phi + (-2 * math.pi * torch.sigmoid(self.pope_delta_raw))  # (T, dk)
        real = mu * torch.cos(phi)                                          # (T, dk)
        imag = mu * torch.sin(phi)                                          # (T, dk)
        return torch.cat([real, imag], dim=-1)                              # (T, 2*dk)

    def forward(self, x_seq):
        """x_seq: (T, d_input) → (T, d_input)"""
        T, d = x_seq.shape
        positions = torch.arange(T, device=x_seq.device)
        # PoPE-transformed q and k: content (magnitude) and position (phase) decoupled
        q = self._apply_pope(self.W_q(x_seq), positions, is_query=True)    # (T, 2*dk)
        k = self._apply_pope(self.W_k(x_seq), positions, is_query=False)   # (T, 2*dk)
        v = F.silu(self.W_v(x_seq))                                        # (T, dv)
        alpha = torch.sigmoid(self.W_alpha(x_seq))                         # (T, 2*dk)
        beta  = torch.sigmoid(self.W_beta(x_seq))                          # (T, 1)

        S = x_seq.new_zeros(self.dk_pope, self.dv)
        outputs = []
        for t in range(T):
            ot, S = _kda_step(q[t], k[t], v[t], alpha[t], beta[t], S)
            outputs.append(ot)
        out = torch.stack(outputs)                       # (T, dv)

        # Norm → ×gate → linear + residual
        out = self.post_norm(out)                                  # (T, dv)
        out = out * torch.sigmoid(self.out_gate(x_seq))            # gate from input
        x_seq = x_seq + self.W_out(out)                            # residual

        # SwiGLU FFN + residual
        h = self.ffn_norm(x_seq)
        x_seq = x_seq + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x_seq


class KDAPolicyNetwork(nn.Module):
    """
    Stacked Mini-KDA with Attention Residuals (AttnRes, arXiv 2603.15031).
    Replaces standard residual accumulation with softmax attention over depth.
    Input : (T, 14)  MMn diff features
    Output: (T, 11)  logits for positions 0-10
    """
    def __init__(self, d_input=14, d_hidden=32, n_actions=11, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.inp_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden), nn.SiLU(), nn.RMSNorm(d_hidden))
        self.kda_layers = nn.ModuleList([
            MiniKDALayer(d_hidden, d_key=16, d_value=16)
            for _ in range(n_layers)])
        # AttnRes: one pseudo-query w_l ∈ R^d per layer + one for output
        # Initialized to zero so initial attention weights are uniform (per paper)
        self.attn_res_norm = nn.RMSNorm(d_hidden)
        self.w = nn.ParameterList([
            nn.Parameter(torch.zeros(d_hidden))
            for _ in range(n_layers + 1)])
        # SwiGLU policy head
        self.head_gate = nn.Linear(d_hidden, d_hidden, bias=False)
        self.head_up   = nn.Linear(d_hidden, d_hidden, bias=False)
        self.head_down = nn.Linear(d_hidden, n_actions, bias=False)

    def _attn_res(self, outputs, w_l):
        """
        AttnRes aggregation: h_l = Σ softmax(w_l · RMSNorm(v_i)) · v_i
        outputs: list of (T, d) tensors from previous layers
        w_l:     (d,) learnable pseudo-query for this layer
        """
        V = torch.stack(outputs)                          # (L_prev, T, d)
        K = self.attn_res_norm(V)                         # (L_prev, T, d)
        logits = torch.einsum('d,ltd->lt', w_l, K)       # (L_prev, T)
        alpha = logits.softmax(0)                         # (L_prev, T)
        return torch.einsum('lt,ltd->td', alpha, V)      # (T, d)

    def forward(self, x):
        v = [self.inp_proj(x)]                            # v_0 = embedding
        for l in range(self.n_layers):
            h_l = self._attn_res(v, self.w[l]) if l > 0 else v[0]
            v.append(self.kda_layers[l](h_l))
        # Final AttnRes over all layer outputs
        h_out = self._attn_res(v, self.w[self.n_layers])
        return self.head_down(F.silu(self.head_gate(h_out)) * self.head_up(h_out))
