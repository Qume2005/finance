"""Muon optimizer: momentum + Newton-Schulz gradient orthogonalization.

Reference: Bernstein & Newhouse 2024 (modded-nanogpt)
Only supports 2D parameters (weight matrices).

NewtonMuon extension: arXiv 2604.01472
Adds right-preconditioning via input activation second moment (ZZ^T)^{-1}.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer


# Newton-Schulz 5-term iteration coefficients (optimal for m >= n)
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


class Muon(Optimizer):
    """Muon optimizer for 2D parameters.

    Applies momentum to the gradient, then orthogonalizes via Newton-Schulz
    iterations before the parameter update.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 nesterov=True, ns_iters=5, eps=1e-7):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_iters=ns_iters, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            ns_iters = group['ns_iters']
            eps = group['eps']
            a, b, c = _NS_COEFFS

            for p in group['params']:
                if p.grad is None:
                    continue

                G = p.grad
                assert G.dim() == 2, "Muon only supports 2D parameters"

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(G)

                state['step'] += 1
                buf = state['momentum_buffer']

                # Momentum
                buf.mul_(mu).add_(G)
                M = buf

                # Nesterov: use lookahead gradient
                if nesterov:
                    M = M * mu + G

                # Newton-Schulz orthogonalization (FP32 for stability)
                X = M.float()
                remainder = max(X.shape[0] - X.shape[1], 0)
                scale = max(X.shape[0], X.shape[1])
                X = X / (X.norm() + eps)

                if X.shape[0] > X.shape[1]:
                    X = X.T  # work with n x m where n <= m

                for _ in range(ns_iters):
                    A = X.T @ X      # (n, n)
                    B = A @ A        # (n, n)
                    X = X @ (a * A + b * B + c * (B @ A))  # (n, m)

                if remainder > 0:
                    X = X.T  # back to (m, n)

                # Scale so that ||update|| ≈ ||G|| * sqrt(min(m,n))
                X = X * scale

                # Cast back to parameter dtype
                update = X.to(p.dtype)

                # Weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Update
                p.add_(update, alpha=-lr)

        return loss


class NewtonMuon(Muon):
    """Newton-Muon: standard Muon with right-preconditioning via (ZZ^T)^{-1}.

    Collects input activations via forward pre-hooks on Linear layers,
    maintains EWMA of ZZ^T/N, and applies damped inverse as right
    preconditioner to the gradient before the standard Muon pipeline.

    Reference: arXiv 2604.01472, Algorithm 1.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 nesterov=True, ns_iters=5, eps=1e-7,
                 beta_ewma=0.95, ridge_scale=0.2, refresh_interval=32):
        super().__init__(params, lr, momentum, weight_decay, nesterov, ns_iters, eps)
        self.beta_ewma = beta_ewma
        self.ridge_scale = ridge_scale
        self.refresh_interval = refresh_interval
        self._cached_ZZt = {}   # param_id -> (n, n) tensor
        self._hooks = []
        self._step_count = 0

    def register_hooks(self, model):
        """Register forward pre-hooks on Linear layers whose weights are in this optimizer.

        Each hook computes ZZ^T/N (small n×n matrix) from the layer input.
        """
        for h in self._hooks:
            h.remove()
        self._hooks = []

        param_ids = {id(p) for group in self.param_groups for p in group['params']}

        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and id(mod.weight) in param_ids:
                pid = id(mod.weight)

                def make_hook(param_id):
                    def hook(module, input):
                        x = input[0]  # (..., d_in)
                        x_flat = x.reshape(-1, x.shape[-1])   # (N, d_in)
                        ZZt = x_flat.T @ x_flat / x_flat.shape[0]  # (d_in, d_in)
                        self._cached_ZZt[param_id] = ZZt.detach()
                    return hook

                self._hooks.append(mod.register_forward_pre_hook(make_hook(pid)))

    @torch.no_grad()
    def update_preconditioner(self):
        """Update K and K_inv from cached ZZ^T via EWMA + damped Cholesky inverse.

        Call every `refresh_interval` steps, after the training forward pass.
        """
        for group in self.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid not in self._cached_ZZt:
                    continue

                state = self.state[p]
                ZZt = self._cached_ZZt[pid]  # (n, n)
                n = ZZt.shape[0]

                # Initialize K
                if 'K' not in state:
                    state['K'] = torch.eye(n, device=ZZt.device) * 1e-3

                K = state['K']
                # EWMA update
                K.mul_(self.beta_ewma).add_(ZZt, alpha=1 - self.beta_ewma)

                # Damped inverse: (K + γ*tr(K)/n * I)^{-1}
                gamma = self.ridge_scale * K.trace() / n
                K_damped = K + gamma * torch.eye(n, device=K.device)
                try:
                    L = torch.linalg.cholesky(K_damped)
                    state['K_inv'] = torch.cholesky_inverse(L)
                except RuntimeError:
                    # Fallback to full inverse if Cholesky fails
                    state['K_inv'] = torch.linalg.inv(K_damped)

        self._cached_ZZt.clear()

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            nesterov = group['nesterov']
            ns_iters = group['ns_iters']
            eps = group['eps']
            a, b, c = _NS_COEFFS

            for p in group['params']:
                if p.grad is None:
                    continue

                G = p.grad.clone()  # clone so we don't modify the raw grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(G)

                state['step'] += 1

                # Right precondition: G = G @ K_inv
                if 'K_inv' in state:
                    G = G @ state['K_inv']

                buf = state['momentum_buffer']

                # Momentum
                buf.mul_(mu).add_(G)
                M = buf

                # Nesterov: use lookahead gradient
                if nesterov:
                    M = M * mu + G

                # Newton-Schulz orthogonalization (FP32 for stability)
                X = M.float()
                remainder = max(X.shape[0] - X.shape[1], 0)
                scale = max(X.shape[0], X.shape[1])
                X = X / (X.norm() + eps)

                if X.shape[0] > X.shape[1]:
                    X = X.T  # work with n x m where n <= m

                for _ in range(ns_iters):
                    A = X.T @ X      # (n, n)
                    B = A @ A        # (n, n)
                    X = X @ (a * A + b * B + c * (B @ A))  # (n, m)

                if remainder > 0:
                    X = X.T  # back to (m, n)

                # Scale so that ||update|| ≈ ||G|| * sqrt(min(m,n))
                X = X * scale

                # Cast back to parameter dtype
                update = X.to(p.dtype)

                # Weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Update
                p.add_(update, alpha=-lr)

        return loss
