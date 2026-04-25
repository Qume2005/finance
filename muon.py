"""Muon optimizer: momentum + Newton-Schulz gradient orthogonalization.

Reference: Bernstein & Newhouse 2024 (modded-nanogpt)
Only supports 2D parameters (weight matrices).
"""

import torch
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
