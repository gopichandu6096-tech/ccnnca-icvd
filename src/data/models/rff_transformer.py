from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class RFFTransformer(nn.Module):
    """
    Maps input Z ∈ R^{batch × input_dim} to
    Q ∈ R^{batch × 2·rff_dim}  via random Fourier features.

    The transformation is:
        projection = Z @ W^T + b
        Q = [cos(projection) || sin(projection)]
    where W ~ N(0, 1/σ²) and b ~ U(0, 2π).
    """

    def __init__(self, input_dim: int = 14, rff_dim: int = 256,
                 sigma: float = 1.0, learnable: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.rff_dim   = rff_dim

        W_init = torch.randn(input_dim, rff_dim) * (1.0 / sigma)
        b_init = torch.zeros(rff_dim).uniform_(0, 2 * np.pi)

        if learnable:
            self.rff_weights = nn.Parameter(W_init)
            self.rff_bias    = nn.Parameter(b_init)
        else:
            self.register_buffer("rff_weights", W_init)
            self.register_buffer("rff_bias",    b_init)

    # ------------------------------------------------------------------ #
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: [batch_size, input_dim] normalized inputs in [0, 1]
        Returns:
            Q: [batch_size, 2 * rff_dim] RKHS feature matrix
        """
        projection = Z @ self.rff_weights + self.rff_bias   # [B, rff_dim]
        Q = torch.cat([torch.cos(projection),
                        torch.sin(projection)], dim=-1)       # [B, 2*rff_dim]
        return Q

    # ------------------------------------------------------------------ #
    def set_bandwidth(self, sigma: float):
        """Update RBF kernel bandwidth (rescales weight matrix)."""
        with torch.no_grad():
            self.rff_weights.mul_(0.0).add_(
                torch.randn_like(self.rff_weights) * (1.0 / sigma))

    # ------------------------------------------------------------------ #
    def verify_approximation(self, Z: torch.Tensor,
                               tol: float = 0.05) -> bool:
        """
        Validate RFF approximation quality by comparing empirical
        kernel matrix with exact RBF kernel.
        """
        with torch.no_grad():
            Q = self.forward(Z)
            K_approx = (Q @ Q.T) / (2 * self.rff_dim)
            # Exact RBF kernel
            diff = Z.unsqueeze(1) - Z.unsqueeze(0)          # [N,N,D]
            sq_dist = (diff ** 2).sum(-1)
            K_exact = torch.exp(-sq_dist / 2.0)
            mae = (K_approx - K_exact).abs().mean().item()
            print(f"[RFFTransformer] Kernel approximation MAE = {mae:.5f} "
                  f"(tol={tol}) → {'✓ OK' if mae < tol else '✗ FAIL'}")
            return mae < tol