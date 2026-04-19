from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from .rff_transformer import RFFTransformer


class CCNNCAModel(nn.Module):
    """
    Full CCNNCA forward pass:
        Z → RFF → Convexified Attention → Linear Projection → [CA_water, CA_heptane, CA_octane]

    Args:
        input_dim       : Number of iCVD process parameters (14)
        rff_dim         : RFF feature dimensionality (256)
        output_dim      : Number of contact angle targets (3)
        lambda_reg      : Nuclear norm regularization strength
        attention_scale : Scaling factor for trace-based attention scores
        sigma           : RBF kernel bandwidth for RFF
    """

    def __init__(self, input_dim: int = 14, rff_dim: int = 256,
                 output_dim: int = 3, lambda_reg: float = 0.01,
                 attention_scale: float = 0.1, sigma: float = 1.0):
        super().__init__()
        self.rff_dim         = rff_dim
        self.output_dim      = output_dim
        self.lambda_reg      = lambda_reg
        self.attention_scale = attention_scale

        # Layer 1: RFF feature transformation
        self.rff = RFFTransformer(input_dim, rff_dim, sigma=sigma, learnable=True)

        # Layer 2: Learnable weight matrix A  (core CCNN parameter)
        # Shape: [2*rff_dim, output_dim]
        self.A = nn.Parameter(
            torch.randn(2 * rff_dim, output_dim) * 0.01
        )

    # ------------------------------------------------------------------ #
    def convexified_attention(self, Q: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute convexified attention over feature sub-matrices.

        Splits Q into sub-matrices Q_i, computes trace-based
        similarity scores Tr(Qᵢᵀ A), normalises with softmax to
        produce convex coefficients α, and returns Q̃ = Σ αᵢ Qᵢ.

        Returns:
            Q_tilde : [batch, 2*rff_dim]  weighted feature combination
            alpha   : [n_splits]          attention coefficients
        """
        # Split feature matrix into sub-matrices along feature axis
        split_size = Q.shape[-1] // self.rff_dim
        Q_list = Q.split(split_size if split_size > 0 else Q.shape[-1], dim=-1)

        scores = []
        A_mean = self.A.mean(dim=-1)   # [2*rff_dim]

        for Q_i in Q_list:
            # Tr(Qᵢᵀ · A̅) via element-wise mean — batch-stable
            trace_val = torch.mean(
                torch.sum(Q_i * A_mean[:Q_i.shape[-1]].unsqueeze(0), dim=-1)
            )
            scaled = trace_val / (self.attention_scale * np.sqrt(self.rff_dim))
            scores.append(scaled)

        scores_tensor = torch.stack(scores)               # [n_splits]
        alpha         = torch.softmax(scores_tensor, dim=0)

        Q_tilde = sum(alpha[i] * Q_list[i] for i in range(len(Q_list)))
        return Q_tilde, alpha

    # ------------------------------------------------------------------ #
    def forward(self, Z: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            Z: [batch_size, input_dim] — normalised process parameters in [0,1]

        Returns:
            predictions : [batch_size, 3]     contact angles (water/heptane/octane)
            AW          : [batch_size, 3]     attention weights (per-sample softmax)
            reg_loss    : scalar              nuclear norm regularisation term
            alpha       : [n_splits]          convexified attention coefficients
        """
        # Stage 1: RFF feature transformation
        Q = self.rff(Z)                                    # [B, 2*rff_dim]

        # Stage 2: Convexified attention
        Q_tilde, alpha = self.convexified_attention(Q)     # [B, 2*rff_dim], [n_splits]

        # Stage 3: Per-sample attention weights
        AW = torch.softmax(
            Q_tilde @ self.A / np.sqrt(self.rff_dim), dim=-1
        )                                                  # [B, output_dim]

        # Stage 4: Contact angle prediction
        predictions = Q_tilde @ self.A                     # [B, 3]

        # Stage 5: Nuclear norm regularisation  ‖A‖_* = Tr(√(AᵀA))
        nuclear_norm = torch.linalg.matrix_norm(self.A, ord="nuc")
        reg_loss     = self.lambda_reg * nuclear_norm

        return predictions, AW, reg_loss, alpha

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str):
        torch.save({"model_state": self.state_dict(),
                    "rff_dim":     self.rff_dim,
                    "output_dim":  self.output_dim,
                    "lambda_reg":  self.lambda_reg,
                    "attention_scale": self.attention_scale}, path)
        print(f"[CCNNCAModel] Checkpoint saved → {path}")

    @classmethod
    def load_checkpoint(cls, path: str, **override_kwargs) -> "CCNNCAModel":
        ckpt   = torch.load(path, map_location="cpu")
        kwargs = dict(rff_dim=ckpt["rff_dim"], output_dim=ckpt["output_dim"],
                      lambda_reg=ckpt["lambda_reg"],
                      attention_scale=ckpt["attention_scale"])
        kwargs.update(override_kwargs)
        model  = cls(**kwargs)
        model.load_state_dict(ckpt["model_state"])
        print(f"[CCNNCAModel] Checkpoint loaded ← {path}")
        return model