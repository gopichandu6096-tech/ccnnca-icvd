from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from src.models.rff_transformer import RFFTransformer


class CCNNCAModel(nn.Module):
    def __init__(self, input_dim=14, rff_dim=256, output_dim=3,
                 lambda_reg=0.01, attention_scale=0.1, sigma=1.0):
        super().__init__()
        self.rff_dim = rff_dim
        self.output_dim = output_dim
        self.lambda_reg = lambda_reg
        self.attention_scale = attention_scale
        self.feature_dim = 2 * rff_dim
        self.rff = RFFTransformer(input_dim, rff_dim, sigma=sigma, learnable=True)
        self.A = nn.Parameter(torch.randn(self.feature_dim, output_dim) * 0.01)

    def convexified_attention(self, Q):
        # Q shape: [batch, feature_dim] where feature_dim = 2 * rff_dim
        # Split into n_heads sub-matrices along feature axis
        n_heads = 4
        feature_dim = Q.shape[-1]
        head_dim = feature_dim // n_heads

        # Ensure head_dim divides evenly
        if feature_dim % n_heads != 0:
            n_heads = 2
            head_dim = feature_dim // n_heads

        Q_list = Q.split(head_dim, dim=-1)
        scores = []

        for Q_i in Q_list:
            # Trace-based attention score for this head
            score = torch.mean(torch.norm(Q_i, dim=-1))
            scaled = score / (self.attention_scale * np.sqrt(feature_dim))
            scores.append(scaled)

        scores_tensor = torch.stack(scores)
        alpha = torch.softmax(scores_tensor, dim=0)

        # Weighted sum reconstructs full feature dim
        # Pad each Q_i back to full feature_dim using alpha weighting
        Q_tilde = torch.zeros_like(Q)
        start = 0
        for i, Q_i in enumerate(Q_list):
            end = start + Q_i.shape[-1]
            Q_tilde[:, start:end] = alpha[i] * Q_i
            start = end

        return Q_tilde, alpha

    def forward(self, Z):
        # Stage 1: RFF transformation -> [batch, 2*rff_dim]
        Q = self.rff(Z)

        # Stage 2: Convexified attention -> [batch, 2*rff_dim]
        Q_tilde, alpha = self.convexified_attention(Q)

        # Stage 3: Linear projection -> [batch, output_dim]
        predictions = Q_tilde @ self.A

        # Stage 4: Attention weights (per-sample softmax over outputs)
        AW = torch.softmax(predictions, dim=-1)

        # Stage 5: Nuclear norm regularisation
        nuclear_norm = torch.linalg.matrix_norm(self.A, ord="nuc")
        reg_loss = self.lambda_reg * nuclear_norm

        return predictions, AW, reg_loss, alpha

    def save_checkpoint(self, path):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "rff_dim": self.rff_dim,
            "output_dim": self.output_dim,
            "lambda_reg": self.lambda_reg,
            "attention_scale": self.attention_scale
        }, path)
        print(f"[CCNNCAModel] Checkpoint saved -> {path}")

    @classmethod
    def load_checkpoint(cls, path, **override_kwargs):
        ckpt = torch.load(path, map_location="cpu")
        kwargs = dict(
            rff_dim=ckpt["rff_dim"],
            output_dim=ckpt["output_dim"],
            lambda_reg=ckpt["lambda_reg"],
            attention_scale=ckpt["attention_scale"]
        )
        kwargs.update(override_kwargs)
        model = cls(**kwargs)
        model.load_state_dict(ckpt["model_state"])
        print(f"[CCNNCAModel] Checkpoint loaded <- {path}")
        return model