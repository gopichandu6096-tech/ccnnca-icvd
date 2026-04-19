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
        self.rff = RFFTransformer(input_dim, rff_dim, sigma=sigma, learnable=True)
        self.A = nn.Parameter(torch.randn(2 * rff_dim, output_dim) * 0.01)

    def convexified_attention(self, Q):
        split_size = Q.shape[-1] // self.rff_dim
        split_size = split_size if split_size > 0 else Q.shape[-1]
        Q_list = Q.split(split_size, dim=-1)
        scores = []
        A_mean = self.A.mean(dim=-1)
        for Q_i in Q_list:
            trace_val = torch.mean(
                torch.sum(Q_i * A_mean[:Q_i.shape[-1]].unsqueeze(0), dim=-1)
            )
            scaled = trace_val / (self.attention_scale * np.sqrt(self.rff_dim))
            scores.append(scaled)
        scores_tensor = torch.stack(scores)
        alpha = torch.softmax(scores_tensor, dim=0)
        Q_tilde = sum(alpha[i] * Q_list[i] for i in range(len(Q_list)))
        return Q_tilde, alpha

    def forward(self, Z):
        Q = self.rff(Z)
        Q_tilde, alpha = self.convexified_attention(Q)
        AW = torch.softmax(Q_tilde @ self.A / np.sqrt(self.rff_dim), dim=-1)
        predictions = Q_tilde @ self.A
        nuclear_norm = torch.linalg.matrix_norm(self.A, ord="nuc")
        reg_loss = self.lambda_reg * nuclear_norm
        return predictions, AW, reg_loss, alpha

    def save_checkpoint(self, path):
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