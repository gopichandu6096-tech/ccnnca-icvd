from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List

from src.data.data_manager import PARAM_NAMES


class InterpretabilityModule:
    """
    Extracts global attention weights from a trained CCNNCAModel
    and produces publication-quality bar chart visualisations.
    """

    def __init__(self, model: torch.nn.Module,
                 param_names: List[str] = PARAM_NAMES,
                 output_dir: str = "outputs/"):
        self.model       = model
        self.param_names = param_names
        self.output_dir  = output_dir
        self.device      = next(model.parameters()).device
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def extract_attention_weights(self, Z_all: np.ndarray) -> np.ndarray:
        """
        Compute global feature importance averaged across all samples.

        Returns:
            importance : [14] normalised attention weights per parameter
        """
        self.model.eval()
        Z_tensor = torch.tensor(Z_all, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, AW, _, alpha = self.model(Z_tensor)

        # alpha shape: [n_splits] — map back to 14 parameters
        global_alpha = alpha.cpu().numpy()
        n_params     = len(self.param_names)

        param_importance = np.zeros(n_params)
        for i, imp in enumerate(global_alpha):
            param_importance[i % n_params] += imp

        param_importance /= param_importance.sum()
        return param_importance

    # ------------------------------------------------------------------ #
    def get_priority_tier(self, weight: float) -> str:
        if weight > 0.10:
            return "High"
        elif weight > 0.05:
            return "Medium"
        return "Low"

    # ------------------------------------------------------------------ #
    def plot_attention_weights(self, importance: np.ndarray,
                                save_name: str = "attention_weights.png",
                                dpi: int = 300) -> plt.Figure:
        """
        Generate attention weight bar chart sorted by importance.
        Blue = High (>0.10), Green = Medium (0.05-0.10), Grey = Low (<0.05)
        """
        sorted_idx  = np.argsort(importance)[::-1]
        sorted_imp  = importance[sorted_idx]
        sorted_names = [self.param_names[i] for i in sorted_idx]

        colors = ["#2196F3" if w > 0.10 else "#4CAF50" if w > 0.05
                  else "#9E9E9E" for w in sorted_imp]

        fig, ax = plt.subplots(figsize=(13, 6))
        bars = ax.bar(range(len(sorted_imp)), sorted_imp,
                       color=colors, edgecolor="white", linewidth=0.8)

        ax.set_xticks(range(len(sorted_imp)))
        ax.set_xticklabels(sorted_names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Attention Weight (αᵢ)", fontsize=12)
        ax.set_title("CCNNCA Feature Importance — iCVD Process Parameters",
                      fontsize=14, fontweight="bold", pad=15)
        ax.set_ylim(0, sorted_imp.max() * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        for bar, w in zip(bars, sorted_imp):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, f"{w:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        legend_patches = [
            mpatches.Patch(color="#2196F3", label="High priority (>0.10)"),
            mpatches.Patch(color="#4CAF50", label="Medium priority (0.05–0.10)"),
            mpatches.Patch(color="#9E9E9E", label="Low priority (<0.05)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[InterpretabilityModule] Plot saved → {save_path}")
        return fig

    # ------------------------------------------------------------------ #
    def print_ranking_table(self, importance: np.ndarray):
        """Print formatted feature importance ranking to console."""
        sorted_idx = np.argsort(importance)[::-1]
        print("\n" + "="*70)
        print(f"  {'Rank':<6} {'Parameter':<42} {'Weight':>7}  {'Tier'}")
        print("="*70)
        for rank, idx in enumerate(sorted_idx, 1):
            tier  = self.get_priority_tier(importance[idx])
            emoji = "🔴" if tier == "High" else "🟡" if tier == "Medium" else "⚪"
            print(f"  {rank:<6} {self.param_names[idx]:<42} "
                  f"{importance[idx]:>7.4f}  {emoji} {tier}")
        print("="*70)