from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Any

from src.models.ccnnca_model import CCNNCAModel


class TrainingEngine:
    
    def __init__(self, model: CCNNCAModel, config: Dict[str, Any],
                 output_dir: str = "outputs/", device: str | None = None):
        self.model      = model
        self.config     = config
        self.output_dir = output_dir
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        os.makedirs(output_dir, exist_ok=True)

        tcfg = config.get("training", config)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tcfg.get("learning_rate", 0.001),
            weight_decay=tcfg.get("weight_decay", 0.01),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=tcfg.get("lr_factor", 0.5),
            patience=tcfg.get("lr_patience", 15),
            min_lr=tcfg.get("min_lr", 1e-6),
        )
        self.max_epochs    = tcfg.get("max_epochs", 300)
        self.patience      = tcfg.get("patience", 20)
        self.grad_clip     = tcfg.get("gradient_clip_norm", 1.0)
        self.batch_size    = tcfg.get("batch_size", 16)
        self.best_val_loss = float("inf")
        self.best_ckpt     = os.path.join(output_dir, "best_model.pt")

    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_loss(predictions: torch.Tensor, targets: torch.Tensor,
                     reg_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mse   = nn.functional.mse_loss(predictions, targets)
        total = mse + reg_loss
        return total, mse

    # ------------------------------------------------------------------ #
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray,   y_val: np.ndarray) -> Dict:
        Xt, yt = self._to_tensor(X_train), self._to_tensor(y_train)
        Xv, yv = self._to_tensor(X_val),   self._to_tensor(y_val)

        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        history = {"train_loss": [], "val_loss": [], "val_r2": []}
        patience_ctr = 0

        for epoch in range(self.max_epochs):
            # ── Train ──
            self.model.train()
            epoch_loss = 0.0
            for Z_batch, y_batch in loader:
                preds, _, reg_loss, _ = self.model(Z_batch)
                loss, _ = self.compute_loss(preds, y_batch, reg_loss)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                epoch_loss += loss.item()

            # ── Validate ──
            self.model.eval()
            with torch.no_grad():
                v_preds, _, v_reg, _ = self.model(Xv)
                val_loss, _ = self.compute_loss(v_preds, yv, v_reg)

            self.scheduler.step(val_loss)

            val_r2 = r2_score(yv.cpu().numpy(),
                               v_preds.cpu().numpy(),
                               multioutput="uniform_average")
            history["train_loss"].append(epoch_loss / len(loader))
            history["val_loss"].append(val_loss.item())
            history["val_r2"].append(val_r2)

            # ── Early stopping ──
            if val_loss.item() < self.best_val_loss:
                self.best_val_loss = val_loss.item()
                self.model.save_checkpoint(self.best_ckpt)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print(f"  Early stop @ epoch {epoch+1}")
                    break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:4d} | "
                      f"train_loss={epoch_loss/len(loader):.4f} | "
                      f"val_loss={val_loss.item():.4f} | val_R²={val_r2:.4f}")

        return history

    # ------------------------------------------------------------------ #
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Full per-target and combined metrics."""
        self.model.eval()
        XT = self._to_tensor(X)
        with torch.no_grad():
            preds, _, _, _ = self.model(XT)
        y_pred = preds.cpu().numpy()

        target_names = ["water", "heptane", "octane"]
        results: Dict[str, float] = {}
        for i, name in enumerate(target_names):
            results[f"{name}_r2"]   = r2_score(y[:, i], y_pred[:, i])
            results[f"{name}_mae"]  = mean_absolute_error(y[:, i], y_pred[:, i])
            results[f"{name}_rmse"] = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
            results[f"{name}_mse"]  = mean_squared_error(y[:, i], y_pred[:, i])

        results["combined_r2"]  = r2_score(y, y_pred, multioutput="uniform_average")
        results["combined_mse"] = mean_squared_error(y.ravel(), y_pred.ravel())
        results["avg_mae"]      = np.mean([results[f"{n}_mae"] for n in target_names])
        return results