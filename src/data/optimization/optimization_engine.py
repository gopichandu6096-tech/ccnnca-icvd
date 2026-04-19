from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any


PARAM_BOUNDS_DEFAULT: List[Tuple[float, float]] = [
    (25.0, 50.0), (25.0, 60.0), (0.0, 100.0), (0.0, 1.0),
    (0.1, 15.0),  (0.1, 0.3),   (0.1, 0.5),   (15.0, 35.0),
    (1800, 3600), (0.0, 1.0),   (600, 1800),  (0.0, 0.5),
    (300, 900),   (0.0, 0.2),
]


class OptimizationEngine:
    """
    Multi-method process condition optimizer.

    Evaluates 5 SciPy methods (SLSQP, trust-constr, COBYLA,
    Nelder-Mead, L-BFGS-B), each with n_starts random starting points.
    """

    METHODS = ["SLSQP", "trust-constr", "COBYLA", "Nelder-Mead", "L-BFGS-B"]

    def __init__(self, model: torch.nn.Module,
                 param_bounds: List[Tuple[float, float]] = PARAM_BOUNDS_DEFAULT,
                 weights: List[float] | None = None,
                 scaler=None,
                 n_starts: int = 10,
                 max_iter: int = 1000,
                 ftol: float = 1e-9):
        self.model   = model
        self.bounds  = param_bounds
        self.weights = weights or [1/3, 1/3, 1/3]
        self.scaler  = scaler
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.ftol     = ftol
        self.device   = next(model.parameters()).device

    # ------------------------------------------------------------------ #
    def _scale(self, Z_raw: np.ndarray) -> np.ndarray:
        """Scale raw parameters using the fitted MinMaxScaler."""
        if self.scaler is not None:
            return self.scaler.transform(Z_raw.reshape(1, -1)).flatten()
        # Manual min-max if no scaler supplied
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return (Z_raw - lo) / (hi - lo + 1e-9)

    # ------------------------------------------------------------------ #
    def objective(self, Z_raw: np.ndarray) -> float:
        """Negative weighted contact angle sum (SciPy minimises)."""
        Z_scaled = self._scale(Z_raw)
        Z_tensor = torch.tensor(Z_scaled, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            preds, _, _, _ = self.model(Z_tensor)
            ca = preds[0]
            value = sum(self.weights[i] * ca[i] for i in range(3))
        return -value.item()

    # ------------------------------------------------------------------ #
    def optimize_single(self, method: str) -> Dict[str, Any]:
        """
        Run multi-start optimisation for a single method.

        Returns the best result across all starting points.
        """
        best_result = None
        best_obj    = float("inf")
        converged   = 0

        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])

        for _ in range(self.n_starts):
            x0 = np.array([np.random.uniform(lo[i], hi[i])
                            for i in range(len(self.bounds))])

            opts: Dict[str, Any] = {"maxiter": self.max_iter}
            if method not in ("Nelder-Mead", "COBYLA"):
                opts["ftol"] = self.ftol

            try:
                res = minimize(
                    self.objective, x0,
                    method=method,
                    bounds=self.bounds if method != "COBYLA" else None,
                    options=opts,
                )
                if res.success and res.fun < best_obj:
                    best_obj    = res.fun
                    best_result = res
                    converged  += 1
            except Exception as e:
                print(f"  [{method}] start failed: {e}")

        return {
            "method":        method,
            "result":        best_result,
            "best_obj":      best_obj,
            "converged":     converged,
            "n_starts":      self.n_starts,
            "optimal_params": best_result.x if best_result else None,
            "predicted_ca":  self._predict_ca(best_result.x) if best_result else None,
        }

    # ------------------------------------------------------------------ #
    def optimize_all(self) -> Dict[str, Dict]:
        """
        Run all 5 optimisation methods and return comparison results.
        """
        all_results = {}
        print("\n[OptimizationEngine] Running 5-method multi-start optimisation...")
        for method in self.METHODS:
            print(f"  → {method} ({self.n_starts} starts) ...", end=" ", flush=True)
            r = self.optimize_single(method)
            all_results[method] = r
            ca = r["predicted_ca"]
            if ca is not None:
                print(f"  Water={ca[0]:.1f}°  Heptane={ca[1]:.1f}°  "
                      f"Octane={ca[2]:.1f}°  [{r['converged']}/{self.n_starts} converged]")
            else:
                print("  No convergence.")
        return all_results

    # ------------------------------------------------------------------ #
    def _predict_ca(self, Z_raw: np.ndarray) -> np.ndarray:
        Z_scaled = self._scale(Z_raw)
        Z_tensor = torch.tensor(Z_scaled, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            preds, _, _, _ = self.model(Z_tensor)
        return preds[0].cpu().numpy()

    # ------------------------------------------------------------------ #
    def print_summary(self, results: Dict[str, Dict],
                      param_names: List[str] | None = None):
        from src.data.data_manager import PARAM_NAMES
        names = param_names or PARAM_NAMES
        print("\n" + "="*65)
        print("  CCNNCA OPTIMISATION RESULTS SUMMARY")
        print("="*65)
        for method, r in results.items():
            if r["optimal_params"] is None:
                print(f"\n{method}: No convergence")
                continue
            ca = r["predicted_ca"]
            print(f"\n{method}  [{r['converged']}/{self.n_starts} runs converged]")
            print(f"  Predicted CAs → Water: {ca[0]:.2f}°  "
                  f"Heptane: {ca[1]:.2f}°  Octane: {ca[2]:.2f}°")
            print("  Optimal parameters:")
            for i, (name, val) in enumerate(zip(names, r["optimal_params"])):
                print(f"    {name:40s}: {val:.4f}")