from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from typing import List, Tuple, Iterator


PARAM_NAMES = [
    "Substrate Temp (°C)", "Reactor Wall Temp (°C)", "Glass Heater Power (%)",
    "Leak Flow Rate (sccm)", "Initiator Flow Rate (sccm)", "Monomer Flow Rate (sccm)",
    "Pressure Diff (torr)", "Filament Power (A·V)", "Deposition Time I (s)",
    "Inert Gas Flow I (sccm)", "Deposition Time II (s)", "Inert Gas Flow II (sccm)",
    "Deposition Time III (s)", "Inert Gas Flow III (sccm)",
]

FEATURE_COLS = [f"z{i}" for i in range(1, 15)]
TARGET_COLS  = ["water_ca", "heptane_ca", "octane_ca"]


class DataManager:
    """Handles all dataset operations for CCNNCA training."""

    def __init__(self, dataset_path: str, feature_cols: List[str] = FEATURE_COLS,
                 target_cols: List[str] = TARGET_COLS, random_seed: int = 42):
        self.dataset_path  = dataset_path
        self.feature_cols  = feature_cols
        self.target_cols   = target_cols
        self.random_seed   = random_seed
        self.scaler        = None
        self.X_raw: np.ndarray | None = None
        self.y_raw: np.ndarray | None = None
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        df = pd.read_csv(self.dataset_path)
        self.X_raw = df[self.feature_cols].values.astype(np.float32)
        self.y_raw = df[self.target_cols].values.astype(np.float32)
        print(f"[DataManager] Loaded {len(df)} samples | "
              f"{len(self.feature_cols)} features | {len(self.target_cols)} targets")

    # ------------------------------------------------------------------ #
    def _fit_transform(self, X_train: np.ndarray, X_val: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Fit scaler on training fold only — prevents data leakage."""
        scaler = MinMaxScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_v_scaled  = scaler.transform(X_val)
        return X_tr_scaled, X_v_scaled, scaler

    # ------------------------------------------------------------------ #
    def _stratification_key(self, y: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Composite quintile stratification key for multi-output regression."""
        keys = []
        for col in range(y.shape[1]):
            bins = np.quantile(y[:, col], np.linspace(0, 1, n_bins + 1))
            bins[0] -= 1e-9; bins[-1] += 1e-9
            keys.append(np.digitize(y[:, col], bins) - 1)
        strat = keys[0] * (n_bins ** 2) + keys[1] * n_bins + keys[2]
        return strat

    # ------------------------------------------------------------------ #
    def kfold_splits(self, k: int = 5) -> Iterator[dict]:
        """Yield k stratified folds with independently fitted scalers."""
        strat_key = self._stratification_key(self.y_raw)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(self.X_raw, strat_key)):
            X_tr, X_val, scaler = self._fit_transform(
                self.X_raw[tr_idx], self.X_raw[val_idx])
            yield {
                "fold":   fold,
                "X_train": X_tr,    "y_train": self.y_raw[tr_idx],
                "X_val":   X_val,   "y_val":   self.y_raw[val_idx],
                "train_idx": tr_idx, "val_idx": val_idx,
                "scaler":  scaler,
            }

    # ------------------------------------------------------------------ #
    def loocv_splits(self) -> Iterator[dict]:
        """Yield leave-one-out splits for final generalization assessment."""
        loo = LeaveOneOut()
        for tr_idx, val_idx in loo.split(self.X_raw):
            X_tr, X_val, scaler = self._fit_transform(
                self.X_raw[tr_idx], self.X_raw[val_idx])
            yield {
                "X_train": X_tr,    "y_train": self.y_raw[tr_idx],
                "X_val":   X_val,   "y_val":   self.y_raw[val_idx],
                "val_idx": val_idx, "scaler":  scaler,
            }

    # ------------------------------------------------------------------ #
    def full_scaled(self) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Return the full dataset scaled on all samples (for final inference)."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X_raw)
        self.scaler = scaler
        return X_scaled, self.y_raw, scaler

    # ------------------------------------------------------------------ #
    @property
    def n_samples(self) -> int:
        return len(self.X_raw)

    @property
    def n_features(self) -> int:
        return self.X_raw.shape[1]


# ─── Synthetic dataset generator (when real data unavailable) ─────────── #
def generate_synthetic_dataset(n_samples: int = 49, seed: int = 42,
                                 save_path: str = "data/ppfda_dataset.csv"):
   
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    bounds = [
        (25.0, 50.0), (25.0, 60.0), (0.0, 100.0), (0.0, 1.0),
        (0.1, 15.0),  (0.1, 0.3),   (0.1, 0.5),   (15.0, 35.0),
        (1800, 3600), (0.0, 1.0),   (600, 1800),  (0.0, 0.5),
        (300, 900),   (0.0, 0.2),
    ]

    X = np.column_stack([
        rng.uniform(lo, hi, n_samples) for lo, hi in bounds
    ])

    # Synthetic contact angles with known parameter importance
    z5 = (X[:, 4] - 0.1) / (15.0 - 0.1)   # initiator
    z2 = (X[:, 1] - 25.0) / (60.0 - 25.0)  # reactor wall temp
    z6 = (X[:, 5] - 0.1) / (0.3 - 0.1)     # monomer

    base = 100 + 30 * z5 + 20 * (1 - z2) + 15 * z6
    water_ca   = base + rng.normal(0, 5, n_samples)
    heptane_ca = base * 0.75 + rng.normal(0, 4, n_samples)
    octane_ca  = base * 0.70 + rng.normal(0, 4, n_samples)

    df = pd.DataFrame(X, columns=[f"z{i}" for i in range(1, 15)])
    df["water_ca"]   = np.clip(water_ca,   80, 170)
    df["heptane_ca"] = np.clip(heptane_ca, 60, 140)
    df["octane_ca"]  = np.clip(octane_ca,  55, 135)
    df.to_csv(save_path, index=False)
    print(f"[DataManager] Synthetic dataset saved → {save_path}")
    return df