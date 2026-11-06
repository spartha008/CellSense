"""Isolation Forest over per-cell features, scaled relative to the rest of the same pack."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from ..config import DATA_DIR

FEATURES = ("rest_offset_mv", "sag_mv", "r_ratio", "fade_slope", "sd_drift")
# smallest spread we trust for each feature, so a very uniform pack does not blow up the scale
FLOORS = np.array([1.0, 1.0, 0.01, 0.002, 0.2])
MODEL_FILE = DATA_DIR / "iforest.joblib"


def feature_matrix(feat: dict) -> np.ndarray:
    return np.column_stack([np.asarray(feat[k], dtype=float) for k in FEATURES])


def pack_scale(X: np.ndarray) -> np.ndarray:
    """Robust z-score of every feature against the pack's own median and MAD."""
    X = np.asarray(X, dtype=float).copy()
    X[:, np.all(~np.isfinite(X), axis=0)] = 0.0  # a feature with no data yet carries no information
    med = np.nanmedian(X, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(X - med), axis=0)
    mad = np.maximum(np.nan_to_num(mad, nan=0.0), FLOORS)
    Z = (X - med) / mad
    return np.clip(np.nan_to_num(Z, nan=0.0), -50, 50)


class AnomalyModel:
    def __init__(self, model: IsolationForest | None = None, threshold: float | None = None):
        self.model = model
        self.threshold = threshold  # calibrated alert level (99.5th percentile of healthy cells)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def fit(self, packs: list[np.ndarray], seed: int = 0) -> "AnomalyModel":
        X = np.vstack([pack_scale(p) for p in packs])
        self.model = IsolationForest(n_estimators=100, max_samples=256, random_state=seed).fit(X)
        self.threshold = float(np.quantile(np.concatenate([self.score(p) for p in packs]), 0.995))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score as defined in the Isolation Forest paper: ~0.5 normal, -> 1 anomalous.

        Walks the fitted trees directly: identical to ``-score_samples`` but ~20x faster for one
        pack of cells, because it skips scikit-learn's per-call parallel-dispatch overhead.
        """
        if not self.ready:
            return np.full(len(X), np.nan)
        Z = np.ascontiguousarray(pack_scale(X), dtype=np.float32)
        m = self.model
        dpl, apl = getattr(m, "_decision_path_lengths", None), getattr(m, "_average_path_length_per_tree", None)
        if dpl is None or apl is None:
            return -m.score_samples(Z)
        depths = np.zeros(len(Z))
        for tree, feats, d, a in zip(m.estimators_, m.estimators_features_, dpl, apl):
            leaves = tree.tree_.apply(np.ascontiguousarray(Z[:, feats]))
            depths += d[leaves] + a[leaves] - 1.0
        denom = len(m.estimators_) * _average_path_length([m._max_samples])[0]
        return 2.0 ** (-depths / denom)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "AnomalyModel":
        path = Path(path) if path else MODEL_FILE
        if not path.exists():
            return cls(None)
        d = joblib.load(path)
        return cls(d["model"], d.get("threshold")) if isinstance(d, dict) else cls(d)
