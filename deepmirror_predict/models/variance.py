# src/automl_molops/modeling/feature_select.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VarianceThreshold:
    """
    Drops binary features (0/1) that are too rare or too common in the TRAIN set,
    using fractions instead of absolute counts.

    Keep feature j if:
      min_frac <= (#ones_j / n_train) <= max_frac

    Example:
      min_frac=0.02 keeps bits present in at least ~2% of train samples.
    """
    min_frac: float = 0.02
    max_frac: float = 0.98

    support_: np.ndarray | None = None
    n_train_: int | None = None

    def fit(self, X: np.ndarray) -> "VarianceThreshold":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if not (0.0 <= self.min_frac <= 1.0) or not (0.0 <= self.max_frac <= 1.0):
            raise ValueError("min_frac/max_frac must be in [0, 1]")
        if self.min_frac > self.max_frac:
            raise ValueError("min_frac must be <= max_frac")

        n = X.shape[0]
        self.n_train_ = n

        counts = np.sum(X > 0.5, axis=0)  # works for float32 0/1 arrays
        freqs = counts / float(n)

        self.support_ = (freqs >= self.min_frac) & (freqs <= self.max_frac)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.support_ is None:
            raise RuntimeError("Call fit() first")
        return np.asarray(X)[:, self.support_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_support(self) -> np.ndarray:
        if self.support_ is None:
            raise RuntimeError("Call fit() first")
        return self.support_