from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE calculation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE calculation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE calculation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 calculation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kendall's tau correlation coefficient."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    stat, _ = kendalltau(y_true, y_pred)
    return float(stat) if stat is not None and np.isfinite(stat) else 0.0


def kendall_tau_pvalue(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kendall's tau p-value."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _, p = kendalltau(y_true, y_pred)
    return float(p) if p is not None and np.isfinite(p) else 1.0


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    stat, _ = spearmanr(y_true, y_pred)
    return float(stat) if stat is not None and np.isfinite(stat) else 0.0


def spearman_pvalue(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation p-value."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _, p = spearmanr(y_true, y_pred)
    return float(p) if p is not None and np.isfinite(p) else 1.0
