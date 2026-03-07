from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from autogluon.tabular import TabularPredictor

from .metrics import (
    rmse,
    mse,
    mae,
    r2,
    kendall_tau,
    kendall_tau_pvalue,
    spearman_rho,
    spearman_pvalue,
)


@dataclass(frozen=True)
class AutoGluonConfig:
    # Your requirement:
    presets: str = "best_quality"

    # Optional controls
    time_limit_s: Optional[int] = None
    eval_metric: Optional[str] = None  # e.g. "root_mean_squared_error", "mean_absolute_error", "r2"
    num_gpus: int | float = 0
    verbosity: int = 0

    # AutoGluon writes to disk; if None, AutoGluon uses its default AutogluonModels/ag-<timestamp>
    path: Optional[str] = None

    # Advanced (usually leave None with best_quality)
    hyperparameters: Optional[dict[str, Any]] = None

    use_bag_holdout: bool = True
    included_model_types: Optional[list[str]] = None

    dynamic_stacking: bool | str = False   # set False to avoid Ray
    ds_args: Optional[dict[str, Any]] = None  # e.g. {"memory_safe_fits": False}


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"AutoGluonRegressor expects 2D feature matrix, got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("AutoGluonRegressor requires numeric features (not SMILES/object).")
    return X


class AutoGluonRegressor(RegressorMixin, BaseEstimator):
    """
    sklearn-compatible wrapper for AutoGluon TabularPredictor (regression).

    - fit(X, y, eval_set=[(X_val, y_val)]) uses tuning_data if provided.
    - predict(X) returns float32 vector.
    - best_model_* helpers expose winning model + hyperparameters + metadata.
    - evaluate_metrics returns rmse/mae/r2 (positive RMSE).
    """

    def __init__(self, cfg: AutoGluonConfig = AutoGluonConfig(), *, random_state: int = 0):
        self.cfg = cfg
        self.random_state = int(random_state)

        self.predictor_: Optional[TabularPredictor] = None
        self.path_: Optional[str] = None
        self.feature_names_: Optional[list[str]] = None

    def __sklearn_is_fitted__(self) -> bool:
        return bool(getattr(self, "is_fitted_", False))

    def _to_df(self, X: np.ndarray) -> pd.DataFrame:
        X = _as_2d_float(X)
        if self.feature_names_ is None:
            cols = [f"f{i}" for i in range(X.shape[1])]
        else:
            if X.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"Feature dimension mismatch: got {X.shape[1]}, expected {len(self.feature_names_)}"
                )
            cols = self.feature_names_
        return pd.DataFrame(X, columns=cols)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> "AutoGluonRegressor":
        X = _as_2d_float(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")

        # Freeze feature naming for consistent DF conversion across predict/eval.
        self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]

        label = "target"
        train_df = self._to_df(X)
        train_df[label] = y

        tuning_df = None
        if eval_set:
            X_val, y_val = eval_set[0]
            X_val = _as_2d_float(X_val)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("X_val rows != y_val length")
            tuning_df = self._to_df(X_val)
            tuning_df[label] = y_val

        self.predictor_ = TabularPredictor(
            label=label,
            problem_type="regression",
            eval_metric=self.cfg.eval_metric,
            path=self.cfg.path,
            verbosity=self.cfg.verbosity,
        )

        # inside fit(), when building fit_kwargs
        fit_kwargs: dict[str, Any] = dict(
            train_data=train_df,
            presets=self.cfg.presets,
            time_limit=self.cfg.time_limit_s,
            tuning_data=tuning_df,
            hyperparameters=self.cfg.hyperparameters,
            num_gpus=self.cfg.num_gpus,
            use_bag_holdout=(self.cfg.use_bag_holdout if tuning_df is not None else None),
            included_model_types=self.cfg.included_model_types,
            dynamic_stacking=self.cfg.dynamic_stacking,
            ds_args=self.cfg.ds_args,
        )

        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

        self.predictor_.fit(**fit_kwargs)
        self.path_ = getattr(self.predictor_, "path", None)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.predictor_ is None or not getattr(self, "is_fitted_", False):
            raise RuntimeError("AutoGluonRegressor.predict called before fit().")
        X = _as_2d_float(X)
        df = self._to_df(X)
        pred = self.predictor_.predict(df)
        return np.asarray(pred, dtype=np.float32).reshape(-1)

    # ----------------------------
    # Introspection helpers
    # ----------------------------

    def best_model_name(self) -> str:
        """
        Returns AutoGluon's current 'best' model used for inference.
        Prefer get_model_best() when available. :contentReference[oaicite:4]{index=4}
        """
        if self.predictor_ is None:
            raise RuntimeError("Call fit() first.")
        get_best = getattr(self.predictor_, "get_model_best", None)
        if callable(get_best):
            return str(get_best())
        # fallback (some versions expose model_best attribute)
        mb = getattr(self.predictor_, "model_best", None)
        if mb is not None:
            return str(mb)

        # last-resort: infer from leaderboard score_val
        lb = self.predictor_.leaderboard(extra_info=False)
        if "score_val" in lb.columns and "model" in lb.columns:
            return str(lb.sort_values("score_val", ascending=False).iloc[0]["model"])
        raise RuntimeError("Unable to determine best model from predictor.")

    def best_model_hyperparameters(self, *, output_format: str = "user") -> dict:
        """
        Returns hyperparameters of the best model. :contentReference[oaicite:5]{index=5}
        output_format: 'user' (non-defaults) or 'all' (full effective set).
        """
        if self.predictor_ is None:
            raise RuntimeError("Call fit() first.")
        m = self.best_model_name()
        return self.predictor_.model_hyperparameters(model=m, output_format=output_format)

    def best_model_info(self) -> dict:
        """
        Returns model metadata dict for the best model. :contentReference[oaicite:6]{index=6}
        """
        if self.predictor_ is None:
            raise RuntimeError("Call fit() first.")
        m = self.best_model_name()
        return self.predictor_.model_info(model=m)

    def leaderboard(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        *,
        extra_info: bool = True,
        score_format: str = "score",
        extra_metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns AutoGluon's leaderboard. If X,y provided, computes test scores too.
        Note: leaderboard scores are "higher is better"; for RMSE they are sign-flipped
        unless you use score_format='error'. :contentReference[oaicite:7]{index=7}
        """
        if self.predictor_ is None:
            raise RuntimeError("Call fit() first.")

        if X is None or y is None:
            return self.predictor_.leaderboard(extra_info=extra_info, score_format=score_format)

        X = _as_2d_float(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        df = self._to_df(X)
        df["target"] = y
        return self.predictor_.leaderboard(
            data=df,
            extra_info=extra_info,
            score_format=score_format,
            extra_metrics=extra_metrics,
        )

    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Returns positive RMSE/MAE and standard R2, computed from predictions.
        (This avoids AutoGluon's "higher-is-better" sign conventions.) :contentReference[oaicite:8]{index=8}
        """
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        y_pred = self.predict(X).astype(np.float32, copy=False)
        return {
            "rmse": float(rmse(y, y_pred)),
            "mae": float(mae(y, y_pred)),
            "r2": float(r2(y, y_pred)),
        }

    def best_model_report(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        One-shot report: best model name, hyperparameters, model info, and rmse/mae/r2.
        """
        best = self.best_model_name()
        return {
            "best_model": best,
            "best_model_hparams_user": self.best_model_hyperparameters(output_format="user"),
            "best_model_hparams_all": self.best_model_hyperparameters(output_format="all"),
            "best_model_info": self.best_model_info(),
            "metrics": self.evaluate_metrics(X, y),
        }
