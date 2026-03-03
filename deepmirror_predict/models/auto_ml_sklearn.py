# src/deepmirror_predict/models/auto_ml_sklearn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from auto_sklearn2 import AutoSklearnRegressor


@dataclass(frozen=True)
class AutoSklearn2Config:
    time_limit: int = 120
    random_state: int = 0
    n_jobs: int = -1
    scoring: str = "neg_mean_squared_error"
    exclude_models: Tuple[str, ...] = ("ransac",)  # optionally add "mlp"


def fit_predict_autosklearn2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    cfg: AutoSklearn2Config = AutoSklearn2Config(),
) -> Tuple[np.ndarray, AutoSklearnRegressor]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    model = AutoSklearnRegressor(
        time_limit=cfg.time_limit,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        scoring=cfg.scoring,
    )

    # filter model pool
    orig_get_models = model._get_models
    def _get_models_filtered():
        m = orig_get_models()
        for k in cfg.exclude_models:
            m.pop(k, None)
        return m
    model._get_models = _get_models_filtered  # type: ignore[method-assign]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return np.asarray(y_pred, dtype=np.float32), model


def autosklearn2_selected(model: AutoSklearnRegressor) -> Dict[str, Any]:
    best_params = getattr(model, "best_params", None)
    best_score = getattr(model, "best_score", None)
    scoring = getattr(model, "scoring", None)

    out: Dict[str, Any] = {
        "best_params": best_params,
        "best_score": best_score,
        "scoring": scoring,
        "best_model_type": type(getattr(model, "best_model", None)).__name__,
    }
    if scoring == "neg_mean_squared_error" and best_score is not None:
        out["best_cv_rmse"] = float(np.sqrt(-float(best_score)))
    return out


def autosklearn2_cv_table(model: AutoSklearnRegressor) -> pd.DataFrame:
    perf = model.get_models_performance()
    df = pd.DataFrame([{"pipeline": k, "cv_score": float(v)} for k, v in perf.items()])
    scoring = getattr(model, "scoring", "r2")

    if scoring == "neg_mean_squared_error":
        df["cv_rmse"] = np.sqrt(-df["cv_score"])
        return df.sort_values("cv_rmse", ascending=True, ignore_index=True)

    return df.sort_values("cv_score", ascending=False, ignore_index=True)