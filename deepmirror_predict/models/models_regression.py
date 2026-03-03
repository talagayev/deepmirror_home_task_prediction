# src/automl_molops/modeling/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


ModelName = Literal["rf", "svm", "xgb", "lgbm"]


@dataclass(frozen=True)
class TrainConfig:
    model: ModelName
    random_state: int = 0
    n_jobs: int = -1  # used by RF, LGBM & XGB
    # preprocessing
    impute: Optional[str] = None  # None | "median"
    scale: bool = False           # For SVM


def _has_nans(X: np.ndarray) -> bool:
    return bool(np.isnan(X).any())


def default_train_config(model: ModelName, X: np.ndarray) -> TrainConfig:
    need_impute = _has_nans(X)

    if model == "svm":
        return TrainConfig(model=model, impute="median" if need_impute else None, scale=True)

    return TrainConfig(model=model, impute="median" if need_impute else None, scale=False)


def _make_estimator(
    model: ModelName,
    params: Optional[Dict[str, Any]],
    *,
    random_state: int,
    n_jobs: int,
):
    params = dict(params or {})

    if model == "rf":
        defaults = dict(
            n_estimators=500,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_leaf=1,
        )
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    if model == "svm":
        defaults = dict(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
            gamma="scale",
        )
        defaults.update(params)
        return SVR(**defaults)

    if model == "xgb":
        defaults = dict(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
        )
        defaults.update(params)
        return XGBRegressor(**defaults)

    if model == "lgbm":
        defaults = dict(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.0,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        defaults.update(params)
        return LGBMRegressor(**defaults)

    raise ValueError(f"Unknown model: {model}")


def build_pipeline(
    X: np.ndarray,
    *,
    cfg: TrainConfig,
    params: Optional[Dict[str, Any]] = None,
):
    steps = []

    if cfg.impute == "median":
        steps.append(("imputer", SimpleImputer(strategy="median")))

    if cfg.scale:
        steps.append(("scaler", StandardScaler()))

    est = _make_estimator(cfg.model, params, random_state=cfg.random_state, n_jobs=cfg.n_jobs)
    steps.append(("model", est))

    return Pipeline(steps)


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    model: ModelName,
    params: Optional[Dict[str, Any]] = None,
    cfg: Optional[TrainConfig] = None,
) -> Tuple[np.ndarray, Any]:
    X_train = np.asarray(X_train, dtype=np.float32)
    X_valid = np.asarray(X_valid, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    cfg = cfg or default_train_config(model, X_train)
    pipe = build_pipeline(X_train, cfg=cfg, params=params)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_valid).astype(np.float32, copy=False)
    return y_pred, pipe