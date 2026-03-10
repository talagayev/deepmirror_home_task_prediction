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

# Requires: deepmirror_predict/models/chemprop_regressor.py (ChempropConfig, ChempropRegressor)
from deepmirror_predict.models.chemprop_regression import ChempropConfig, ChempropRegressor
from deepmirror_predict.models.autogluon_regressor import AutoGluonConfig, AutoGluonRegressor

ModelName = Literal["rf", "svm", "xgb", "lgbm", "chemprop", "autogluon"]


@dataclass(frozen=True)
class TrainConfig:
    model: ModelName
    random_state: int = 0
    n_jobs: int = -1  # used by RF, LGBM & XGB (ignored by Chemprop)
    # preprocessing
    impute: Optional[str] = None  # None | "median"
    scale: bool = False           # For SVM


def _has_nans(X: np.ndarray) -> bool:
    """
    NaN detection that is safe for both numeric matrices and object arrays (e.g., SMILES).
    """
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.number):
        return False
    return bool(np.isnan(X).any())


def default_train_config(model: ModelName, X: np.ndarray) -> TrainConfig:
    """
    Default preprocessing choices per model.

    - chemprop: X is SMILES (object), no imputation/scaling in sklearn pipeline
    - svm: scaling recommended, impute if needed
    - others: impute if needed, no scaling by default
    """
    if model == "chemprop":
        return TrainConfig(model=model, impute=None, scale=False)

    need_impute = _has_nans(X)

    if model == "svm":
        return TrainConfig(model=model, impute="median" if need_impute else None, scale=True)

    if model == "autogluon":
        return TrainConfig(model=model, impute=None, scale=False)

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

    if model == "chemprop":
        # Allow overriding random_state via params for consistency with other estimators
        rs = int(params.pop("random_state", random_state))
        params.pop("n_jobs", None)  # not used by ChempropRegressor

        cfg = params.pop("cfg", None)
        if cfg is None:
            cfg = ChempropConfig(**params)
        elif not isinstance(cfg, ChempropConfig):
            raise TypeError(f"chemprop param 'cfg' must be ChempropConfig, got {type(cfg)}")

        return ChempropRegressor(cfg=cfg, random_state=rs)

    if model == "autogluon":
        rs = int(params.pop("random_state", random_state))
        params.pop("n_jobs", None)  # AutoGluon doesn't use sklearn n_jobs here

        cfg = params.pop("cfg", None)
        if cfg is None:
            cfg = AutoGluonConfig(**params)
        elif not isinstance(cfg, AutoGluonConfig):
            raise TypeError(f"autogluon param 'cfg' must be AutoGluonConfig, got {type(cfg)}")

        return AutoGluonRegressor(cfg=cfg, random_state=rs)

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
    y_valid: Optional[np.ndarray] = None,  # required for chemprop / optional for autogluon
) -> Tuple[np.ndarray, Any]:
    """
    Fit model on (X_train, y_train) and predict on X_valid.

    - For numeric models: X_* are float32 feature matrices.
    - For chemprop: X_* are SMILES arrays/lists (object dtype).
    - For autogluon: X_* are numeric feature matrices; eval_set is optional.

    Returns: (y_pred, fitted_pipeline)
    """
    if model == "chemprop":
        X_train = np.asarray(X_train, dtype=object)
        X_valid = np.asarray(X_valid, dtype=object)
    else:
        X_train = np.asarray(X_train, dtype=np.float32)
        X_valid = np.asarray(X_valid, dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.float32)

    cfg = cfg or default_train_config(model, X_train)
    pipe = build_pipeline(X_train, cfg=cfg, params=params)

    if model == "chemprop":
        if y_valid is None:
            raise ValueError("fit_predict(model='chemprop') requires y_valid for eval_set.")
        y_valid = np.asarray(y_valid, dtype=np.float32)
        pipe.fit(X_train, y_train, model__eval_set=[(X_valid, y_valid)])

    elif model == "autogluon":
        if y_valid is not None:
            y_valid = np.asarray(y_valid, dtype=np.float32)
            pipe.fit(X_train, y_train, model__eval_set=[(X_valid, y_valid)])
        else:
            pipe.fit(X_train, y_train)

    else:
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_valid).astype(np.float32, copy=False)
    return y_pred, pipe
