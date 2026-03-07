from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold

from .metrics import mae as _mae
from .metrics import r2 as _r2
from .metrics import rmse as _rmse
from .models_regression import ModelName, TrainConfig, build_pipeline, default_train_config

try:
    import optuna
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Optuna is required for hyperparameter optimization. Install with `pip install optuna`."
    ) from e


MetricName = Literal["rmse", "mae", "r2"]
DirectionName = Literal["minimize", "maximize"]


def _metric_fn(metric: MetricName) -> Callable[[np.ndarray, np.ndarray], float]:
    if metric == "rmse":
        return _rmse
    if metric == "mae":
        return _mae
    if metric == "r2":
        return _r2
    raise ValueError(f"Unknown metric: {metric}")


def _direction(metric: MetricName) -> DirectionName:
    return "maximize" if metric == "r2" else "minimize"


def suggest_params(trial: "optuna.Trial", model: ModelName) -> Dict[str, Any]:
    """
    Model-specific Optuna search spaces.

    Returned params are passed into:
        deepmirror_predict.models.models_regression.build_pipeline(..., params=params)
    so keys must match the underlying estimator constructor args.
    """
    if model == "rf":
        max_depth_choice = trial.suggest_categorical("max_depth_choice", ["none", "int"])
        max_depth = None if max_depth_choice == "none" else trial.suggest_int("max_depth", 2, 40)

        max_features_mode = trial.suggest_categorical("max_features_mode", ["sqrt", "log2", "fraction"])
        if max_features_mode == "fraction":
            max_features = trial.suggest_float("max_features", 0.1, 1.0)
        else:
            max_features = max_features_mode

        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 2000, step=100),
            max_depth=max_depth,
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=max_features,
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )

    if model == "svm":
        gamma_mode = trial.suggest_categorical("gamma_mode", ["scale", "auto", "value"])
        gamma = (
            trial.suggest_float("gamma", 1e-6, 1e-1, log=True)
            if gamma_mode == "value"
            else gamma_mode
        )
        return dict(
            kernel="rbf",
            C=trial.suggest_float("C", 1e-2, 1e3, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
            gamma=gamma,
        )

    if model == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 6000, step=100),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 0.1, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )

    if model == "lgbm":
        max_depth_choice = trial.suggest_categorical("max_depth_choice", ["-1", "int"])
        max_depth = -1 if max_depth_choice == "-1" else trial.suggest_int("max_depth", 3, 20)

        return dict(
            n_estimators=trial.suggest_int("n_estimators", 500, 10000, step=250),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 256),
            max_depth=max_depth,
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )

    raise ValueError(f"Unknown model: {model}")


@dataclass(frozen=True)
class OptunaTuningConfig:
    metric: MetricName = "rmse"
    cv_folds: int = 5
    n_trials: int = 50
    timeout_s: Optional[int] = None
    random_state: int = 0

    # Optuna knobs
    sampler: Optional["optuna.samplers.BaseSampler"] = None
    pruner: Optional["optuna.pruners.BasePruner"] = None
    study_name: Optional[str] = None
    storage: Optional[str] = None
    load_if_exists: bool = False
    show_progress_bar: bool = False


def _objective(
    trial: "optuna.Trial",
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: ModelName,
    cfg: TrainConfig,
    tuning: OptunaTuningConfig,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> float:
    metric_fn = _metric_fn(tuning.metric)

    params: Dict[str, Any] = {}
    if fixed_params:
        params.update(fixed_params)
    params.update(suggest_params(trial, model))

    cv = KFold(n_splits=tuning.cv_folds, shuffle=True, random_state=tuning.random_state)

    scores = []
    for step, (tr_idx, va_idx) in enumerate(cv.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        pipe = build_pipeline(X_tr, cfg=cfg, params=params)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        score = float(metric_fn(y_va, pred))
        scores.append(score)

        # prune on running mean
        trial.report(float(np.mean(scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: ModelName,
    tuning: Optional[OptunaTuningConfig] = None,
    cfg: Optional[TrainConfig] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> "optuna.Study":
    """
    Run Optuna HPO using CV on (X, y). Returns the Optuna Study.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    tuning = tuning or OptunaTuningConfig()
    cfg = cfg or default_train_config(model, X)

    sampler = tuning.sampler or optuna.samplers.TPESampler(seed=tuning.random_state)
    pruner = tuning.pruner or optuna.pruners.MedianPruner(n_warmup_steps=1)

    study = optuna.create_study(
        direction=_direction(tuning.metric),
        sampler=sampler,
        pruner=pruner,
        study_name=tuning.study_name,
        storage=tuning.storage,
        load_if_exists=tuning.load_if_exists,
    )

    study.optimize(
        lambda t: _objective(t, X, y, model=model, cfg=cfg, tuning=tuning, fixed_params=fixed_params),
        n_trials=tuning.n_trials,
        timeout=tuning.timeout_s,
        show_progress_bar=tuning.show_progress_bar,
    )
    return study


def extract_model_params(best_params: Dict[str, Any], *, model: ModelName) -> Dict[str, Any]:
    """
    Removes helper params used for conditional search spaces.
    """
    p = dict(best_params)

    if model in {"rf", "lgbm"}:
        p.pop("max_depth_choice", None)
    if model == "rf":
        p.pop("max_features_mode", None)
    if model == "svm":
        p.pop("gamma_mode", None)

    return p


def fit_best_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: ModelName,
    study: "optuna.Study",
    cfg: Optional[TrainConfig] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
):
    """
    Fit the best pipeline from the Optuna study on all data and return the fitted Pipeline.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    cfg = cfg or default_train_config(model, X)
    best = extract_model_params(study.best_trial.params, model=model)

    params: Dict[str, Any] = {}
    if fixed_params:
        params.update(fixed_params)
    params.update(best)

    pipe = build_pipeline(X, cfg=cfg, params=params)
    pipe.fit(X, y)
    return pipe


def tune_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    model: ModelName,
    tuning: Optional[OptunaTuningConfig] = None,
    cfg: Optional[TrainConfig] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Any, "optuna.Study"]:
    """
    Workflow helper:
      1) CV-based Optuna tuning on (X_train, y_train)
      2) fit best model on full train
      3) predict on X_valid

    Returns: (y_pred, fitted_pipeline, study)
    """
    study = tune_hyperparameters(
        X_train,
        y_train,
        model=model,
        tuning=tuning,
        cfg=cfg,
        fixed_params=fixed_params,
    )
    pipe = fit_best_pipeline(
        X_train,
        y_train,
        model=model,
        study=study,
        cfg=cfg,
        fixed_params=fixed_params,
    )
    X_valid = np.asarray(X_valid, dtype=np.float32)
    y_pred = pipe.predict(X_valid).astype(np.float32, copy=False)
    return y_pred, pipe, study
