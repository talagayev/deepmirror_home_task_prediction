from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence
import json
import math
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from deepmirror_predict.analysis.applicability_domain import parse_feature_set, parse_feature_token
from deepmirror_predict.features.avalon import AvalonFPConfig, avalon_bits_feature_names, avalon_bits_from_smiles
from deepmirror_predict.features.chemeleon import CheMeleonConfig, chemeleon_batch_from_smiles
from deepmirror_predict.features.mordred import mordred2d_batch_from_smiles, prune_mordred_matrix
from deepmirror_predict.features.morgan import MorganFPConfig, morgan_bits_feature_names, morgan_bits_from_smiles
from deepmirror_predict.features.rdkit_path import (
    RDKitPathFPConfig,
    rdkit_path_bits_feature_names,
    rdkit_path_bits_from_smiles,
)
from deepmirror_predict.models.chemprop_regression import ChempropConfig
from deepmirror_predict.models.metrics import (
    kendall_tau,
    kendall_tau_pvalue,
    mae,
    mse,
    r2,
    rmse,
    spearman_pvalue,
    spearman_rho,
)
from deepmirror_predict.models.models_regression import ModelName, TrainConfig, build_pipeline, default_train_config

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None


MetricName = Literal[
    "rmse",
    "mse",
    "mae",
    "r2",
    "kendall_tau",
    "kendall_tau_pvalue",
    "spearman_rho",
    "spearman_pvalue",
]
SplitMethod = Literal["random", "scaffold", "group", "time_series"]


@dataclass(frozen=True)
class SplitConfig:
    method: SplitMethod = "random"
    outer_folds: int = 5
    inner_folds: int = 5
    shuffle: bool = True
    random_state: int = 0
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    time_ascending: bool = True
    scaffold_include_chirality: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    feature_set: str = "morgan_r3_b1024"
    mordred_max_nan_frac: float = 0.2
    mordred_drop_constant: bool = True


@dataclass(frozen=True)
class OptimizationConfig:
    enabled: bool = False
    metric: str = "rmse"
    n_trials: int = 30
    timeout_s: Optional[int] = None
    random_state: int = 0


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    output_dir: str
    smiles_column: str
    target_column: str
    metrics: tuple[MetricName, ...] = ("rmse", "mae", "r2")
    primary_metric: str = "rmse"
    models: tuple[str, ...] = ("rf",)
    feature_sets: tuple[str, ...] = ("morgan_r3_b1024",)
    split: SplitConfig = SplitConfig()
    feature_params: FeatureConfig = FeatureConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    dropna_target: bool = True
    dropna_smiles: bool = True
    row_id_column: Optional[str] = None
    refit_best_model: bool = True
    refit_validation_fraction: float = 0.1
    save_fold_predictions: bool = True
    save_best_params: bool = True
    random_state: int = 0
    n_jobs: int = -1
    model_params: Optional[dict[str, dict[str, Any]]] = None


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _metric_fn(name: MetricName):
    mapping = {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "kendall_tau": kendall_tau,
        "kendall_tau_pvalue": kendall_tau_pvalue,
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
    }
    if name not in mapping:
        raise ValueError(f"Unknown metric: {name}")
    return mapping[name]


def _is_higher_better(metric: str) -> bool:
    return metric in {"r2", "kendall_tau", "spearman_rho"}


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray, metrics: Iterable[MetricName]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for metric in metrics:
        scores[metric] = float(_metric_fn(metric)(y_true, y_pred))
    return scores


def _smiles_to_scaffold(smiles: str, include_chirality: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def _greedy_group_assignment(groups: dict[str, list[int]], n_splits: int) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    bucket_sizes = [0] * n_splits
    for idxs in sorted(groups.values(), key=len, reverse=True):
        bucket_id = int(np.argmin(bucket_sizes))
        buckets[bucket_id].extend(idxs)
        bucket_sizes[bucket_id] += len(idxs)
    return buckets


def _make_scaffold_splits(
    df: pd.DataFrame,
    smiles_column: str,
    n_splits: int,
    include_chirality: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    scaffolds: dict[str, list[int]] = {}
    smiles = df[smiles_column].astype(str).tolist()
    for i, smi in enumerate(smiles):
        scaffolds.setdefault(_smiles_to_scaffold(smi, include_chirality=include_chirality), []).append(i)

    buckets = _greedy_group_assignment(scaffolds, n_splits=n_splits)
    all_idx = np.arange(len(df), dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for va in buckets:
        va_idx = np.array(sorted(va), dtype=int)
        tr_mask = np.ones(len(df), dtype=bool)
        tr_mask[va_idx] = False
        tr_idx = all_idx[tr_mask]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            continue
        splits.append((tr_idx, va_idx))
    if len(splits) < 2:
        raise ValueError("Scaffold split produced fewer than 2 non-empty folds. Reduce n_splits.")
    return splits


def make_splits(
    df: pd.DataFrame,
    smiles_column: str,
    split_cfg: SplitConfig,
    *,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    n_samples = len(df)
    if n_splits > n_samples:
        raise ValueError(f"n_splits={n_splits} cannot exceed number of rows ({n_samples})")

    idx = np.arange(n_samples, dtype=int)

    if split_cfg.method == "random":
        cv = KFold(
            n_splits=n_splits,
            shuffle=split_cfg.shuffle,
            random_state=split_cfg.random_state if split_cfg.shuffle else None,
        )
        return [(tr.astype(int), va.astype(int)) for tr, va in cv.split(idx)]

    if split_cfg.method == "group":
        if not split_cfg.group_column:
            raise ValueError("group_column must be provided for group split")
        if split_cfg.group_column not in df.columns:
            raise ValueError(f"group_column '{split_cfg.group_column}' not found in dataframe")
        groups = df[split_cfg.group_column].astype(str).to_numpy()
        unique_groups = pd.unique(groups)
        if n_splits > len(unique_groups):
            raise ValueError(
                f"group split requires at least {n_splits} unique groups, found {len(unique_groups)}"
            )
        cv = GroupKFold(n_splits=n_splits)
        return [(tr.astype(int), va.astype(int)) for tr, va in cv.split(idx, groups=groups)]

    if split_cfg.method == "scaffold":
        return _make_scaffold_splits(
            df,
            smiles_column=smiles_column,
            n_splits=n_splits,
            include_chirality=split_cfg.scaffold_include_chirality,
        )

    if split_cfg.method == "time_series":
        if not split_cfg.time_column:
            raise ValueError("time_column must be provided for time_series split")
        if split_cfg.time_column not in df.columns:
            raise ValueError(f"time_column '{split_cfg.time_column}' not found in dataframe")
        cv = TimeSeriesSplit(n_splits=n_splits)
        return [(tr.astype(int), va.astype(int)) for tr, va in cv.split(idx)]

    raise ValueError(f"Unknown split method: {split_cfg.method}")


def _single_validation_split(
    df_train: pd.DataFrame,
    smiles_column: str,
    split_cfg: SplitConfig,
    valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = len(df_train)
    if n_train < 4:
        raise ValueError("Need at least 4 rows to create an internal validation split")

    if split_cfg.method == "time_series":
        n_valid = max(1, int(math.ceil(n_train * valid_fraction)))
        n_valid = min(n_valid, n_train - 1)
        tr_idx = np.arange(0, n_train - n_valid, dtype=int)
        va_idx = np.arange(n_train - n_valid, n_train, dtype=int)
        return tr_idx, va_idx

    if split_cfg.method == "random":
        n_valid = max(1, int(math.ceil(n_train * valid_fraction)))
        n_valid = min(n_valid, n_train - 1)
        rng = np.random.default_rng(split_cfg.random_state)
        perm = rng.permutation(n_train)
        va_idx = np.sort(perm[:n_valid]).astype(int)
        tr_mask = np.ones(n_train, dtype=bool)
        tr_mask[va_idx] = False
        tr_idx = np.arange(n_train, dtype=int)[tr_mask]
        return tr_idx, va_idx

    n_splits = min(5, n_train)
    splits = make_splits(df_train, smiles_column, split_cfg, n_splits=n_splits)
    if not splits:
        raise ValueError("Unable to create validation split from training data")
    return splits[0]


def _stack_features(rows: list[np.ndarray], name: str) -> np.ndarray:
    if not rows:
        raise ValueError(f"No features built for {name}")
    return np.vstack(rows).astype(np.float32, copy=False)


def _split_feature_tokens(feature_set: str) -> list[str]:
    return [tok.strip() for tok in feature_set.split("+") if tok.strip()]


def _parse_chemprop_feature_mode(feature_set: str) -> tuple[str, list[str]]:
    tokens = _split_feature_tokens(feature_set)
    if not tokens:
        raise ValueError("Empty feature_set")

    backbone_tokens = [t for t in tokens if t in {"smiles", "chemeleon"}]
    if len(backbone_tokens) != 1:
        raise ValueError(
            "Chemprop feature_set must contain exactly one backbone token "
            "('smiles' or 'chemeleon')"
        )

    backbone = backbone_tokens[0]
    descriptor_tokens = [t for t in tokens if t not in {"smiles", "chemeleon"}]

    for token in descriptor_tokens:
        spec = parse_feature_token(token)
        if spec.kind == "chemeleon":
            raise ValueError(
                "CheMeleon may only be used as the Chemprop backbone token, not as an extra descriptor"
            )

    return backbone, descriptor_tokens


def _is_valid_chemprop_feature_set(feature_set: str) -> bool:
    try:
        _parse_chemprop_feature_mode(feature_set)
        return True
    except Exception:
        return False


def featurize_smiles(smiles_list: Sequence[str], feature_cfg: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    specs = parse_feature_set(feature_cfg.feature_set)

    blocks: list[np.ndarray] = []
    names: list[str] = []

    for spec in specs:
        if spec.kind == "morgan":
            cfg = MorganFPConfig(
                radius=spec.params["radius"],
                n_bits=spec.params["n_bits"],
            )
            rows = [morgan_bits_from_smiles(s, cfg) for s in smiles_list]
            if any(v is None for v in rows):
                bad = [s for s, v in zip(smiles_list, rows) if v is None][:5]
                raise ValueError(f"Invalid SMILES found during Morgan featurization: {bad}")
            Xi = _stack_features(rows, spec.raw_name)
            ni = [f"{spec.raw_name}::{x}" for x in morgan_bits_feature_names(cfg)]

        elif spec.kind == "avalon":
            cfg = AvalonFPConfig(n_bits=spec.params["n_bits"])
            rows = [avalon_bits_from_smiles(s, cfg) for s in smiles_list]
            if any(v is None for v in rows):
                bad = [s for s, v in zip(smiles_list, rows) if v is None][:5]
                raise ValueError(f"Invalid SMILES found during Avalon featurization: {bad}")
            Xi = _stack_features(rows, spec.raw_name)
            ni = [f"{spec.raw_name}::{x}" for x in avalon_bits_feature_names(cfg)]

        elif spec.kind == "rdkit_path":
            cfg = RDKitPathFPConfig(
                min_path=spec.params["min_path"],
                max_path=spec.params["max_path"],
                n_bits=spec.params["n_bits"],
            )
            rows = [rdkit_path_bits_from_smiles(s, cfg) for s in smiles_list]
            if any(v is None for v in rows):
                bad = [s for s, v in zip(smiles_list, rows) if v is None][:5]
                raise ValueError(f"Invalid SMILES found during RDKit-path featurization: {bad}")
            Xi = _stack_features(rows, spec.raw_name)
            ni = [f"{spec.raw_name}::{x}" for x in rdkit_path_bits_feature_names(cfg)]

        elif spec.kind == "mordred":
            Xi, raw_names = mordred2d_batch_from_smiles(smiles_list)
            Xi, raw_names = prune_mordred_matrix(
                Xi,
                raw_names,
                max_nan_frac=feature_cfg.mordred_max_nan_frac,
                drop_constant=feature_cfg.mordred_drop_constant,
            )
            Xi = Xi.astype(np.float32, copy=False)
            ni = [f"{spec.raw_name}::{x}" for x in raw_names]

        elif spec.kind == "chemeleon":
            Xi, raw_names = chemeleon_batch_from_smiles(smiles_list, cfg=CheMeleonConfig())
            Xi = Xi.astype(np.float32, copy=False)
            ni = [f"{spec.raw_name}::{x}" for x in raw_names]

        else:
            raise ValueError(f"Unsupported feature kind in CV featurization: {spec.kind}")

        if Xi.dtype == object:
            raise ValueError(f"Feature block '{spec.raw_name}' produced object dtype unexpectedly")
        blocks.append(Xi)
        names.extend(ni)

    if not blocks:
        raise ValueError(f"No feature blocks produced for feature_set '{feature_cfg.feature_set}'")

    if len(blocks) == 1:
        return blocks[0].astype(np.float32, copy=False), names

    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), names


def _make_train_cfg(model: ModelName, X_train: np.ndarray, random_state: int, n_jobs: int) -> TrainConfig:
    base = default_train_config(model, X_train)
    return TrainConfig(
        model=base.model,
        random_state=random_state,
        n_jobs=n_jobs,
        impute=base.impute,
        scale=base.scale,
    )


def _fit_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model: ModelName,
    params: Optional[dict[str, Any]],
    cfg: TrainConfig,
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
):
    pipe = build_pipeline(X_train, cfg=cfg, params=params)
    if model in {"chemprop", "autogluon"} and X_eval is not None and y_eval is not None:
        pipe.fit(X_train, y_train, model__eval_set=[(X_eval, y_eval)])
    else:
        pipe.fit(X_train, y_train)
    return pipe


def _predict(pipe, X_valid: np.ndarray) -> np.ndarray:
    return np.asarray(pipe.predict(X_valid), dtype=np.float32).reshape(-1)


def _supports_optuna(model: str) -> bool:
    return model in {"rf", "svm", "xgb", "lgbm", "chemprop"}


def _augment_fixed_params_for_combo(
    model_name: str,
    feature_set: str,
    fixed_params: dict[str, Any],
) -> dict[str, Any]:
    params = dict(fixed_params)

    if model_name != "chemprop":
        return params

    backbone, descriptor_tokens = _parse_chemprop_feature_mode(feature_set)

    if backbone == "chemeleon":
        params["from_foundation"] = "chemeleon"

    existing = list(params.get("extra_descriptor_tokens", ()))
    for token in descriptor_tokens:
        if token not in existing:
            existing.append(token)
    if existing:
        params["extra_descriptor_tokens"] = tuple(existing)

    return params


def _base_chemprop_cfg_dict(params: dict[str, Any]) -> dict[str, Any]:
    if "cfg" in params:
        cfg = params["cfg"]
        if isinstance(cfg, ChempropConfig):
            base = asdict(cfg)
        elif isinstance(cfg, dict):
            base = dict(cfg)
        else:
            raise TypeError(f"Unsupported chemprop cfg type: {type(cfg)}")
    else:
        allowed = set(ChempropConfig.__dataclass_fields__.keys())
        base = {k: v for k, v in params.items() if k in allowed}
    return base


def _suggest_params(
    trial: "optuna.Trial",
    model: str,
    base_params: dict[str, Any],
    checkpoint_root: Optional[Path],
) -> dict[str, Any]:
    params: dict[str, Any] = dict(base_params)

    if model == "rf":
        max_depth_choice = trial.suggest_categorical("max_depth_choice", ["none", "int"])
        max_depth = None if max_depth_choice == "none" else trial.suggest_int("max_depth", 2, 40)
        max_features_mode = trial.suggest_categorical("max_features_mode", ["sqrt", "log2", "fraction"])
        if max_features_mode == "fraction":
            max_features = trial.suggest_float("max_features", 0.1, 1.0)
        else:
            max_features = max_features_mode
        params.update(
            dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 2000, step=100),
                max_depth=max_depth,
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                max_features=max_features,
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            )
        )
        return params

    if model == "svm":
        gamma_mode = trial.suggest_categorical("gamma_mode", ["scale", "auto", "value"])
        gamma = trial.suggest_float("gamma", 1e-6, 1e-1, log=True) if gamma_mode == "value" else gamma_mode
        params.update(
            dict(
                kernel="rbf",
                C=trial.suggest_float("C", 1e-2, 1e3, log=True),
                epsilon=trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
                gamma=gamma,
            )
        )
        return params

    if model == "xgb":
        params.update(
            dict(
                n_estimators=trial.suggest_int("n_estimators", 300, 6000, step=100),
                learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 2, 12),
                min_child_weight=trial.suggest_float("min_child_weight", 0.1, 20.0, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            )
        )
        return params

    if model == "lgbm":
        max_depth_choice = trial.suggest_categorical("max_depth_choice", ["-1", "int"])
        max_depth = -1 if max_depth_choice == "-1" else trial.suggest_int("max_depth", 3, 20)
        params.update(
            dict(
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
        )
        return params

    if model == "chemprop":
        cfg_dict = _base_chemprop_cfg_dict(base_params)

        cfg_dict.setdefault("max_epochs", 30)
        cfg_dict.setdefault("early_stopping", True)
        cfg_dict.setdefault("early_stopping_patience", 10)
        cfg_dict.setdefault("enable_progress_bar", False)
        cfg_dict.setdefault("accelerator", "auto")
        cfg_dict.setdefault("devices", 1)

        from_foundation = str(cfg_dict.get("from_foundation") or "").lower()
        is_chemeleon_foundation = from_foundation == "chemeleon"

        if checkpoint_root is None:
            checkpoint_root = Path(tempfile.mkdtemp(prefix="chemprop_optuna_"))
        ckpt_dir = checkpoint_root / f"trial_{trial.number}"

        tuned_cfg = dict(
            ffn_hidden_dim=trial.suggest_categorical("ffn_hidden_dim", [128, 256, 300, 384, 512]),
            ffn_layers=trial.suggest_int("ffn_layers", 1, 3),
            ffn_dropout=trial.suggest_float("ffn_dropout", 0.0, 0.4),
            batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
            warmup_epochs=trial.suggest_int("warmup_epochs", 1, 5),
            init_lr=trial.suggest_float("init_lr", 1e-5, 5e-4, log=True),
            max_lr=trial.suggest_float("max_lr", 5e-4, 5e-3, log=True),
            final_lr=trial.suggest_float("final_lr", 1e-5, 5e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        )

        if not is_chemeleon_foundation:
            tuned_cfg.update(
                dict(
                    message_hidden_dim=trial.suggest_categorical("message_hidden_dim", [128, 256, 300, 384, 512]),
                    message_depth=trial.suggest_int("message_depth", 2, 6),
                    message_dropout=trial.suggest_float("message_dropout", 0.0, 0.3),
                )
            )

        cfg_dict.update(tuned_cfg)
        cfg_dict["checkpoint_dir"] = str(ckpt_dir)

        return {"cfg": ChempropConfig(**cfg_dict)}

    raise ValueError(f"Optuna search space not implemented for model={model}")


def _clean_params_for_export(params: dict[str, Any], model: Optional[str] = None) -> dict[str, Any]:
    p = dict(params)

    if model in {None, "rf", "lgbm"}:
        p.pop("max_depth_choice", None)
    if model in {None, "rf"}:
        p.pop("max_features_mode", None)
    if model in {None, "svm"}:
        p.pop("gamma_mode", None)

    if model == "chemprop":
        cfg_dict = _base_chemprop_cfg_dict(p)
        return {"cfg": cfg_dict}

    out: dict[str, Any] = {}
    for k, v in p.items():
        if isinstance(v, ChempropConfig):
            out[k] = asdict(v)
        else:
            out[k] = v
    return out


def _best_params_from_study(best_params: dict[str, Any], model: str, fixed_params: dict[str, Any]) -> dict[str, Any]:
    if model == "chemprop":
        cfg_dict = _base_chemprop_cfg_dict(fixed_params)
        cfg_dict.update(best_params)
        return {"cfg": cfg_dict}
    return _clean_params_for_export(best_params, model=model)


def _optuna_objective(
    trial: "optuna.Trial",
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    df_train: pd.DataFrame,
    smiles_column: str,
    model: ModelName,
    split_cfg: SplitConfig,
    inner_folds: int,
    metric: MetricName,
    random_state: int,
    n_jobs: int,
    fixed_params: dict[str, Any],
    checkpoint_root: Optional[Path],
) -> float:
    params = _suggest_params(trial, model, fixed_params, checkpoint_root)
    cfg = _make_train_cfg(model, X_train, random_state=random_state, n_jobs=n_jobs)
    inner_splits = make_splits(df_train, smiles_column, split_cfg, n_splits=inner_folds)
    metric_fn = _metric_fn(metric)

    scores: list[float] = []
    for step_idx, (tr_idx, va_idx) in enumerate(inner_splits):
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train[va_idx]
        y_va = y_train[va_idx]

        if model in {"chemprop", "autogluon"}:
            pipe = _fit_pipeline(
                X_tr,
                y_tr,
                model=model,
                params=params,
                cfg=cfg,
                X_eval=X_va,
                y_eval=y_va,
            )
        else:
            pipe = _fit_pipeline(X_tr, y_tr, model=model, params=params, cfg=cfg)

        y_pred = _predict(pipe, X_va)
        score = float(metric_fn(y_va, y_pred))
        scores.append(score)

        running = float(np.mean(scores))
        trial.report(running, step=step_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def tune_hyperparameters_nested(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    df_train: pd.DataFrame,
    smiles_column: str,
    model: ModelName,
    split_cfg: SplitConfig,
    inner_folds: int,
    optimization_cfg: OptimizationConfig,
    random_state: int,
    n_jobs: int,
    fixed_params: Optional[dict[str, Any]] = None,
    checkpoint_root: Optional[Path] = None,
):
    if optuna is None:
        raise ImportError("Optuna is required for optimization. Install it in your environment.")

    fixed_params = dict(fixed_params or {})
    direction = "maximize" if _is_higher_better(optimization_cfg.metric) else "minimize"
    sampler = optuna.samplers.TPESampler(seed=optimization_cfg.random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: _optuna_objective(
            trial,
            X_train=X_train,
            y_train=y_train,
            df_train=df_train,
            smiles_column=smiles_column,
            model=model,
            split_cfg=split_cfg,
            inner_folds=inner_folds,
            metric=optimization_cfg.metric,  # type: ignore[arg-type]
            random_state=random_state,
            n_jobs=n_jobs,
            fixed_params=fixed_params,
            checkpoint_root=checkpoint_root,
        ),
        n_trials=optimization_cfg.n_trials,
        timeout=optimization_cfg.timeout_s,
        show_progress_bar=False,
    )
    return study, _best_params_from_study(study.best_trial.params, model, fixed_params)


def _materialize_params_for_training(
    best_params: dict[str, Any],
    fixed_params: dict[str, Any],
    model: str,
    checkpoint_dir: Optional[Path] = None,
) -> dict[str, Any]:
    if model == "chemprop":
        merged = dict(fixed_params)
        merged.update(best_params)
        cfg_dict = _base_chemprop_cfg_dict(merged)
        if checkpoint_dir is not None:
            cfg_dict["checkpoint_dir"] = str(checkpoint_dir)
        return {"cfg": ChempropConfig(**cfg_dict)}

    params = dict(fixed_params)
    params.update(best_params)

    if model in {"rf", "lgbm"}:
        params.pop("max_depth_choice", None)
    if model == "rf":
        params.pop("max_features_mode", None)
    if model == "svm":
        params.pop("gamma_mode", None)

    return params


def run_nested_cross_validation(run_cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _ensure_dir(run_cfg.output_dir)
    df = pd.read_csv(run_cfg.input_path)

    if run_cfg.smiles_column not in df.columns:
        raise ValueError(f"smiles_column '{run_cfg.smiles_column}' not found in input data")
    if run_cfg.target_column not in df.columns:
        raise ValueError(f"target_column '{run_cfg.target_column}' not found in input data")
    if run_cfg.row_id_column and run_cfg.row_id_column not in df.columns:
        raise ValueError(f"row_id_column '{run_cfg.row_id_column}' not found in input data")

    if run_cfg.dropna_smiles:
        df = df[df[run_cfg.smiles_column].notna()].copy()
    if run_cfg.dropna_target:
        df = df[df[run_cfg.target_column].notna()].copy()

    if run_cfg.split.method == "time_series":
        if not run_cfg.split.time_column:
            raise ValueError("time_column must be set for time_series split")
        if run_cfg.split.time_column not in df.columns:
            raise ValueError(f"time_column '{run_cfg.split.time_column}' not found in input data")
        df[run_cfg.split.time_column] = pd.to_datetime(df[run_cfg.split.time_column])
        df = df.sort_values(run_cfg.split.time_column, ascending=run_cfg.split.time_ascending).copy()

    if len(df) < max(run_cfg.split.outer_folds, 2):
        raise ValueError("Not enough rows for outer CV")

    all_fold_rows: list[dict[str, Any]] = []
    all_pred_tables: list[pd.DataFrame] = []
    all_best_params: dict[str, Any] = {}
    numeric_feature_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    smiles_all = df[run_cfg.smiles_column].astype(str).tolist()

    for feature_set in run_cfg.feature_sets:
        feature_tokens = _split_feature_tokens(feature_set)

        for model_name in run_cfg.models:
            if model_name == "chemprop":
                if not _is_valid_chemprop_feature_set(feature_set):
                    print(
                        f"[WARN] Skipping incompatible combo model={model_name}, feature_set={feature_set}; "
                        "Chemprop requires exactly one backbone token ('smiles' or 'chemeleon') "
                        "plus zero or more descriptor tokens."
                    )
                    continue
                X_all = np.asarray(smiles_all, dtype=object)
                feature_names = ["smiles"]
            else:
                if "smiles" in feature_tokens:
                    print(
                        f"[WARN] Skipping incompatible combo model={model_name}, feature_set={feature_set}; "
                        "'smiles' backbone is reserved for Chemprop."
                    )
                    continue
                if feature_set not in numeric_feature_cache:
                    X_num, names_num = featurize_smiles(
                        smiles_all,
                        replace(run_cfg.feature_params, feature_set=feature_set),
                    )
                    numeric_feature_cache[feature_set] = (X_num, names_num)
                X_all, feature_names = numeric_feature_cache[feature_set]

            combo_key = f"{model_name}__{feature_set}"
            fixed_params = _augment_fixed_params_for_combo(
                model_name,
                feature_set,
                dict((run_cfg.model_params or {}).get(model_name, {})),
            )
            combo_output = Path(run_cfg.output_dir) / combo_key
            _ensure_dir(combo_output)

            outer_splits = make_splits(df, run_cfg.smiles_column, run_cfg.split, n_splits=run_cfg.split.outer_folds)
            combo_best_params: list[dict[str, Any]] = []

            for outer_fold, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
                df_outer_train = df.iloc[tr_idx].copy()
                df_outer_test = df.iloc[te_idx].copy()

                X_outer_train = X_all[tr_idx]
                y_outer_train = df_outer_train[run_cfg.target_column].to_numpy(dtype=np.float32)
                X_outer_test = X_all[te_idx]
                y_outer_test = df_outer_test[run_cfg.target_column].to_numpy(dtype=np.float32)

                best_params_clean: dict[str, Any] = {}
                checkpoint_root = combo_output / f"outer_fold_{outer_fold}" / "optuna_checkpoints"

                if run_cfg.optimization.enabled:
                    if not _supports_optuna(model_name):
                        print(f"[WARN] Optimization not implemented for model={model_name}; using fixed/default params.")
                    else:
                        study, best_params_clean = tune_hyperparameters_nested(
                            X_train=X_outer_train,
                            y_train=y_outer_train,
                            df_train=df_outer_train,
                            smiles_column=run_cfg.smiles_column,
                            model=model_name,  # type: ignore[arg-type]
                            split_cfg=run_cfg.split,
                            inner_folds=run_cfg.split.inner_folds,
                            optimization_cfg=run_cfg.optimization,
                            random_state=run_cfg.random_state,
                            n_jobs=run_cfg.n_jobs,
                            fixed_params=fixed_params,
                            checkpoint_root=checkpoint_root,
                        )
                        _ = study

                combo_best_params.append(_clean_params_for_export(best_params_clean, model=model_name))

                fold_ckpt_dir = combo_output / f"outer_fold_{outer_fold}" / "final_model_ckpt"
                train_params = _materialize_params_for_training(
                    best_params_clean,
                    fixed_params,
                    model=model_name,
                    checkpoint_dir=fold_ckpt_dir if model_name == "chemprop" else None,
                )

                cfg = _make_train_cfg(model_name, X_outer_train, random_state=run_cfg.random_state, n_jobs=run_cfg.n_jobs)

                if model_name in {"chemprop", "autogluon"}:
                    fit_idx, val_idx = _single_validation_split(
                        df_outer_train,
                        run_cfg.smiles_column,
                        run_cfg.split,
                        valid_fraction=run_cfg.refit_validation_fraction,
                    )
                    X_fit = X_outer_train[fit_idx]
                    y_fit = y_outer_train[fit_idx]
                    X_es = X_outer_train[val_idx]
                    y_es = y_outer_train[val_idx]
                    pipe = _fit_pipeline(
                        X_fit,
                        y_fit,
                        model=model_name,  # type: ignore[arg-type]
                        params=train_params,
                        cfg=cfg,
                        X_eval=X_es,
                        y_eval=y_es,
                    )
                else:
                    pipe = _fit_pipeline(
                        X_outer_train,
                        y_outer_train,
                        model=model_name,  # type: ignore[arg-type]
                        params=train_params,
                        cfg=cfg,
                    )

                y_pred = _predict(pipe, X_outer_test)

                fold_row = {
                    "model": model_name,
                    "feature_set": feature_set,
                    "outer_fold": outer_fold,
                    "n_train": int(len(tr_idx)),
                    "n_test": int(len(te_idx)),
                }
                fold_row.update(_score_predictions(y_outer_test, y_pred, run_cfg.metrics))
                all_fold_rows.append(fold_row)

                if run_cfg.save_fold_predictions:
                    pred_df = pd.DataFrame(
                        {
                            "row_index": df_outer_test.index.to_numpy(),
                            "row_id": (
                                df_outer_test[run_cfg.row_id_column].to_numpy()
                                if run_cfg.row_id_column
                                else df_outer_test.index.to_numpy()
                            ),
                            "model": model_name,
                            "feature_set": feature_set,
                            "outer_fold": outer_fold,
                            "y_true": y_outer_test,
                            "y_pred": y_pred,
                        }
                    )
                    all_pred_tables.append(pred_df)

            all_best_params[combo_key] = combo_best_params

            if run_cfg.save_best_params:
                with open(combo_output / "best_params_per_outer_fold.json", "w", encoding="utf-8") as f:
                    json.dump(combo_best_params, f, indent=2)
                with open(combo_output / "feature_names.json", "w", encoding="utf-8") as f:
                    json.dump(feature_names, f, indent=2)

    df_folds = pd.DataFrame(all_fold_rows)

    summary_records: list[dict[str, Any]] = []
    for (model_name, feature_set), g in df_folds.groupby(["model", "feature_set"], sort=False):
        row = {"model": model_name, "feature_set": feature_set}
        for metric in run_cfg.metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
        summary_records.append(row)
    df_summary = pd.DataFrame(summary_records)

    df_folds.to_csv(Path(run_cfg.output_dir) / "nested_cv_fold_results.csv", index=False)
    df_summary.to_csv(Path(run_cfg.output_dir) / "nested_cv_summary.csv", index=False)

    if all_pred_tables:
        pd.concat(all_pred_tables, axis=0, ignore_index=True).to_csv(
            Path(run_cfg.output_dir) / "nested_cv_predictions.csv",
            index=False,
        )

    if len(df_summary) == 0:
        raise RuntimeError("No CV results were produced")

    df_ranked = df_summary.sort_values(
        f"{run_cfg.primary_metric}_mean",
        ascending=not _is_higher_better(run_cfg.primary_metric),
    ).reset_index(drop=True)
    best_row = df_ranked.iloc[0].to_dict()

    if run_cfg.refit_best_model:
        best_model = str(best_row["model"])
        best_feature_set = str(best_row["feature_set"])
        refit_output = Path(run_cfg.output_dir) / "refit"
        _ensure_dir(refit_output)

        if best_model == "chemprop":
            X_refit = np.asarray(smiles_all, dtype=object)
            feature_names = ["smiles"]
        else:
            X_refit, feature_names = featurize_smiles(
                smiles_all,
                replace(run_cfg.feature_params, feature_set=best_feature_set),
            )

        y_full = df[run_cfg.target_column].to_numpy(dtype=np.float32)
        df_full = df.copy()

        fixed_params = _augment_fixed_params_for_combo(
            best_model,
            best_feature_set,
            dict((run_cfg.model_params or {}).get(best_model, {})),
        )

        final_best_clean: dict[str, Any] = {}
        if run_cfg.optimization.enabled and _supports_optuna(best_model):
            _, final_best_clean = tune_hyperparameters_nested(
                X_train=X_refit,
                y_train=y_full,
                df_train=df_full,
                smiles_column=run_cfg.smiles_column,
                model=best_model,  # type: ignore[arg-type]
                split_cfg=run_cfg.split,
                inner_folds=run_cfg.split.inner_folds,
                optimization_cfg=run_cfg.optimization,
                random_state=run_cfg.random_state,
                n_jobs=run_cfg.n_jobs,
                fixed_params=fixed_params,
                checkpoint_root=refit_output / "optuna_checkpoints",
            )

        final_params = _materialize_params_for_training(
            final_best_clean,
            fixed_params,
            model=best_model,
            checkpoint_dir=refit_output / "chemprop_checkpoint" if best_model == "chemprop" else None,
        )

        cfg = _make_train_cfg(best_model, X_refit, random_state=run_cfg.random_state, n_jobs=run_cfg.n_jobs)

        if best_model in {"chemprop", "autogluon"}:
            fit_idx, val_idx = _single_validation_split(
                df_full,
                run_cfg.smiles_column,
                run_cfg.split,
                valid_fraction=run_cfg.refit_validation_fraction,
            )
            pipe = _fit_pipeline(
                X_refit[fit_idx],
                y_full[fit_idx],
                model=best_model,  # type: ignore[arg-type]
                params=final_params,
                cfg=cfg,
                X_eval=X_refit[val_idx],
                y_eval=y_full[val_idx],
            )
        else:
            pipe = _fit_pipeline(
                X_refit,
                y_full,
                model=best_model,  # type: ignore[arg-type]
                params=final_params,
                cfg=cfg,
            )

        if joblib is not None:
            try:
                joblib.dump(pipe, refit_output / f"best_model_{best_model}_{best_feature_set}.joblib")
            except Exception:
                pass

        with open(refit_output / "best_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": best_model,
                    "feature_set": best_feature_set,
                    "primary_metric": run_cfg.primary_metric,
                    "summary_row": best_row,
                    "best_params": _clean_params_for_export(final_params, model=best_model),
                },
                f,
                indent=2,
            )

        with open(refit_output / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)

    return df_folds, df_summary, {"best_row": best_row, "best_params_per_combo": all_best_params}
