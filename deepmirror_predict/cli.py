from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from deepmirror_predict.analysis.applicability_domain import (
    build_dual_test_ad_table,
    combined_user_cutoff,
    compute_dual_applicability_domain,
    expand_feature_sets,
    plot_dual_applicability_domain,
    run_dual_applicability_domain_batch,
)
from deepmirror_predict.data_preprocession.dedpulication import deduplicate_smiles
from deepmirror_predict.data_preprocession.preprocessing import standardize_smiles
from deepmirror_predict.models.cross_validation import (
    FeatureConfig,
    OptimizationConfig,
    RunConfig,
    SplitConfig,
    run_nested_cross_validation,
)
from deepmirror_predict.models.predict_model import predict_from_refit


def preprocess_smiles_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "SMILES",
    output_column: str = "SMILES_std",
    reason_column: str = "SMILES_std_reason",
    keep_isomeric: bool = True,
    canonical_tautomer: bool = True,
    uncharge: bool = True,
    protonate: bool = False,
    ph: float = 7.4,
) -> pd.DataFrame:
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in input file")

    results = df[smiles_column].apply(
        lambda s: standardize_smiles(
            s,
            keep_isomeric=keep_isomeric,
            canonical_tautomer=canonical_tautomer,
            uncharge=uncharge,
            protonate=protonate,
            ph=ph,
        )
    )

    df = df.copy()
    df[output_column] = results.apply(lambda r: r.smiles_std)
    df[reason_column] = results.apply(lambda r: r.reason)
    return df


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_csv_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping/object at the top level")
    return data


def _get_command_config(config: dict, command_name: str) -> dict:
    section = config.get(command_name)
    if isinstance(section, dict):
        return section
    return config


def _get_arg_or_config(args, config: dict, name: str, default=None):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, default)


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as boolean")


def _resolve_bool_arg(args, config: dict, cli_name: str, config_name: str, default: bool) -> bool:
    cli_value = getattr(args, cli_name, None)
    if cli_value is not None:
        return cli_value
    return _as_bool(config.get(config_name), default)


def _ensure_int_list(value, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, str):
        return _parse_csv_int_list(value)
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(x) for x in value]
    raise ValueError(f"Expected int/list[str]/list[int], got {type(value)}")


def _expand_cv_feature_sets(
    base_feature_sets: list[str],
    *,
    morgan_radius: list[int] | int | None = None,
    morgan_bits: list[int] | int | None = None,
    avalon_bits: list[int] | int | None = None,
    rdkit_path_min: list[int] | int | None = None,
    rdkit_path_max: list[int] | int | None = None,
    rdkit_path_bits: list[int] | int | None = None,
) -> list[str]:
    """
    Like applicability-domain expand_feature_sets(), but also supports feature sets
    that include a Chemprop-only 'smiles' backbone token, e.g.:
      - smiles
      - smiles+morgan
      - smiles+morgan+avalon
      - smiles+rdkit_path+mordred
    """
    expanded: list[str] = []

    for feature_set in base_feature_sets:
        parts = [p.strip() for p in feature_set.split("+") if p.strip()]
        if not parts:
            continue

        if "smiles" not in parts:
            expanded.extend(
                expand_feature_sets(
                    [feature_set],
                    morgan_radius=morgan_radius,
                    morgan_bits=morgan_bits,
                    avalon_bits=avalon_bits,
                    rdkit_path_min=rdkit_path_min,
                    rdkit_path_max=rdkit_path_max,
                    rdkit_path_bits=rdkit_path_bits,
                )
            )
            continue

        other_parts = [p for p in parts if p != "smiles"]
        if not other_parts:
            expanded.append("smiles")
            continue

        expanded_suffixes = expand_feature_sets(
            ["+".join(other_parts)],
            morgan_radius=morgan_radius,
            morgan_bits=morgan_bits,
            avalon_bits=avalon_bits,
            rdkit_path_min=rdkit_path_min,
            rdkit_path_max=rdkit_path_max,
            rdkit_path_bits=rdkit_path_bits,
        )
        expanded.extend([f"smiles+{suffix}" for suffix in expanded_suffixes])

    seen: set[str] = set()
    deduped: list[str] = []
    for fs in expanded:
        if fs not in seen:
            seen.add(fs)
            deduped.append(fs)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepMirror preprocessing, deduplication, cross-validation and applicability-domain CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess-smiles",
        help="Standardize SMILES from a CSV column",
    )
    preprocess_parser.add_argument("--config", default=None, help="Path to YAML config file")
    preprocess_parser.add_argument("--input", default=None, help="Path to input CSV")
    preprocess_parser.add_argument("--output", default=None, help="Path to output CSV")
    preprocess_parser.add_argument("--smiles-column", default=None, help="Column containing SMILES strings")
    preprocess_parser.add_argument("--output-column", default=None, help="Column name for standardized SMILES")
    preprocess_parser.add_argument("--reason-column", default=None, help="Column name for preprocessing status/reason")

    preprocess_parser.add_argument(
        "--keep-isomeric",
        dest="keep_isomeric",
        action="store_true",
        default=None,
        help="Keep stereochemistry in output SMILES",
    )
    preprocess_parser.add_argument(
        "--no-keep-isomeric",
        dest="keep_isomeric",
        action="store_false",
        help="Do not keep stereochemistry in output SMILES",
    )

    preprocess_parser.add_argument(
        "--canonical-tautomer",
        dest="canonical_tautomer",
        action="store_true",
        default=None,
        help="Canonicalize tautomers",
    )
    preprocess_parser.add_argument(
        "--no-canonical-tautomer",
        dest="canonical_tautomer",
        action="store_false",
        help="Do not canonicalize tautomers",
    )

    preprocess_parser.add_argument(
        "--uncharge",
        dest="uncharge",
        action="store_true",
        default=None,
        help="Uncharge molecules",
    )
    preprocess_parser.add_argument(
        "--no-uncharge",
        dest="uncharge",
        action="store_false",
        help="Do not uncharge molecules",
    )

    preprocess_parser.add_argument(
        "--protonate",
        dest="protonate",
        action="store_true",
        default=None,
        help="Protonate molecules at the requested pH",
    )
    preprocess_parser.add_argument(
        "--no-protonate",
        dest="protonate",
        action="store_false",
        help="Do not protonate molecules",
    )

    preprocess_parser.add_argument("--ph", type=float, default=None, help="pH to use if protonation is enabled")

    dedup_parser = subparsers.add_parser(
        "deduplicate",
        help="Deduplicate rows by standardized SMILES and aggregate target values",
    )
    dedup_parser.add_argument("--config", default=None, help="Path to YAML config file")
    dedup_parser.add_argument("--input", default=None, help="Path to input CSV")
    dedup_parser.add_argument("--output", default=None, help="Path to output CSV")
    dedup_parser.add_argument("--key-cols", default=None, help="Comma-separated key columns")
    dedup_parser.add_argument("--target-col", default=None, help="Target column to aggregate")
    dedup_parser.add_argument("--method", choices=["mean", "median", "min", "max"], default=None)
    dedup_parser.add_argument("--keep-cols", default=None, help="Comma-separated metadata columns to keep")
    dedup_parser.add_argument("--prefer-col", default=None, help="Optional preference column")
    dedup_parser.add_argument("--prefer-value", default=None, help="Optional preferred value in prefer-col")

    dedup_parser.add_argument(
        "--keep-missing-keys",
        dest="keep_missing_keys",
        action="store_true",
        default=None,
        help="Keep rows with missing key columns",
    )
    dedup_parser.add_argument(
        "--drop-missing-keys",
        dest="keep_missing_keys",
        action="store_false",
        help="Drop rows with missing key columns",
    )

    dedup_parser.add_argument(
        "--keep-missing-target",
        dest="keep_missing_target",
        action="store_true",
        default=None,
        help="Keep rows with missing target",
    )
    dedup_parser.add_argument(
        "--drop-missing-target",
        dest="keep_missing_target",
        action="store_false",
        help="Drop rows with missing target",
    )

    cv_parser = subparsers.add_parser(
        "model-crossvalidation",
        help="Run nested cross-validation with configurable splits, features, models and metrics",
    )
    cv_parser.add_argument("--config", default=None, help="Path to YAML config file")

    predict_parser = subparsers.add_parser(
        "predict-model",
        help="Run inference with a saved refit model from nested cross-validation",
    )
    predict_parser.add_argument("--config", default=None, help="Path to YAML config file")
    predict_parser.add_argument("--input", default=None, help="Path to prediction CSV")
    predict_parser.add_argument("--refit-dir", default=None, help="Path to the saved refit directory")
    predict_parser.add_argument("--output", default=None, help="Path to output CSV with predictions")
    predict_parser.add_argument("--smiles-column", default=None, help="Column containing standardized SMILES")
    predict_parser.add_argument("--prediction-column", default=None, help="Name of output prediction column")

    ad_parser = subparsers.add_parser(
        "applicability-domain",
        help="Assess whether test molecules are in the applicability domain of the training set and a fine-tune subset",
    )
    ad_parser.add_argument("--config", default=None, help="Path to YAML config file")

    ad_parser.add_argument("--train-input", default=None, help="Path to training CSV")
    ad_parser.add_argument("--test-input", default=None, help="Path to test CSV")
    ad_parser.add_argument("--test-output", default=None, help="Path to output CSV with AD flags")
    ad_parser.add_argument("--plot-output", default=None, help="Path to output plot PNG")
    ad_parser.add_argument("--output-dir", default=None, help="Directory for batch mode outputs")
    ad_parser.add_argument("--smiles-column", default=None, help="Column containing standardized SMILES")

    ad_parser.add_argument("--feature-set", default=None, help="Single feature set for one run")
    ad_parser.add_argument(
        "--feature-sets",
        default=None,
        help=(
            "Comma-separated base feature sets for batch mode. "
            "Use morgan, avalon, rdkit_path, mordred or combinations like "
            "'morgan+avalon', 'rdkit_path+mordred'. Morgan/Avalon/RDKit path "
            "placeholders will be expanded using the provided size/radius/path settings."
        ),
    )

    ad_parser.add_argument(
        "--morgan-radius",
        default=None,
        help="Comma-separated Morgan radii to expand, e.g. 2,3",
    )
    ad_parser.add_argument(
        "--morgan-bits",
        default=None,
        help="Comma-separated Morgan bit sizes to expand, e.g. 512,1024",
    )
    ad_parser.add_argument(
        "--avalon-bits",
        default=None,
        help="Comma-separated Avalon bit sizes to expand, e.g. 512,1024",
    )
    ad_parser.add_argument(
        "--rdkit-path-min",
        default=None,
        help="Comma-separated RDKit path minimum lengths to expand, e.g. 1",
    )
    ad_parser.add_argument(
        "--rdkit-path-max",
        default=None,
        help="Comma-separated RDKit path maximum lengths to expand, e.g. 5,7",
    )
    ad_parser.add_argument(
        "--rdkit-path-bits",
        default=None,
        help="Comma-separated RDKit path bit sizes to expand, e.g. 512,1024",
    )

    ad_parser.add_argument(
        "--embedding-method",
        choices=["pca", "tsne"],
        default=None,
        help="2D embedding method for visualization",
    )
    ad_parser.add_argument("--top-k", type=int, default=None, help="Top-k neighbors for mean similarity")
    ad_parser.add_argument(
        "--tanimoto-cutoff",
        type=float,
        default=None,
        help="User cutoff for binary fingerprints (Morgan/Avalon/RDKit path)",
    )
    ad_parser.add_argument(
        "--cosine-cutoff",
        type=float,
        default=None,
        help="User cutoff for continuous descriptors (Mordred)",
    )
    ad_parser.add_argument(
        "--use-train-p5-cutoff",
        action="store_true",
        help="Also report AD flags using the 5th percentile of training nearest-neighbor similarity",
    )
    ad_parser.add_argument(
        "--best-by",
        choices=["finetune_p5", "all_p5", "median_finetune", "median_all"],
        default=None,
        help="Criterion to select the best feature set in batch mode",
    )
    ad_parser.add_argument(
        "--finetune-subset-column",
        default=None,
        help="Column in training data identifying the fine-tune subset",
    )
    ad_parser.add_argument(
        "--finetune-subset-values",
        default=None,
        help="Comma-separated values in finetune-subset-column to define the fine-tune subset",
    )

    args = parser.parse_args()

    if args.command == "preprocess-smiles":
        full_config = _load_yaml_config(args.config) if args.config else {}
        config = _get_command_config(full_config, args.command)

        input_path = _get_arg_or_config(args, config, "input")
        output_path = _get_arg_or_config(args, config, "output")
        smiles_column = _get_arg_or_config(args, config, "smiles_column", "SMILES")
        output_column = _get_arg_or_config(args, config, "output_column", "SMILES_standardized")
        reason_column = _get_arg_or_config(args, config, "reason_column", "preprocess_reason")

        keep_isomeric = _resolve_bool_arg(args, config, "keep_isomeric", "keep_isomeric", True)
        canonical_tautomer = _resolve_bool_arg(args, config, "canonical_tautomer", "canonical_tautomer", True)
        uncharge = _resolve_bool_arg(args, config, "uncharge", "uncharge", True)
        protonate = _resolve_bool_arg(args, config, "protonate", "protonate", False)
        ph = float(_get_arg_or_config(args, config, "ph", 7.4))

        if input_path is None or output_path is None:
            raise ValueError("input and output must be provided via CLI or YAML config")

        df = pd.read_csv(input_path)
        df_out = preprocess_smiles_dataframe(
            df,
            smiles_column=smiles_column,
            output_column=output_column,
            reason_column=reason_column,
            keep_isomeric=keep_isomeric,
            canonical_tautomer=canonical_tautomer,
            uncharge=uncharge,
            protonate=protonate,
            ph=ph,
        )
        _ensure_parent_dir(output_path)
        df_out.to_csv(output_path, index=False)

    elif args.command == "deduplicate":
        full_config = _load_yaml_config(args.config) if args.config else {}
        config = _get_command_config(full_config, args.command)

        input_path = _get_arg_or_config(args, config, "input")
        output_path = _get_arg_or_config(args, config, "output")
        key_cols_value = _get_arg_or_config(args, config, "key_cols", "SMILES_standardized")
        target_col = _get_arg_or_config(args, config, "target_col")
        method = _get_arg_or_config(args, config, "method", "mean")
        keep_cols_value = _get_arg_or_config(args, config, "keep_cols", "")
        prefer_col = _get_arg_or_config(args, config, "prefer_col")
        prefer_value = _get_arg_or_config(args, config, "prefer_value")

        keep_missing_keys = _resolve_bool_arg(args, config, "keep_missing_keys", "keep_missing_keys", False)
        keep_missing_target = _resolve_bool_arg(args, config, "keep_missing_target", "keep_missing_target", False)

        if input_path is None or output_path is None or target_col is None:
            raise ValueError("input, output and target_col must be provided via CLI or YAML config")

        df = pd.read_csv(input_path)

        if isinstance(key_cols_value, str):
            key_cols = _parse_csv_list(key_cols_value)
        elif isinstance(key_cols_value, list):
            key_cols = [str(x).strip() for x in key_cols_value if str(x).strip()]
        else:
            raise ValueError("key_cols in config must be a list or comma-separated string")

        if isinstance(keep_cols_value, str):
            keep_cols = _parse_csv_list(keep_cols_value) if keep_cols_value else []
        elif isinstance(keep_cols_value, list):
            keep_cols = [str(x).strip() for x in keep_cols_value if str(x).strip()]
        else:
            raise ValueError("keep_cols in config must be a list or comma-separated string")

        df_out = deduplicate_smiles(
            df=df,
            key_cols=key_cols,
            target_col=target_col,
            method=method,
            keep_cols=keep_cols,
            drop_missing_keys=not keep_missing_keys,
            drop_missing_target=not keep_missing_target,
            prefer_col=prefer_col,
            prefer_value=prefer_value,
        )
        _ensure_parent_dir(output_path)
        df_out.to_csv(output_path, index=False)

    elif args.command == "model-crossvalidation":
        full_config = _load_yaml_config(args.config) if args.config else {}
        config = _get_command_config(full_config, args.command)

        input_path = _get_arg_or_config(args, config, "input")
        output_dir = config.get("output_dir", "cv_outputs")
        smiles_column = config.get("smiles_column", "SMILES_standardized")
        target_column = config.get("target_column")
        row_id_column = config.get("row_id_column")

        if input_path is None or target_column is None:
            raise ValueError("input and target_column must be provided in YAML config")

        models = config.get("models", ["rf"])
        if isinstance(models, str):
            models = _parse_csv_list(models)

        feature_sets = config.get("feature_sets", [config.get("feature_set", "morgan")])
        if isinstance(feature_sets, str):
            base_feature_sets = _parse_csv_list(feature_sets)
        elif isinstance(feature_sets, list):
            base_feature_sets = [str(x).strip() for x in feature_sets if str(x).strip()]
        else:
            raise ValueError("feature_sets in config must be a list or comma-separated string")

        metrics = config.get("metrics", ["rmse", "mae", "r2"])
        if isinstance(metrics, str):
            metrics = _parse_csv_list(metrics)

        morgan_radius = _ensure_int_list(config.get("morgan_radius"), [3])
        morgan_bits = _ensure_int_list(config.get("morgan_bits"), [1024])
        avalon_bits = _ensure_int_list(config.get("avalon_bits"), [1024])
        rdkit_path_min = _ensure_int_list(config.get("rdkit_path_min"), [1])
        rdkit_path_max = _ensure_int_list(config.get("rdkit_path_max"), [7])
        rdkit_path_bits = _ensure_int_list(config.get("rdkit_path_bits"), [1024])

        expanded_feature_sets = _expand_cv_feature_sets(
            base_feature_sets,
            morgan_radius=morgan_radius,
            morgan_bits=morgan_bits,
            avalon_bits=avalon_bits,
            rdkit_path_min=rdkit_path_min,
            rdkit_path_max=rdkit_path_max,
            rdkit_path_bits=rdkit_path_bits,
        )

        split_raw = config.get("split", {})
        split_cfg = SplitConfig(
            method=split_raw.get("method", "random"),
            outer_folds=int(split_raw.get("outer_folds", 5)),
            inner_folds=int(split_raw.get("inner_folds", 5)),
            shuffle=bool(split_raw.get("shuffle", True)),
            random_state=int(split_raw.get("random_state", config.get("random_state", 0))),
            group_column=split_raw.get("group_column"),
            time_column=split_raw.get("time_column"),
            time_ascending=bool(split_raw.get("time_ascending", True)),
            scaffold_include_chirality=bool(split_raw.get("scaffold_include_chirality", False)),
        )

        feature_raw = config.get("features", {})
        feature_cfg = FeatureConfig(
            feature_set=expanded_feature_sets[0] if expanded_feature_sets else "morgan_r3_b1024",
            mordred_max_nan_frac=float(feature_raw.get("mordred_max_nan_frac", 0.2)),
            mordred_drop_constant=bool(feature_raw.get("mordred_drop_constant", True)),
        )

        optimization_raw = config.get("optimization", {})
        optimization_cfg = OptimizationConfig(
            enabled=bool(optimization_raw.get("enabled", False)),
            metric=optimization_raw.get("metric", config.get("primary_metric", "rmse")),
            n_trials=int(optimization_raw.get("n_trials", 30)),
            timeout_s=optimization_raw.get("timeout_s"),
            random_state=int(optimization_raw.get("random_state", config.get("random_state", 0))),
        )

        run_cfg = RunConfig(
            input_path=input_path,
            output_dir=output_dir,
            smiles_column=smiles_column,
            target_column=target_column,
            metrics=tuple(metrics),
            primary_metric=config.get("primary_metric", "rmse"),
            models=tuple(models),
            feature_sets=tuple(expanded_feature_sets),
            split=split_cfg,
            feature_params=feature_cfg,
            optimization=optimization_cfg,
            dropna_target=bool(config.get("dropna_target", True)),
            dropna_smiles=bool(config.get("dropna_smiles", True)),
            row_id_column=row_id_column,
            refit_best_model=bool(config.get("refit_best_model", True)),
            refit_validation_fraction=float(config.get("refit_validation_fraction", 0.1)),
            save_fold_predictions=bool(config.get("save_fold_predictions", True)),
            save_best_params=bool(config.get("save_best_params", True)),
            random_state=int(config.get("random_state", 0)),
            n_jobs=int(config.get("n_jobs", -1)),
            model_params=config.get("model_params", {}),
        )

        _, df_summary, artifact = run_nested_cross_validation(run_cfg)
        print(df_summary.to_string(index=False))
        best_row = artifact["best_row"]
        print(f"best_model={best_row['model']}")
        print(f"best_feature_set={best_row['feature_set']}")
        print(f"primary_metric={run_cfg.primary_metric}")
        print(f"best_score={best_row[f'{run_cfg.primary_metric}_mean']}")

    elif args.command == "predict-model":
        full_config = _load_yaml_config(args.config) if args.config else {}
        config = _get_command_config(full_config, args.command)

        input_path = _get_arg_or_config(args, config, "input")
        refit_dir = _get_arg_or_config(args, config, "refit_dir")
        output_path = _get_arg_or_config(args, config, "output")
        smiles_column = _get_arg_or_config(args, config, "smiles_column", "SMILES_standardized")
        prediction_column = _get_arg_or_config(args, config, "prediction_column", "prediction")

        if input_path is None or refit_dir is None or output_path is None:
            raise ValueError("input, refit_dir and output must be provided via CLI or YAML config")

        df_pred = predict_from_refit(
            input_path=input_path,
            refit_dir=refit_dir,
            output_path=output_path,
            smiles_column=smiles_column,
            prediction_column=prediction_column,
        )
        print(f"wrote_predictions={output_path}")
        print(f"n_rows={len(df_pred)}")

    elif args.command == "applicability-domain":
        full_config = _load_yaml_config(args.config) if args.config else {}
        config = _get_command_config(full_config, args.command)

        train_input = _get_arg_or_config(args, config, "train_input")
        test_input = _get_arg_or_config(args, config, "test_input")
        test_output = _get_arg_or_config(args, config, "test_output")
        plot_output = _get_arg_or_config(args, config, "plot_output")
        output_dir = _get_arg_or_config(args, config, "output_dir")
        smiles_column = _get_arg_or_config(args, config, "smiles_column", "SMILES_standardized")

        feature_set = _get_arg_or_config(args, config, "feature_set", "morgan")
        feature_sets = _get_arg_or_config(args, config, "feature_sets")

        embedding_method = _get_arg_or_config(args, config, "embedding_method", "pca")
        top_k = int(_get_arg_or_config(args, config, "top_k", 5))
        tanimoto_cutoff = _get_arg_or_config(args, config, "tanimoto_cutoff")
        cosine_cutoff = _get_arg_or_config(args, config, "cosine_cutoff")
        best_by = _get_arg_or_config(args, config, "best_by", "finetune_p5")

        morgan_radius = args.morgan_radius
        if morgan_radius is not None:
            morgan_radius = _parse_csv_int_list(morgan_radius)
        else:
            morgan_radius = config.get("morgan_radius")

        morgan_bits = args.morgan_bits
        if morgan_bits is not None:
            morgan_bits = _parse_csv_int_list(morgan_bits)
        else:
            morgan_bits = config.get("morgan_bits")

        avalon_bits = args.avalon_bits
        if avalon_bits is not None:
            avalon_bits = _parse_csv_int_list(avalon_bits)
        else:
            avalon_bits = config.get("avalon_bits")

        rdkit_path_min = _get_arg_or_config(args, config, "rdkit_path_min", [1])
        if isinstance(rdkit_path_min, str):
            rdkit_path_min = _parse_csv_int_list(rdkit_path_min)
        elif isinstance(rdkit_path_min, int):
            rdkit_path_min = [rdkit_path_min]

        rdkit_path_max = _get_arg_or_config(args, config, "rdkit_path_max", [7])
        if isinstance(rdkit_path_max, str):
            rdkit_path_max = _parse_csv_int_list(rdkit_path_max)
        elif isinstance(rdkit_path_max, int):
            rdkit_path_max = [rdkit_path_max]

        rdkit_path_bits = _get_arg_or_config(args, config, "rdkit_path_bits", [1024])
        if isinstance(rdkit_path_bits, str):
            rdkit_path_bits = _parse_csv_int_list(rdkit_path_bits)
        elif isinstance(rdkit_path_bits, int):
            rdkit_path_bits = [rdkit_path_bits]

        use_train_p5_cutoff = args.use_train_p5_cutoff or bool(config.get("use_train_p5_cutoff", False))

        finetune_subset_column = _get_arg_or_config(args, config, "finetune_subset_column", None)
        finetune_subset_values = _get_arg_or_config(args, config, "finetune_subset_values", None)
        if isinstance(finetune_subset_values, str):
            finetune_subset_values = _parse_csv_list(finetune_subset_values)

        if train_input is None or test_input is None:
            raise ValueError("train_input and test_input must be provided via CLI or YAML config")

        df_train = pd.read_csv(train_input)
        df_test = pd.read_csv(test_input)

        if smiles_column not in df_train.columns:
            raise ValueError(f"Column '{smiles_column}' not found in training file")
        if smiles_column not in df_test.columns:
            raise ValueError(f"Column '{smiles_column}' not found in test file")

        if finetune_subset_column is None or not finetune_subset_values:
            raise ValueError("finetune_subset_column and finetune_subset_values must be provided")

        if finetune_subset_column not in df_train.columns:
            raise ValueError(f"Column '{finetune_subset_column}' not found in training file")

        df_train = df_train[df_train[smiles_column].notna()].copy()
        df_test = df_test[df_test[smiles_column].notna()].copy()

        finetune_mask = df_train[finetune_subset_column].isin(finetune_subset_values)
        df_train_finetune = df_train[finetune_mask].copy()
        df_train_other = df_train[~finetune_mask].copy()

        if len(df_train_finetune) == 0:
            raise ValueError(
                "Fine-tune subset filter produced zero rows. "
                f"column={finetune_subset_column}, values={finetune_subset_values}"
            )

        finetune_subset_label = ",".join(finetune_subset_values)

        if feature_sets:
            if isinstance(feature_sets, str):
                base_feature_sets = _parse_csv_list(feature_sets)
            elif isinstance(feature_sets, list):
                base_feature_sets = [str(x).strip() for x in feature_sets if str(x).strip()]
            else:
                raise ValueError("feature_sets in config must be a list or comma-separated string")

            expanded_feature_sets = expand_feature_sets(
                base_feature_sets,
                morgan_radius=morgan_radius,
                morgan_bits=morgan_bits,
                avalon_bits=avalon_bits,
                rdkit_path_min=rdkit_path_min,
                rdkit_path_max=rdkit_path_max,
                rdkit_path_bits=rdkit_path_bits,
            )

            if output_dir is None:
                raise ValueError("output_dir is required in batch mode")

            summary = run_dual_applicability_domain_batch(
                df_train_other=df_train_other,
                df_train_finetune=df_train_finetune,
                df_test=df_test,
                smiles_column=smiles_column,
                feature_sets=expanded_feature_sets,
                output_dir=output_dir,
                embedding_method=embedding_method,
                top_k=top_k,
                tanimoto_cutoff=tanimoto_cutoff,
                cosine_cutoff=cosine_cutoff,
                use_train_p5_cutoff=use_train_p5_cutoff,
                best_by=best_by,
                finetune_subset_label=finetune_subset_label,
            )
            print(summary.to_string(index=False))

        else:
            if test_output is None or plot_output is None:
                raise ValueError("test_output and plot_output are required in single-run mode")

            train_other_smiles = df_train_other[smiles_column].astype(str).tolist()
            train_finetune_smiles = df_train_finetune[smiles_column].astype(str).tolist()
            test_smiles = df_test[smiles_column].astype(str).tolist()

            result = compute_dual_applicability_domain(
                train_other_smiles=train_other_smiles,
                train_finetune_smiles=train_finetune_smiles,
                test_smiles=test_smiles,
                feature_set=feature_set,
                embedding_method=embedding_method,
                top_k=top_k,
            )

            user_cutoff = combined_user_cutoff(
                feature_set,
                tanimoto_cutoff=tanimoto_cutoff,
                cosine_cutoff=cosine_cutoff,
            )

            train_all_p5_cutoff = None
            train_finetune_p5_cutoff = None
            if use_train_p5_cutoff:
                train_all_p5_cutoff = float(np.percentile(result.train_all_nn_similarity, 5))
                train_finetune_p5_cutoff = float(np.percentile(result.train_finetune_nn_similarity, 5))

            df_test_out = build_dual_test_ad_table(
                df_test,
                feature_set=feature_set,
                test_max_similarity_to_train_all=result.test_max_similarity_to_train_all,
                test_mean_topk_similarity_to_train_all=result.test_mean_topk_similarity_to_train_all,
                test_max_similarity_to_train_finetune=result.test_max_similarity_to_train_finetune,
                test_mean_topk_similarity_to_train_finetune=result.test_mean_topk_similarity_to_train_finetune,
                user_cutoff=user_cutoff,
                train_all_p5_cutoff=train_all_p5_cutoff,
                train_finetune_p5_cutoff=train_finetune_p5_cutoff,
            )
            _ensure_parent_dir(test_output)
            df_test_out.to_csv(test_output, index=False)

            _ensure_parent_dir(plot_output)
            plot_dual_applicability_domain(
                embedding_coords=result.embedding_coords,
                n_train_other=result.n_train_other,
                n_train_finetune=result.n_train_finetune,
                test_max_similarity_to_train_all=result.test_max_similarity_to_train_all,
                test_max_similarity_to_train_finetune=result.test_max_similarity_to_train_finetune,
                out_path=plot_output,
                title_prefix=f"Applicability Domain ({feature_set})",
                embedding_method=embedding_method,
                user_cutoff=user_cutoff,
                train_all_p5_cutoff=train_all_p5_cutoff,
                train_finetune_p5_cutoff=train_finetune_p5_cutoff,
            )

            print(f"feature_set={feature_set}")
            print(f"finetune_subset_column={finetune_subset_column}")
            print(f"finetune_subset_values={finetune_subset_values}")
            print(f"n_train_other={len(df_train_other)}")
            print(f"n_train_finetune={len(df_train_finetune)}")
            print(f"n_test={len(df_test)}")
            print(f"median_similarity_to_train_all={float(np.median(result.test_max_similarity_to_train_all)):.4f}")
            print(f"median_similarity_to_train_finetune={float(np.median(result.test_max_similarity_to_train_finetune)):.4f}")

            if user_cutoff is not None:
                n_outside_user_all = int(df_test_out["outside_ad_train_all_user_cutoff"].sum())
                n_outside_user_ft = int(df_test_out["outside_ad_train_finetune_user_cutoff"].sum())
                print(f"user_cutoff={user_cutoff:.4f}")
                print(f"n_outside_ad_user_cutoff_all={n_outside_user_all}")
                print(f"pct_outside_ad_user_cutoff_all={100.0 * n_outside_user_all / len(df_test_out):.2f}")
                print(f"n_outside_ad_user_cutoff_finetune={n_outside_user_ft}")
                print(f"pct_outside_ad_user_cutoff_finetune={100.0 * n_outside_user_ft / len(df_test_out):.2f}")

            if train_all_p5_cutoff is not None:
                n_outside_p5_all = int(df_test_out["outside_ad_train_all_p5"].sum())
                print(f"train_all_p5_cutoff={train_all_p5_cutoff:.4f}")
                print(f"n_outside_ad_train_all_p5={n_outside_p5_all}")
                print(f"pct_outside_ad_train_all_p5={100.0 * n_outside_p5_all / len(df_test_out):.2f}")

            if train_finetune_p5_cutoff is not None:
                n_outside_p5_ft = int(df_test_out["outside_ad_train_finetune_p5"].sum())
                print(f"train_finetune_p5_cutoff={train_finetune_p5_cutoff:.4f}")
                print(f"n_outside_ad_train_finetune_p5={n_outside_p5_ft}")
                print(f"pct_outside_ad_train_finetune_p5={100.0 * n_outside_p5_ft / len(df_test_out):.2f}")


if __name__ == "__main__":
    main()
