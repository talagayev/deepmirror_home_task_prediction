from __future__ import annotations

import argparse

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


def _get_arg_or_config(args, config: dict, name: str, default=None):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepMirror preprocessing, deduplication and applicability-domain CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess-smiles",
        help="Standardize SMILES from a CSV column",
    )
    preprocess_parser.add_argument("--input", required=True, help="Path to input CSV")
    preprocess_parser.add_argument("--output", required=True, help="Path to output CSV")
    preprocess_parser.add_argument("--smiles-column", default="SMILES", help="Column containing SMILES strings")
    preprocess_parser.add_argument("--output-column", default="SMILES_std", help="Column name for standardized SMILES")
    preprocess_parser.add_argument("--reason-column", default="SMILES_std_reason", help="Column name for preprocessing status/reason")
    preprocess_parser.add_argument("--keep-isomeric", action="store_true", default=True, help="Keep stereochemistry in output SMILES")
    preprocess_parser.add_argument("--canonical-tautomer", action="store_true", default=True, help="Canonicalize tautomers")
    preprocess_parser.add_argument("--uncharge", action="store_true", default=True, help="Uncharge molecules")
    preprocess_parser.add_argument("--protonate", action="store_true", default=False, help="Protonate molecules at the requested pH")
    preprocess_parser.add_argument("--ph", type=float, default=7.4, help="pH to use if protonation is enabled")

    dedup_parser = subparsers.add_parser(
        "deduplicate",
        help="Deduplicate rows by standardized SMILES and aggregate target values",
    )
    dedup_parser.add_argument("--input", required=True, help="Path to input CSV")
    dedup_parser.add_argument("--output", required=True, help="Path to output CSV")
    dedup_parser.add_argument("--key-cols", default="SMILES_standardized", help="Comma-separated key columns")
    dedup_parser.add_argument("--target-col", required=True, help="Target column to aggregate")
    dedup_parser.add_argument("--method", choices=["mean", "median", "min", "max"], default="mean")
    dedup_parser.add_argument("--keep-cols", default="", help="Comma-separated metadata columns to keep")
    dedup_parser.add_argument("--prefer-col", default=None, help="Optional preference column")
    dedup_parser.add_argument("--prefer-value", default=None, help="Optional preferred value in prefer-col")
    dedup_parser.add_argument("--keep-missing-keys", action="store_true")
    dedup_parser.add_argument("--keep-missing-target", action="store_true")

    ad_parser = subparsers.add_parser(
        "applicability-domain",
        help="Assess whether test molecules are in the applicability domain of the training set and a fine-tune subset",
    )
    ad_parser.add_argument("--config", default=None, help="Path to YAML config file")
    ad_parser.add_argument("--train-input", default=None, help="Path to training CSV")
    ad_parser.add_argument("--test-input", default=None, help="Path to test CSV")
    ad_parser.add_argument("--test-output", default=None, help="Single-run output CSV")
    ad_parser.add_argument("--plot-output", default=None, help="Single-run plot PNG")
    ad_parser.add_argument("--output-dir", default=None, help="Batch-mode output directory")
    ad_parser.add_argument("--smiles-column", default=None, help="Column containing standardized SMILES")

    ad_parser.add_argument("--feature-set", default=None, help="Single feature set")
    ad_parser.add_argument("--feature-sets", default=None, help="Comma-separated base feature sets for batch mode")

    ad_parser.add_argument("--embedding-method", choices=["pca", "umap"], default=None)
    ad_parser.add_argument("--top-k", type=int, default=None)
    ad_parser.add_argument("--tanimoto-cutoff", type=float, default=None)
    ad_parser.add_argument("--cosine-cutoff", type=float, default=None)
    ad_parser.add_argument(
        "--best-by",
        choices=["all_train_p5", "finetune_p5", "user_cutoff_all", "user_cutoff_finetune"],
        default=None,
    )

    ad_parser.add_argument("--morgan-radius", default=None)
    ad_parser.add_argument("--morgan-bits", default=None)
    ad_parser.add_argument("--avalon-bits", default=None)
    ad_parser.add_argument("--rdkit-path-min", default=None)
    ad_parser.add_argument("--rdkit-path-max", default=None)
    ad_parser.add_argument("--rdkit-path-bits", default=None)

    ad_parser.add_argument("--use-train-p5-cutoff", action="store_true")

    ad_parser.add_argument(
        "--finetune-subset-column",
        default=None,
        help="Training column used to define fine-tune subset, e.g. dataset",
    )
    ad_parser.add_argument(
        "--finetune-subset-values",
        default=None,
        help="Comma-separated values for fine-tune subset, e.g. openadmet_expansion_train,openadmet_polaris",
    )

    args = parser.parse_args()

    if args.command == "preprocess-smiles":
        df = pd.read_csv(args.input)

        df_out = preprocess_smiles_dataframe(
            df=df,
            smiles_column=args.smiles_column,
            output_column=args.output_column,
            reason_column=args.reason_column,
            keep_isomeric=args.keep_isomeric,
            canonical_tautomer=args.canonical_tautomer,
            uncharge=args.uncharge,
            protonate=args.protonate,
            ph=args.ph,
        )
        df_out.to_csv(args.output, index=False)

    elif args.command == "deduplicate":
        df = pd.read_csv(args.input)
        key_cols = _parse_csv_list(args.key_cols)
        keep_cols = _parse_csv_list(args.keep_cols) if args.keep_cols else []

        df_out = deduplicate_smiles(
            df=df,
            key_cols=key_cols,
            target_col=args.target_col,
            method=args.method,
            keep_cols=keep_cols,
            drop_missing_keys=not args.keep_missing_keys,
            drop_missing_target=not args.keep_missing_target,
            prefer_col=args.prefer_col,
            prefer_value=args.prefer_value,
        )
        df_out.to_csv(args.output, index=False)

    elif args.command == "applicability-domain":
        config = _load_yaml_config(args.config) if args.config else {}

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
            df_test_out.to_csv(test_output, index=False)

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
