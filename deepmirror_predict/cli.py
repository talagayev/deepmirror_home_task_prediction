from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from deepmirror_predict.analysis.applicability_domain import (
    build_test_ad_table,
    compute_applicability_domain,
    plot_applicability_domain,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepMirror preprocessing, deduplication and AD CLI")
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
        help="Assess whether test molecules are in the applicability domain of the training set",
    )
    ad_parser.add_argument("--train-input", required=True, help="Path to training CSV")
    ad_parser.add_argument("--test-input", required=True, help="Path to test CSV")
    ad_parser.add_argument("--test-output", required=True, help="Path to save test AD CSV")
    ad_parser.add_argument("--plot-output", required=True, help="Path to save plot PNG")
    ad_parser.add_argument(
        "--smiles-column",
        default="SMILES_standardized",
        help="Column containing standardized SMILES",
    )
    ad_parser.add_argument(
        "--feature-kind",
        choices=["morgan", "avalon", "rdkit_path", "mordred", "chemeleon"],
        default="morgan",
        help="Feature representation used for AD analysis",
    )
    ad_parser.add_argument(
        "--similarity-metric",
        choices=["auto", "tanimoto", "cosine"],
        default="auto",
        help="Similarity metric",
    )
    ad_parser.add_argument(
        "--embedding-method",
        choices=["pca", "umap"],
        default="pca",
        help="2D embedding method for visualization",
    )
    ad_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest train neighbors used in mean_topk_similarity_to_train",
    )
    ad_parser.add_argument(
        "--user-cutoff",
        type=float,
        default=None,
        help="Optional user-defined AD cutoff on max_similarity_to_train",
    )
    ad_parser.add_argument(
        "--use-train-p5-cutoff",
        action="store_true",
        help="Also compute a training-derived cutoff as the 5th percentile of train nearest-neighbor similarity",
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
        df_train = pd.read_csv(args.train_input)
        df_test = pd.read_csv(args.test_input)

        if args.smiles_column not in df_train.columns:
            raise ValueError(f"Column '{args.smiles_column}' not found in training file")
        if args.smiles_column not in df_test.columns:
            raise ValueError(f"Column '{args.smiles_column}' not found in test file")

        df_train = df_train[df_train[args.smiles_column].notna()].copy()
        df_test = df_test[df_test[args.smiles_column].notna()].copy()

        train_smiles = df_train[args.smiles_column].astype(str).tolist()
        test_smiles = df_test[args.smiles_column].astype(str).tolist()

        result = compute_applicability_domain(
            train_smiles=train_smiles,
            test_smiles=test_smiles,
            feature_kind=args.feature_kind,
            similarity_metric=args.similarity_metric,
            embedding_method=args.embedding_method,
            top_k=args.top_k,
        )

        train_p5_cutoff = None
        if args.use_train_p5_cutoff:
            train_p5_cutoff = float(np.percentile(result.train_nn_similarity, 5))

        df_test_out = build_test_ad_table(
            df_test,
            test_max_similarity=result.test_max_similarity,
            test_mean_topk_similarity=result.test_mean_topk_similarity,
            user_cutoff=args.user_cutoff,
            train_p5_cutoff=train_p5_cutoff,
        )
        df_test_out.to_csv(args.test_output, index=False)

        plot_applicability_domain(
            embedding_coords=result.embedding_coords,
            n_train=len(df_train),
            test_max_similarity=result.test_max_similarity,
            out_path=args.plot_output,
            title_prefix=f"Applicability Domain ({args.feature_kind})",
            embedding_method=args.embedding_method,
            user_cutoff=args.user_cutoff,
            train_p5_cutoff=train_p5_cutoff,
        )

        print(f"n_train={len(df_train)}")
        print(f"n_test={len(df_test)}")
        print(f"test_max_similarity_min={float(result.test_max_similarity.min()):.4f}")
        print(f"test_max_similarity_median={float(np.median(result.test_max_similarity)):.4f}")
        print(f"test_max_similarity_max={float(result.test_max_similarity.max()):.4f}")

        if args.user_cutoff is not None:
            n_outside_user = int((df_test_out["outside_ad_user_cutoff"]).sum())
            print(f"user_cutoff={args.user_cutoff:.4f}")
            print(f"n_outside_ad_user_cutoff={n_outside_user}")
            print(f"pct_outside_ad_user_cutoff={100.0 * n_outside_user / len(df_test_out):.2f}")

        if train_p5_cutoff is not None:
            n_outside_p5 = int((df_test_out["outside_ad_train_p5"]).sum())
            print(f"train_p5_cutoff={train_p5_cutoff:.4f}")
            print(f"n_outside_ad_train_p5={n_outside_p5}")
            print(f"pct_outside_ad_train_p5={100.0 * n_outside_p5 / len(df_test_out):.2f}")


if __name__ == "__main__":
    main()
