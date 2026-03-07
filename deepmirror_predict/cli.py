from __future__ import annotations

import argparse
import pandas as pd
from rdkit import Chem

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SMILES in a CSV file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess-smiles",
        help="Standardize SMILES from a CSV column",
    )
    preprocess_parser.add_argument("--input", required=True, help="Path to input CSV")
    preprocess_parser.add_argument("--output", required=True, help="Path to output CSV")
    preprocess_parser.add_argument(
        "--smiles-column",
        default="SMILES",
        help="Column containing SMILES strings",
    )
    preprocess_parser.add_argument(
        "--output-column",
        default="SMILES_std",
        help="Column name for standardized SMILES",
    )
    preprocess_parser.add_argument(
        "--reason-column",
        default="SMILES_std_reason",
        help="Column name for preprocessing status/reason",
    )
    preprocess_parser.add_argument(
        "--keep-isomeric",
        action="store_true",
        default=True,
        help="Keep stereochemistry in output SMILES",
    )
    preprocess_parser.add_argument(
        "--canonical-tautomer",
        action="store_true",
        default=True,
        help="Canonicalize tautomers",
    )
    preprocess_parser.add_argument(
        "--uncharge",
        action="store_true",
        default=True,
        help="Uncharge molecules",
    )
    preprocess_parser.add_argument(
        "--protonate",
        action="store_true",
        default=False,
        help="Protonate molecules at the requested pH",
    )
    preprocess_parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH to use if protonation is enabled",
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


if __name__ == "__main__":
    main()