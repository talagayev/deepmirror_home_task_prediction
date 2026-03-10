from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from deepmirror_predict.models.cross_validation import FeatureConfig, featurize_smiles


def _load_refit_model_path(refit_dir: Path) -> Path:
    candidates = sorted(refit_dir.glob("best_model_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No saved refit model found in {refit_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Expected exactly one best_model_*.joblib in {refit_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _align_feature_matrix(
    X: np.ndarray,
    current_feature_names: list[str],
    expected_feature_names: list[str],
) -> np.ndarray:
    if current_feature_names == expected_feature_names:
        return np.asarray(X, dtype=np.float32)

    current_df = pd.DataFrame(X, columns=current_feature_names)
    aligned_df = current_df.reindex(columns=expected_feature_names, fill_value=np.nan)
    return aligned_df.to_numpy(dtype=np.float32, copy=False)


def predict_from_refit(
    *,
    input_path: str,
    refit_dir: str,
    output_path: str,
    smiles_column: str = "SMILES_standardized",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    refit_path = Path(refit_dir)
    metadata_path = refit_path / "best_model_metadata.json"
    feature_names_path = refit_path / "feature_names.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names_expected: list[str] | None = None
    if feature_names_path.exists():
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_names_expected = list(json.load(f))

    model_name = str(metadata["model"])
    feature_set = str(metadata["feature_set"])
    model_path = _load_refit_model_path(refit_path)
    pipe = joblib.load(model_path)

    df = pd.read_csv(input_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in input file")

    smiles = df[smiles_column].astype(str).tolist()

    if model_name == "chemprop":
        X_pred = smiles
    else:
        X_pred, feature_names_current = featurize_smiles(
            smiles,
            FeatureConfig(feature_set=feature_set),
        )
        if feature_names_expected is not None:
            X_pred = _align_feature_matrix(X_pred, feature_names_current, feature_names_expected)

    y_pred = np.asarray(pipe.predict(X_pred), dtype=np.float32).reshape(-1)

    df_out = df.copy()
    df_out[prediction_column] = y_pred
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    return df_out
