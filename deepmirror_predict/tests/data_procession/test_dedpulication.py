# tests/test_deduplicate.py
import numpy as np
import pandas as pd
import pytest

from deepmirror_predict.data_preprocession.dedpulication import deduplicate_smiles


def test_dedup_mean_basic_with_smiles_like_keys():
    """Test if dedpulcation recognizes identical Smiles"""
    df = pd.DataFrame(
        {
            "smiles_std": ["CCO", "CCO", "c1ccccc1"],
            "activity": [1.0, 3.0, 10.0],
        }
    )

    out = deduplicate_smiles(
        df,
        key_cols=("smiles_std",),
        target_col="activity",
        method="mean",
    )

    assert set(out["smiles_std"]) == {"CCO", "c1ccccc1"}

    e = out[out["smiles_std"] == "CCO"].iloc[0]
    b = out[out["smiles_std"] == "c1ccccc1"].iloc[0]

    assert e["y"] == 2.0
    assert e["n_reps"] == 2
    assert np.isclose(e["y_std"], np.std([1.0, 3.0], ddof=1))
    assert e["y_min"] == 1.0
    assert e["y_max"] == 3.0

    assert b["y"] == 10.0
    assert b["n_reps"] == 1
    assert np.isnan(b["y_std"])
    assert b["y_min"] == 10.0
    assert b["y_max"] == 10.0


def test_dedup_keeps_first_nonnull_metadata_with_smiles_like_keys():
    df = pd.DataFrame(
        {
            "smiles_std": ["CCN", "CCN", "CCN"],
            "activity": [1.0, 2.0, 3.0],
            "assay_id": [np.nan, "assay_1", "assay_2"],
            "smiles": ["CCN", "C-C-N", "CC[NH2]"],
        }
    )

    out = deduplicate_smiles(
        df,
        key_cols=("smiles_std",),
        target_col="activity",
        method="median",
        keep_cols=("assay_id", "smiles"),
    )
    row = out.iloc[0]

    # Here we check if it uses the metadata from the first with metadata
    assert row["smiles_std"] == "CCN"
    assert row["y"] == 2.0
    assert row["assay_id"] == "assay_1"
    assert row["smiles"] == "CCN"


def test_dedup_drops_missing_keys_and_targets_by_default():
    """"""
    df = pd.DataFrame(
        {
            "smiles_std": ["CCO", None, "c1ccccc1"],
            "activity": [1.0, 2.0, None],
        }
    )
    out = deduplicate_smiles(df, key_cols=("smiles_std",), target_col="activity", method="mean")
    # Check if only CCO survived, since the others are missing smiles or activity
    assert len(out) == 1
    assert out.shape[0] == 1
    assert out.iloc[0]["smiles_std"] == "CCO"
    assert out.iloc[0]["y"] == 1.0


def test_dedup_missing_columns_raise():
    df = pd.DataFrame({"smiles_std": ["CCO"], "activity": [1.0]})

    with pytest.raises(ValueError, match="Missing target_col"):
        deduplicate_smiles(df, key_cols=("smiles_std",), target_col="y")

    with pytest.raises(ValueError, match="Missing key_cols"):
        deduplicate_smiles(df, key_cols=("nope",), target_col="activity")