import numpy as np
import pandas as pd
import pytest

from deepmirror_predict.data_preprocession.dedpulication import deduplicate_smiles


def test_dedup_mean_basic_with_smiles_like_keys():
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

    assert e["activity_deduplicated"] == 2.0
    assert e["n_reps"] == 2
    assert np.isclose(e["y_std"], np.std([1.0, 3.0], ddof=1))
    assert e["y_min"] == 1.0
    assert e["y_max"] == 3.0

    assert b["activity_deduplicated"] == 10.0
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

    assert row["smiles_std"] == "CCN"
    assert row["activity_deduplicated"] == 2.0
    assert row["assay_id"] == "assay_1"
    assert row["smiles"] == "CCN"


def test_dedup_drops_missing_keys_and_targets_by_default():
    df = pd.DataFrame(
        {
            "smiles_std": ["CCO", None, "c1ccccc1"],
            "activity": [1.0, 2.0, None],
        }
    )

    out = deduplicate_smiles(
        df,
        key_cols=("smiles_std",),
        target_col="activity",
        method="mean",
    )

    assert len(out) == 1
    assert out.iloc[0]["smiles_std"] == "CCO"
    assert out.iloc[0]["activity_deduplicated"] == 1.0


def test_dedup_missing_columns_raise():
    df = pd.DataFrame({"smiles_std": ["CCO"], "activity": [1.0]})

    with pytest.raises(ValueError, match="Missing target_col"):
        deduplicate_smiles(df, key_cols=("smiles_std",), target_col="y")

    with pytest.raises(ValueError, match="Missing key_cols"):
        deduplicate_smiles(df, key_cols=("nope",), target_col="activity")


def test_dedup_prefer_false_when_group_contains_true_and_false():
    df = pd.DataFrame(
        {
            "SMILES_standardized": ["CCO", "CCO", "c1ccccc1"],
            "activity": [1.0, 3.0, 10.0],
            "scaling_was_applied": [False, True, True],
            "dataset": ["fang_dataset", "chembl_dataset", "chembl_dataset"],
        }
    )

    out = deduplicate_smiles(
        df,
        key_cols=("SMILES_standardized",),
        target_col="activity",
        method="mean",
        keep_cols=("dataset", "scaling_was_applied"),
        prefer_col="scaling_was_applied",
        prefer_value=False,
    )

    row = out[out["SMILES_standardized"] == "CCO"].iloc[0]
    assert row["activity_deduplicated"] == 1.0
    assert row["n_reps"] == 1
    assert row["scaling_was_applied"] == False


def test_dedup_preference_does_not_apply_when_group_is_uniform():
    df = pd.DataFrame(
        {
            "SMILES_standardized": ["CCO", "CCO"],
            "activity": [1.0, 3.0],
            "scaling_was_applied": [True, True],
        }
    )

    out = deduplicate_smiles(
        df,
        key_cols=("SMILES_standardized",),
        target_col="activity",
        method="mean",
        prefer_col="scaling_was_applied",
        prefer_value=False,
    )

    row = out.iloc[0]
    assert row["activity_deduplicated"] == 2.0
    assert row["n_reps"] == 2


def test_dedup_preference_requires_value_if_column_is_given():
    df = pd.DataFrame(
        {
            "SMILES_standardized": ["CCO"],
            "activity": [1.0],
            "scaling_was_applied": [False],
        }
    )

    with pytest.raises(ValueError, match="prefer_value must be provided"):
        deduplicate_smiles(
            df,
            key_cols=("SMILES_standardized",),
            target_col="activity",
            prefer_col="scaling_was_applied",
        )
