from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def cv_module(monkeypatch):
    """Import cross_validation with lightweight stubs for optional heavy deps.

    The stubs are installed only for the duration of each test so they do not
    leak into the rest of the test suite.
    """
    sys.modules.pop("deepmirror_predict.models.cross_validation", None)

    chemeleon_feature_stub = types.ModuleType("deepmirror_predict.features.chemeleon")

    @dataclass(frozen=True)
    class CheMeleonConfig:
        pass

    def chemeleon_batch_from_smiles(smiles_list, cfg=None):
        return np.zeros((len(smiles_list), 1), dtype=np.float32), ["chemeleon_0"]

    chemeleon_feature_stub.CheMeleonConfig = CheMeleonConfig
    chemeleon_feature_stub.chemeleon_batch_from_smiles = chemeleon_batch_from_smiles
    monkeypatch.setitem(sys.modules, "deepmirror_predict.features.chemeleon", chemeleon_feature_stub)

    mordred_feature_stub = types.ModuleType("deepmirror_predict.features.mordred")

    def mordred2d_batch_from_smiles(smiles_list):
        return np.zeros((len(smiles_list), 1), dtype=np.float32), ["mordred_0"]

    def prune_mordred_matrix(X, names, max_nan_frac=0.2, drop_constant=True):
        return X, names

    mordred_feature_stub.mordred2d_batch_from_smiles = mordred2d_batch_from_smiles
    mordred_feature_stub.prune_mordred_matrix = prune_mordred_matrix
    monkeypatch.setitem(sys.modules, "deepmirror_predict.features.mordred", mordred_feature_stub)

    chemprop_stub = types.ModuleType("deepmirror_predict.models.chemprop_regression")

    @dataclass(frozen=True)
    class ChempropConfig:
        extra_descriptor_tokens: tuple[str, ...] = ()
        from_foundation: str | None = None

    class ChempropRegressor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.float32)

    chemprop_stub.ChempropConfig = ChempropConfig
    chemprop_stub.ChempropRegressor = ChempropRegressor
    monkeypatch.setitem(sys.modules, "deepmirror_predict.models.chemprop_regression", chemprop_stub)

    autogluon_stub = types.ModuleType("deepmirror_predict.models.autogluon_regressor")

    @dataclass(frozen=True)
    class AutoGluonConfig:
        presets: str = "best_quality"

    class AutoGluonRegressor:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, X, y, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.float32)

    autogluon_stub.AutoGluonConfig = AutoGluonConfig
    autogluon_stub.AutoGluonRegressor = AutoGluonRegressor
    monkeypatch.setitem(sys.modules, "deepmirror_predict.models.autogluon_regressor", autogluon_stub)

    module = importlib.import_module("deepmirror_predict.models.cross_validation")
    yield module
    sys.modules.pop("deepmirror_predict.models.cross_validation", None)


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": ["CC", "CCC", "c1ccccc1", "c1ccccc1O", "CCO", "CCN"],
            "group": ["a", "a", "b", "b", "c", "d"],
            "time": [1, 2, 3, 4, 5, 6],
        }
    )


def test_make_splits_random_returns_partitioned_folds(cv_module, simple_df):
    split_cfg = cv_module.SplitConfig(method="random", shuffle=True, random_state=7)

    splits = cv_module.make_splits(simple_df, "smiles", split_cfg, n_splits=3)

    assert len(splits) == 3
    seen_valid = []
    for tr_idx, va_idx in splits:
        assert len(tr_idx) + len(va_idx) == len(simple_df)
        assert set(tr_idx).isdisjoint(set(va_idx))
        seen_valid.extend(va_idx.tolist())

    assert sorted(seen_valid) == list(range(len(simple_df)))


def test_make_splits_group_requires_column(cv_module, simple_df):
    split_cfg = cv_module.SplitConfig(method="group", group_column=None)

    with pytest.raises(ValueError, match="group_column must be provided"):
        cv_module.make_splits(simple_df, "smiles", split_cfg, n_splits=2)


def test_make_splits_group_rejects_too_few_unique_groups(cv_module, simple_df):
    df = simple_df.copy()
    df["group"] = ["only"] * len(df)
    split_cfg = cv_module.SplitConfig(method="group", group_column="group")

    with pytest.raises(ValueError, match="unique groups"):
        cv_module.make_splits(df, "smiles", split_cfg, n_splits=2)


def test_make_splits_scaffold_produces_non_empty_train_and_valid(cv_module, simple_df):
    split_cfg = cv_module.SplitConfig(method="scaffold")

    splits = cv_module.make_splits(simple_df, "smiles", split_cfg, n_splits=2)

    assert len(splits) >= 2
    for tr_idx, va_idx in splits:
        assert len(tr_idx) > 0
        assert len(va_idx) > 0
        assert set(tr_idx).isdisjoint(set(va_idx))


def test_make_splits_time_series_requires_time_column(cv_module, simple_df):
    split_cfg = cv_module.SplitConfig(method="time_series", time_column=None)

    with pytest.raises(ValueError, match="time_column must be provided"):
        cv_module.make_splits(simple_df, "smiles", split_cfg, n_splits=3)


def test_single_validation_split_random_is_repeatable(cv_module, simple_df):
    split_cfg = cv_module.SplitConfig(method="random", random_state=123)

    tr1, va1 = cv_module._single_validation_split(simple_df, "smiles", split_cfg, valid_fraction=0.34)
    tr2, va2 = cv_module._single_validation_split(simple_df, "smiles", split_cfg, valid_fraction=0.34)

    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)
    assert len(va1) == 3


def test_single_validation_split_time_series_uses_tail_as_validation(cv_module, simple_df):
    split_cfg = cv_module.SplitConfig(method="time_series", time_column="time")

    tr_idx, va_idx = cv_module._single_validation_split(simple_df, "smiles", split_cfg, valid_fraction=0.34)

    assert tr_idx.tolist() == [0, 1, 2]
    assert va_idx.tolist() == [3, 4, 5]


def test_metric_fn_rejects_unknown_metric(cv_module):
    with pytest.raises(ValueError, match="Unknown metric"):
        cv_module._metric_fn("not_a_metric")


def test_parse_chemprop_feature_mode_valid_combo(cv_module):
    backbone, descriptors = cv_module._parse_chemprop_feature_mode("chemeleon+morgan_r2_b1024+avalon_b512")

    assert backbone == "chemeleon"
    assert descriptors == ["morgan_r2_b1024", "avalon_b512"]


@pytest.mark.parametrize(
    "feature_set",
    [
        "morgan_r2_b1024",
        "smiles+chemeleon",
        "smiles+chemeleon+morgan_r2_b1024",
    ],
)
def test_parse_chemprop_feature_mode_rejects_invalid_backbone_configs(cv_module, feature_set):
    with pytest.raises(ValueError, match="exactly one backbone token"):
        cv_module._parse_chemprop_feature_mode(feature_set)


def test_augment_fixed_params_for_chemprop_adds_backbone_and_descriptors(cv_module):
    params = cv_module._augment_fixed_params_for_combo(
        model_name="chemprop",
        feature_set="chemeleon+morgan_r2_b1024+avalon_b512",
        fixed_params={"extra_descriptor_tokens": ("morgan_r2_b1024",)},
    )

    assert params["from_foundation"] == "chemeleon"
    assert params["extra_descriptor_tokens"] == ("morgan_r2_b1024", "avalon_b512")


def test_augment_fixed_params_for_non_chemprop_is_passthrough(cv_module):
    original = {"max_depth": 10}

    result = cv_module._augment_fixed_params_for_combo("rf", "morgan_r2_b1024", original)

    assert result == original
    assert result is not original
