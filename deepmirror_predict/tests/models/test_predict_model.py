from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest



def _import_predict_model_module():
    """Import predict_model with lightweight stubs for optional heavy deps."""
    sys.modules.pop("deepmirror_predict.models.predict_model", None)
    sys.modules.pop("deepmirror_predict.models.cross_validation", None)

    cv_stub = types.ModuleType("deepmirror_predict.models.cross_validation")

    @dataclass(frozen=True)
    class FeatureConfig:
        feature_set: str = "morgan_r3_b1024"

    def featurize_smiles(smiles_list, feature_cfg):
        raise AssertionError("Test should monkeypatch featurize_smiles before use")

    cv_stub.FeatureConfig = FeatureConfig
    cv_stub.featurize_smiles = featurize_smiles
    sys.modules["deepmirror_predict.models.cross_validation"] = cv_stub

    return importlib.import_module("deepmirror_predict.models.predict_model")


pm = _import_predict_model_module()


class DummyPipe:
    def __init__(self):
        self.last_X = None

    def predict(self, X):
        self.last_X = X
        X = np.asarray(X)
        return np.arange(1, len(X) + 1, dtype=np.float32)


def test_load_refit_model_path_requires_exactly_one_candidate(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No saved refit model"):
        pm._load_refit_model_path(tmp_path)

    (tmp_path / "best_model_a.joblib").write_bytes(b"a")
    (tmp_path / "best_model_b.joblib").write_bytes(b"b")
    with pytest.raises(ValueError, match="Expected exactly one"):
        pm._load_refit_model_path(tmp_path)


def test_align_feature_matrix_reorders_and_pads_with_nan():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    aligned = pm._align_feature_matrix(
        X,
        current_feature_names=["f_b", "f_a"],
        expected_feature_names=["f_a", "f_missing", "f_b"],
    )

    assert aligned.shape == (2, 3)
    assert aligned.dtype == np.float32
    assert np.allclose(aligned[:, 0], [2.0, 4.0])
    assert np.isnan(aligned[:, 1]).all()
    assert np.allclose(aligned[:, 2], [1.0, 3.0])


def test_predict_from_refit_requires_metadata_file(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"SMILES_standardized": ["CC"]}).to_csv(input_csv, index=False)

    with pytest.raises(FileNotFoundError, match="Missing metadata file"):
        pm.predict_from_refit(
            input_path=str(input_csv),
            refit_dir=str(tmp_path),
            output_path=str(tmp_path / "pred.csv"),
        )


def test_predict_from_refit_requires_smiles_column(tmp_path: Path):
    refit_dir = tmp_path / "refit"
    refit_dir.mkdir()
    (refit_dir / "best_model_metadata.json").write_text(json.dumps({"model": "rf", "feature_set": "morgan_r2_b8"}))
    joblib.dump(DummyPipe(), refit_dir / "best_model_rf.joblib")

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"wrong_col": ["CC"]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="Column 'SMILES_standardized' not found"):
        pm.predict_from_refit(
            input_path=str(input_csv),
            refit_dir=str(refit_dir),
            output_path=str(tmp_path / "pred.csv"),
        )


def test_predict_from_refit_featurizes_aligns_and_writes_csv(monkeypatch, tmp_path: Path):
    refit_dir = tmp_path / "refit"
    refit_dir.mkdir()
    (refit_dir / "best_model_metadata.json").write_text(
        json.dumps({"model": "rf", "feature_set": "morgan_r2_b8"}),
        encoding="utf-8",
    )
    (refit_dir / "feature_names.json").write_text(json.dumps(["f1", "f_missing", "f0"]), encoding="utf-8")

    pipe = DummyPipe()
    joblib.dump(pipe, refit_dir / "best_model_rf.joblib")

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"SMILES_standardized": ["CC", "CCC"], "row_id": [1, 2]}).to_csv(input_csv, index=False)

    calls = {}

    def fake_featurize_smiles(smiles_list, feature_cfg):
        calls["smiles_list"] = list(smiles_list)
        calls["feature_set"] = feature_cfg.feature_set
        return np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32), ["f0", "f1"]

    monkeypatch.setattr(pm, "featurize_smiles", fake_featurize_smiles)

    output_csv = tmp_path / "predictions" / "pred.csv"
    df_out = pm.predict_from_refit(
        input_path=str(input_csv),
        refit_dir=str(refit_dir),
        output_path=str(output_csv),
        prediction_column="pred",
    )

    assert calls == {"smiles_list": ["CC", "CCC"], "feature_set": "morgan_r2_b8"}
    assert output_csv.exists()
    assert list(df_out.columns) == ["SMILES_standardized", "row_id", "pred"]
    assert df_out["pred"].tolist() == [1.0, 2.0]

    saved = pd.read_csv(output_csv)
    assert saved["pred"].tolist() == [1.0, 2.0]

    loaded_pipe = joblib.load(refit_dir / "best_model_rf.joblib")
    assert loaded_pipe.last_X is None


def test_predict_from_refit_chemprop_uses_raw_smiles_without_featurization(monkeypatch, tmp_path: Path):
    refit_dir = tmp_path / "refit"
    refit_dir.mkdir()
    (refit_dir / "best_model_metadata.json").write_text(
        json.dumps({"model": "chemprop", "feature_set": "smiles"}),
        encoding="utf-8",
    )

    pipe = DummyPipe()
    joblib.dump(pipe, refit_dir / "best_model_chemprop.joblib")

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"SMILES_standardized": ["CC", "CCC"]}).to_csv(input_csv, index=False)

    def fail_featurize(*args, **kwargs):
        raise AssertionError("featurize_smiles should not be called for chemprop refit prediction")

    monkeypatch.setattr(pm, "featurize_smiles", fail_featurize)

    df_out = pm.predict_from_refit(
        input_path=str(input_csv),
        refit_dir=str(refit_dir),
        output_path=str(tmp_path / "pred.csv"),
    )

    assert df_out["prediction"].tolist() == [1.0, 2.0]
