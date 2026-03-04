# deepmirror_predict/tests/models/test_models_regression.py
import numpy as np
import pytest

# Normal imports (no importorskip)
import chemprop  # noqa: F401
import lightning  # noqa: F401
import torch  # noqa: F401

from deepmirror_predict.models.models_regression import (
    TrainConfig,
    build_pipeline,
    default_train_config,
    fit_predict,
)


def _make_regression_data(n=80, d=32, seed=0):
    """Preparation of data for the numeric models tests"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d,)).astype(np.float32)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _make_smiles_regression_data():
    # Small, valid SMILES set for chemprop test
    smiles = [
        "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
        "CCO", "CCCO", "CCCCO", "CCN", "CCCN",
        "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
        "CC(=O)O", "CC(=O)N", "CCS",
        "COC", "CCCl", "CCBr", "CCF",
    ]
    y = np.array([len(s) + (i % 3) * 0.1 for i, s in enumerate(smiles)], dtype=np.float32)
    return np.array(smiles, dtype=object), y


def test_default_train_config_impute_detects_nans():
    """Check if the impute works"""
    X, _ = _make_regression_data()
    X[0, 0] = np.nan

    cfg_rf = default_train_config("rf", X)
    assert cfg_rf.impute == "median"
    assert cfg_rf.scale is False

    cfg_svm = default_train_config("svm", X)
    assert cfg_svm.impute == "median"
    assert cfg_svm.scale is True


def test_default_train_config_chemprop_no_preprocessing():
    X = np.array(["CCO", "c1ccccc1"], dtype=object)
    cfg = default_train_config("chemprop", X)
    assert cfg.impute is None
    assert cfg.scale is False


def test_build_pipeline_steps_for_svm():
    """test building"""
    X, _ = _make_regression_data()
    X[0, 0] = np.nan

    cfg = TrainConfig(model="svm", impute="median", scale=True)
    pipe = build_pipeline(X, cfg=cfg, params={"C": 1.0, "epsilon": 0.1})

    names = [name for name, _ in pipe.steps]
    assert names == ["imputer", "scaler", "model"]


def test_build_pipeline_rf_has_no_scaler():
    """test building of RF"""
    X, _ = _make_regression_data()
    cfg = TrainConfig(model="rf", impute=None, scale=False)
    pipe = build_pipeline(X, cfg=cfg, params={"n_estimators": 10, "random_state": 0})

    names = [name for name, _ in pipe.steps]
    assert names[-1] == "model"
    assert "scaler" not in names


def test_build_pipeline_chemprop_has_only_model_step(tmp_path):
    X = np.array(["CCO", "c1ccccc1"], dtype=object)
    cfg = TrainConfig(model="chemprop", impute=None, scale=False)
    pipe = build_pipeline(
        X,
        cfg=cfg,
        params={"max_epochs": 1, "early_stopping": False, "checkpoint_dir": str(tmp_path)},
    )
    names = [name for name, _ in pipe.steps]
    assert names == ["model"]


@pytest.mark.parametrize(
    "model,params",
    [
        ("rf", {"n_estimators": 50, "max_depth": 10, "random_state": 0}),
        ("svm", {"C": 3.0, "epsilon": 0.1, "gamma": "scale"}),
        ("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "random_state": 0}),
        ("lgbm", {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "random_state": 0}),
    ],
)
def test_fit_predict_runs_and_outputs_finite_numeric(model, params):
    X, y = _make_regression_data(n=100, d=24, seed=1)

    X_train, y_train = X[:75], y[:75]
    X_val, y_val = X[75:], y[75:]

    y_pred, _pipe = fit_predict(
        X_train,
        y_train,
        X_val,
        model=model,
        params=params,
    )

    assert y_pred.shape == (len(y_val),)
    assert y_pred.dtype == np.float32
    assert np.isfinite(y_pred).all()


def test_fit_predict_runs_and_outputs_finite_chemprop(tmp_path):
    X, y = _make_smiles_regression_data()

    X_train, y_train = X[:16], y[:16]
    X_val, y_val = X[16:], y[16:]

    # Keep it extremely small/fast and CPU-only for CI.
    params = {
        "message_hidden_dim": 64,
        "message_depth": 2,
        "message_dropout": 0.0,
        "ffn_hidden_dim": 64,
        "ffn_layers": 1,
        "ffn_dropout": 0.0,
        "batch_norm": False,
        "warmup_epochs": 1,
        "init_lr": 1e-4,
        "max_lr": 3e-4,
        "final_lr": 1e-4,
        "max_epochs": 2,
        "batch_size": 8,
        "num_workers": 0,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "early_stopping": False,
        "checkpoint_dir": str(tmp_path),
    }

    y_pred, _pipe = fit_predict(
        X_train,
        y_train,
        X_val,
        model="chemprop",
        params=params,
        y_valid=y_val,
    )

    assert y_pred.shape == (len(y_val),)
    assert y_pred.dtype == np.float32
    assert np.isfinite(y_pred).all()


def test_fit_predict_chemprop_requires_y_valid(tmp_path):
    X, y = _make_smiles_regression_data()
    X_train, y_train = X[:16], y[:16]
    X_val = X[16:]

    params = {
        "max_epochs": 1,
        "batch_size": 8,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "early_stopping": False,
        "checkpoint_dir": str(tmp_path),
    }

    with pytest.raises(ValueError):
        fit_predict(X_train, y_train, X_val, model="chemprop", params=params)


def test_fit_predict_handles_nans_with_imputation_rf():
    X, y = _make_regression_data(n=90, d=16, seed=2)

    X[0, 0] = np.nan
    X[5, 3] = np.nan
    X[80, 7] = np.nan

    X_train, y_train = X[:70], y[:70]
    X_val, y_val = X[70:], y[70:]

    y_pred, _ = fit_predict(X_train, y_train, X_val, model="rf", params={"n_estimators": 50, "random_state": 0})
    assert y_pred.shape == (len(y_val),)
    assert np.isfinite(y_pred).all()


def test_fit_predict_repeatable_for_rf_with_fixed_seed():
    X, y = _make_regression_data(n=90, d=20, seed=3)
    X_train, y_train = X[:70], y[:70]
    X_val = X[70:]

    params = {"n_estimators": 100, "random_state": 123}

    y_pred1, _ = fit_predict(X_train, y_train, X_val, model="rf", params=params)
    y_pred2, _ = fit_predict(X_train, y_train, X_val, model="rf", params=params)

    assert np.allclose(y_pred1, y_pred2, rtol=0.0, atol=0.0)


def test_fit_predict_repeatable_for_svm_with_fixed_params():
    X, y = _make_regression_data(n=120, d=20, seed=10)
    X_train, y_train = X[:90], y[:90]
    X_val = X[90:]

    params = {"C": 3.0, "epsilon": 0.1, "gamma": "scale", "kernel": "rbf"}

    y_pred1, _ = fit_predict(X_train, y_train, X_val, model="svm", params=params)
    y_pred2, _ = fit_predict(X_train, y_train, X_val, model="svm", params=params)

    assert np.allclose(y_pred1, y_pred2, rtol=0.0, atol=0.0)


def test_fit_predict_repeatable_for_xgb_with_fixed_seed_single_thread():
    X, y = _make_regression_data(n=140, d=24, seed=11)
    X_train, y_train = X[:110], y[:110]
    X_val = X[110:]

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 123,
        "n_jobs": 1,
        "tree_method": "hist",
    }

    y_pred1, _ = fit_predict(X_train, y_train, X_val, model="xgb", params=params)
    y_pred2, _ = fit_predict(X_train, y_train, X_val, model="xgb", params=params)

    assert np.allclose(y_pred1, y_pred2, rtol=0.0, atol=1e-6)


def test_fit_predict_repeatable_for_lgbm_with_fixed_seed_single_thread():
    X, y = _make_regression_data(n=140, d=24, seed=12)
    X_train, y_train = X[:110], y[:110]
    X_val = X[110:]

    params = {
        "n_estimators": 500,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 123,
        "n_jobs": 1,
    }

    y_pred1, _ = fit_predict(X_train, y_train, X_val, model="lgbm", params=params)
    y_pred2, _ = fit_predict(X_train, y_train, X_val, model="lgbm", params=params)

    assert np.allclose(y_pred1, y_pred2, rtol=0.0, atol=1e-6)