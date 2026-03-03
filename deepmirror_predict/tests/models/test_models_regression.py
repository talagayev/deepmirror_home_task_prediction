# tests/test_modeling_factory.py
import numpy as np
import pytest

from deepmirror_predict.models.models_regression import (
    TrainConfig,
    build_pipeline,
    default_train_config,
    fit_predict,
)


def _make_regression_data(n=80, d=32, seed=0):
    """Preparation of data for the models tests"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d,)).astype(np.float32)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


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


@pytest.mark.parametrize(
    "model,params",
    [
        ("rf", {"n_estimators": 50, "max_depth": 10, "random_state": 0}),
        ("svm", {"C": 3.0, "epsilon": 0.1, "gamma": "scale"}),
        ("xgb", {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "random_state": 0}),
        ("lgbm", {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05, "random_state": 0}),
    ],
)
def test_fit_predict_runs_and_outputs_finite(model, params):
    X, y = _make_regression_data(n=100, d=24, seed=1)

    X_train, y_train = X[:75], y[:75]
    X_val, y_val = X[75:], y[75:]

    y_pred, pipe = fit_predict(
        X_train,
        y_train,
        X_val,
        model=model,
        params=params,
    )

    # check that there are results
    assert y_pred.shape == (len(y_val),)
    assert y_pred.dtype == np.float32
    # Check that no NaN appear
    assert np.isfinite(y_pred).all()


def test_fit_predict_handles_nans_with_imputation_rf():
    X, y = _make_regression_data(n=90, d=16, seed=2)

    # introduce NaNs in train and val
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

    # Single-thread + no subsampling => deterministic in practice
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