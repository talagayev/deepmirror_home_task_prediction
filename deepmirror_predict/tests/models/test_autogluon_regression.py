import numpy as np
import pandas as pd
import pytest

autogluon = pytest.importorskip("autogluon")
pytest.importorskip("autogluon.tabular")

from deepmirror_predict.models.autogluon_regressor import (
    AutoGluonConfig,
    AutoGluonRegressor,
)


def _make_regression_data(n=120, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d,)).astype(np.float32)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def test_autogluon_fit_predict_runs_and_outputs_finite(tmp_path):
    X, y = _make_regression_data(n=100, d=12, seed=1)

    X_train, y_train = X[:80], y[:80]
    X_val, y_val = X[80:], y[80:]

    cfg = AutoGluonConfig(
        presets="medium_quality",
        time_limit_s=15,
        eval_metric="root_mean_squared_error",
        num_gpus=0,
        verbosity=0,
        path=str(tmp_path / "ag_fit_predict"),
        included_model_types=["RF"],
        dynamic_stacking=False,
    )

    model = AutoGluonRegressor(cfg=cfg, random_state=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    y_pred = model.predict(X_val)

    assert y_pred.shape == (len(y_val),)
    assert y_pred.dtype == np.float32
    assert np.isfinite(y_pred).all()


def test_autogluon_evaluate_metrics_returns_expected_keys(tmp_path):
    X, y = _make_regression_data(n=90, d=10, seed=2)

    X_train, y_train = X[:70], y[:70]
    X_val, y_val = X[70:], y[70:]

    cfg = AutoGluonConfig(
        presets="medium_quality",
        time_limit_s=15,
        eval_metric="root_mean_squared_error",
        num_gpus=0,
        verbosity=0,
        path=str(tmp_path / "ag_metrics"),
        included_model_types=["RF"],
        dynamic_stacking=False,
    )

    model = AutoGluonRegressor(cfg=cfg, random_state=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    metrics = model.evaluate_metrics(X_val, y_val)

    assert set(metrics.keys()) == {
        "rmse",
        "mse",
        "mae",
        "r2",
        "kendall_tau",
        "kendall_tau_pvalue",
        "spearman_rho",
        "spearman_pvalue",
    }
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["r2"])
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0

    assert np.isfinite(metrics["mse"])
    assert np.isfinite(metrics["kendall_tau"])
    assert np.isfinite(metrics["kendall_tau_pvalue"])
    assert np.isfinite(metrics["spearman_rho"])
    assert np.isfinite(metrics["spearman_pvalue"])
    
    assert metrics["mse"] >= 0.0
    assert -1.0 <= metrics["kendall_tau"] <= 1.0
    assert -1.0 <= metrics["spearman_rho"] <= 1.0
    assert 0.0 <= metrics["kendall_tau_pvalue"] <= 1.0
    assert 0.0 <= metrics["spearman_pvalue"] <= 1.0


def test_autogluon_best_model_helpers_work(tmp_path):
    X, y = _make_regression_data(n=100, d=12, seed=3)

    X_train, y_train = X[:80], y[:80]
    X_val, y_val = X[80:], y[80:]

    cfg = AutoGluonConfig(
        presets="medium_quality",
        time_limit_s=15,
        eval_metric="root_mean_squared_error",
        num_gpus=0,
        verbosity=0,
        path=str(tmp_path / "ag_best_model"),
        included_model_types=["RF"],
        dynamic_stacking=False,
    )

    model = AutoGluonRegressor(cfg=cfg, random_state=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    best_name = model.best_model_name()
    best_hparams_user = model.best_model_hyperparameters(output_format="user")
    best_hparams_all = model.best_model_hyperparameters(output_format="all")
    best_info = model.best_model_info()
    report = model.best_model_report(X_val, y_val)

    assert isinstance(best_name, str)
    assert len(best_name) > 0

    assert isinstance(best_hparams_user, dict)
    assert isinstance(best_hparams_all, dict)
    assert isinstance(best_info, dict)

    assert "best_model" in report
    assert "best_model_hparams_user" in report
    assert "best_model_hparams_all" in report
    assert "best_model_info" in report
    assert "metrics" in report

    assert report["best_model"] == best_name
    assert set(report["metrics"].keys()) == {
        "rmse",
        "mse",
        "mae",
        "r2",
        "kendall_tau",
        "kendall_tau_pvalue",
        "spearman_rho",
        "spearman_pvalue",
    }


def test_autogluon_leaderboard_returns_dataframe(tmp_path):
    X, y = _make_regression_data(n=100, d=12, seed=4)

    X_train, y_train = X[:80], y[:80]
    X_val, y_val = X[80:], y[80:]

    cfg = AutoGluonConfig(
        presets="medium_quality",
        time_limit_s=15,
        eval_metric="root_mean_squared_error",
        num_gpus=0,
        verbosity=0,
        path=str(tmp_path / "ag_leaderboard"),
        included_model_types=["RF"],
        dynamic_stacking=False,
    )

    model = AutoGluonRegressor(cfg=cfg, random_state=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    lb_train = model.leaderboard(extra_info=False)
    lb_val = model.leaderboard(X_val, y_val, extra_info=False)

    assert isinstance(lb_train, pd.DataFrame)
    assert isinstance(lb_val, pd.DataFrame)

    assert "model" in lb_train.columns
    assert "model" in lb_val.columns
    assert len(lb_train) >= 1
    assert len(lb_val) >= 1


def test_autogluon_predict_before_fit_raises(tmp_path):
    X, _ = _make_regression_data(n=20, d=6, seed=5)

    cfg = AutoGluonConfig(
        presets="medium_quality",
        time_limit_s=5,
        num_gpus=0,
        verbosity=0,
        path=str(tmp_path / "ag_not_fit"),
        included_model_types=["RF"],
        dynamic_stacking=False,
    )

    model = AutoGluonRegressor(cfg=cfg, random_state=0)

    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(X)


def test_autogluon_fit_raises_on_non_numeric_features(tmp_path):
    X = np.array([["CCO"], ["CCN"], ["CCC"]], dtype=object)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    cfg = AutoGluonConfig(
        presets="medium_quality",
        time_limit_s=5,
        num_gpus=0,
        verbosity=0,
        path=str(tmp_path / "ag_bad_input"),
        included_model_types=["RF"],
        dynamic_stacking=False,
    )

    model = AutoGluonRegressor(cfg=cfg, random_state=0)

    with pytest.raises(ValueError, match="numeric features"):
        model.fit(X, y)


def test_autogluon_fit_raises_on_wrong_shape(tmp_path):
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    cfg = AutoGluonConfig(
        presets="medium_quality",
        time_limit_s=5,
        num_gpus=0,
        verbosity=0,
        path=str(tmp_path / "ag_wrong_shape"),
        included_model_types=["RF"],
        dynamic_stacking=False,
    )

    model = AutoGluonRegressor(cfg=cfg, random_state=0)

    with pytest.raises(ValueError, match="2D feature matrix"):
        model.fit(X, y)
