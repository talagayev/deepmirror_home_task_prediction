# tests/test_auto_sklearn2.py
import numpy as np

from deepmirror_predict.models.auto_ml_sklearn import (
    AutoSklearn2Config,
    fit_predict_autosklearn2,
    autosklearn2_selected,
    autosklearn2_cv_table,
)


def _make_regression_data(n=80, d=24, seed=0):
    """Make data for regression"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal((d,), dtype=np.float32)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def test_fit_predict_autosklearn2_runs_and_outputs_finite():
    """Test that autosklearn generates data and data is finite"""
    X, y = _make_regression_data(n=90, d=32, seed=1)
    X_train, y_train = X[:70], y[:70]
    X_val, y_val = X[70:], y[70:]

    cfg = AutoSklearn2Config(
        time_limit=5,  # keep test fast
        random_state=0,
        scoring="neg_mean_squared_error",
        exclude_models=("ransac",),  # avoid fold-size failures on high-D data
    )

    y_pred, model = fit_predict_autosklearn2(X_train, y_train, X_val, cfg=cfg)

    assert y_pred.shape == (len(y_val),)
    assert y_pred.dtype == np.float32
    assert np.isfinite(y_pred).all()


def test_autosklearn2_selected_contains_best_params_and_cv_rmse_when_neg_mse():
    """Test if the best parameteres are returned"""
    X, y = _make_regression_data(n=90, d=16, seed=2)
    X_train, y_train = X[:70], y[:70]
    X_val = X[70:]

    cfg = AutoSklearn2Config(
        time_limit=5,
        random_state=0,
        scoring="neg_mean_squared_error",
        exclude_models=("ransac",),
    )
    _, model = fit_predict_autosklearn2(X_train, y_train, X_val, cfg=cfg)

    info = autosklearn2_selected(model)

    assert "best_params" in info
    assert isinstance(info["best_params"], dict)
    assert "preprocessor" in info["best_params"]
    assert "regressor" in info["best_params"]

    assert info["scoring"] == "neg_mean_squared_error"
    assert "best_cv_rmse" in info
    assert np.isfinite(info["best_cv_rmse"])
    assert info["best_cv_rmse"] >= 0.0


def test_autosklearn2_cv_table_has_rmse_and_is_sorted():
    """Test information about RMSE"""
    X, y = _make_regression_data(n=90, d=16, seed=3)
    X_train, y_train = X[:70], y[:70]
    X_val = X[70:]

    cfg = AutoSklearn2Config(
        time_limit=5,
        random_state=0,
        scoring="neg_mean_squared_error",
        exclude_models=("ransac",),
    )
    _, model = fit_predict_autosklearn2(X_train, y_train, X_val, cfg=cfg)

    df = autosklearn2_cv_table(model)

    assert {"pipeline", "cv_score", "cv_rmse"}.issubset(df.columns)
    assert len(df) >= 1

    # sorted ascending by cv_rmse (lower is better)
    rmse_vals = df["cv_rmse"].to_numpy(dtype=float)
    assert np.all(rmse_vals[:-1] <= rmse_vals[1:] + 1e-12)