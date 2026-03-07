import numpy as np

from deepmirror_predict.models.metrics import (
    rmse,
    mse,
    mae,
    r2,
    kendall_tau,
    kendall_tau_pvalue,
    spearman_rho,
    spearman_pvalue,
)


def test_rmse_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0], dtype=float)
    y_pred = np.array([3.0, 4.0], dtype=float)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(12.5))


def test_mse_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert mse(y, y) == 0.0


def test_mse_known_value():
    y_true = np.array([0.0, 0.0], dtype=float)
    y_pred = np.array([3.0, 4.0], dtype=float)
    assert np.isclose(mse(y_true, y_pred), 12.5)


def test_mae_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert mae(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([0.0, 0.0], dtype=float)
    y_pred = np.array([3.0, 4.0], dtype=float)
    assert mae(y_true, y_pred) == 3.5


def test_r2_perfect_is_one():
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert r2(y, y) == 1.0


def test_r2_constant_target_returns_zero():
    y_true = np.array([2.0, 2.0, 2.0], dtype=float)
    y_pred = np.array([2.0, 2.0, 2.0], dtype=float)
    assert r2(y_true, y_pred) == 0.0


def test_rank_correlations_perfect_order():
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_pred = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)

    assert np.isclose(kendall_tau(y_true, y_pred), 1.0)
    assert np.isclose(spearman_rho(y_true, y_pred), 1.0)

    kp = kendall_tau_pvalue(y_true, y_pred)
    sp = spearman_pvalue(y_true, y_pred)

    assert np.isfinite(kp)
    assert np.isfinite(sp)
    assert 0.0 <= kp <= 1.0
    assert 0.0 <= sp <= 1.0


def test_rank_correlations_reverse_order():
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_pred = np.array([40.0, 30.0, 20.0, 10.0], dtype=float)

    assert np.isclose(kendall_tau(y_true, y_pred), -1.0)
    assert np.isclose(spearman_rho(y_true, y_pred), -1.0)


def test_rank_correlations_constant_prediction_are_safe():
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_pred = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)

    assert kendall_tau(y_true, y_pred) == 0.0
    assert spearman_rho(y_true, y_pred) == 0.0
    assert kendall_tau_pvalue(y_true, y_pred) == 1.0
    assert spearman_pvalue(y_true, y_pred) == 1.0
