# tests/test_metrics.py
import numpy as np

from deepmirror_predict.models.metrics import rmse, mae, r2


def test_rmse_zero_when_equal():
    """Test for case where RMSE is 0"""
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    """Test for RMSE with values"""
    y_true = np.array([0.0, 0.0], dtype=float)
    y_pred = np.array([3.0, 4.0], dtype=float)
    # sqrt(mean([9,16])) = sqrt(12.5)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(12.5))


def test_mae_zero_when_equal():
    """Test case for MAE when results are 0"""
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert mae(y, y) == 0.0


def test_mae_known_value():
    """Test MAE for non 0 values"""
    y_true = np.array([0.0, 0.0], dtype=float)
    y_pred = np.array([3.0, 4.0], dtype=float)
    assert mae(y_true, y_pred) == 3.5


def test_r2_perfect_is_one():
    """Test case for when R^2 is 1"""
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert r2(y, y) == 1.0


def test_r2_constant_target_returns_zero():
    """Test case for R^2 is 0"""
    y_true = np.array([2.0, 2.0, 2.0], dtype=float)
    y_pred = np.array([2.0, 2.0, 2.0], dtype=float)
    assert r2(y_true, y_pred) == 0.0