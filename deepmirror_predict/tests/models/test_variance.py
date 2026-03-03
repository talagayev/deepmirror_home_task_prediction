# tests/test_feature_select.py
import numpy as np
import pytest

from deepmirror_predict.models.variance import VarianceThreshold


def test_binary_frequency_filter_frac_drops_rare_and_common_bits():
    # 10 samples, 5 features
    # f0: 0/10 on  -> drop (too rare)
    # f1: 1/10 on  -> drop if min_frac=0.2
    # f2: 3/10 on  -> keep
    # f3: 9/10 on  -> drop if max_frac=0.8
    # f4: 10/10 on -> drop (too common)
    X = np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )

    filt = VarianceThreshold(min_frac=0.2, max_frac=0.8)
    X2 = filt.fit_transform(X)

    support = filt.get_support()
    assert support.shape == (5,)
    assert support.tolist() == [False, False, True, False, False]

    # Only f2 remains
    assert X2.shape == (10, 1)
    assert np.all((X2 == 0.0) | (X2 == 1.0))


def test_binary_frequency_filter_frac_transform_matches_support():
    X_train = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    # freqs: f0=0.0, f1=0.2, f2=0.6

    filt = VarianceThreshold(min_frac=0.2, max_frac=0.9).fit(X_train)
    assert filt.get_support().tolist() == [False, True, True]

    X_val = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.float32)
    X_val2 = filt.transform(X_val)
    assert X_val2.shape == (2, 2)
    assert np.array_equal(X_val2, X_val[:, [1, 2]])


def test_binary_frequency_filter_frac_transform_before_fit_raises():
    filt = VarianceThreshold(min_frac=0.1, max_frac=0.9)
    with pytest.raises(RuntimeError, match="Call fit\\(\\) first"):
        filt.transform(np.zeros((2, 3), dtype=np.float32))


def test_binary_frequency_filter_frac_invalid_bounds_raise():
    X = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        VarianceThreshold(min_frac=-0.1, max_frac=0.9).fit(X)

    with pytest.raises(ValueError):
        VarianceThreshold(min_frac=0.1, max_frac=1.1).fit(X)

    with pytest.raises(ValueError):
        VarianceThreshold(min_frac=0.9, max_frac=0.1).fit(X)