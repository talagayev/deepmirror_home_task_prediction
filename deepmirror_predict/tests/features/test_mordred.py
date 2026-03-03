# tests/test_mordred2d.py
import numpy as np
import pytest

from deepmirror_predict.features.mordred import (
    mordred2d_feature_names,
    mordred2d_from_smiles,
    mordred2d_batch_from_smiles,
    prune_mordred_matrix,
)


def test_mordred_feature_names_nonempty_unique():
    """Test that the features are recognized correctly"""
    names = mordred2d_feature_names()
    assert isinstance(names, list)
    assert len(names) > 100  # should be around 1600
    assert len(set(names)) == len(names)


def test_mordred_from_smiles_shape_dtype_and_nans_possible():
    """Test feature generation"""
    names = mordred2d_feature_names()
    v = mordred2d_from_smiles("CC(=O)O")
    assert v is not None
    assert v.shape == (len(names),)
    assert v.dtype == np.float32


def test_mordred_invalid_smiles_returns_none():
    """Test invalid smiles case"""
    v = mordred2d_from_smiles("not_a_smiles")
    assert v is None


def test_mordred_batch_from_smiles_shape():
    """Test correct shape recognition"""
    smiles = ["CC(=O)O", "c1ccccc1", "CCN"]
    X, names = mordred2d_batch_from_smiles(smiles)
    assert X.shape == (len(smiles), len(names))
    assert X.dtype == np.float32


def test_mordred_batch_invalid_smiles_raises():
    """Test ValuError raising during invalid smiles"""
    smiles = ["CC(=O)O", "not_a_smiles"]
    with pytest.raises(ValueError, match="Invalid SMILES encountered in processed data"):
        mordred2d_batch_from_smiles(smiles)


def test_prune_mordred_matrix_reduces_or_equal_dims():
    """Test mordred feature pruning"""
    # identical molecules => many descriptor columns identical --> prunning
    smiles = ["CC(=O)O"] * 5
    X, names = mordred2d_batch_from_smiles(smiles)

    Xp, namesp = prune_mordred_matrix(X, names, max_nan_frac=1.0, drop_constant=True)

    assert Xp.shape[0] == X.shape[0]
    assert Xp.shape[1] <= X.shape[1]
    assert len(namesp) == Xp.shape[1]