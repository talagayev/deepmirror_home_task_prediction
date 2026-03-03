# tests/test_chemeleon.py
import numpy as np
import pytest

from deepmirror_predict.features.chemeleon import (
    CheMeleonConfig,
    CheMeleonFingerprint,
    chemeleon_batch_from_smiles,
    chemeleon_feature_names,
)


def test_chemeleon_feature_names_length_unique():
    """Test if the feature name recognition is correct"""
    names = chemeleon_feature_names(128)
    assert len(names) == 128
    assert len(set(names)) == 128
    assert names[0] == "chemeleon_0"
    assert names[-1] == "chemeleon_127"


def test_chemeleon_embedding_shape_dtype_and_names_match():
    """Test feature generation"""
    cfg = CheMeleonConfig(device="cpu", reduce_dim=64)
    smiles = ["CCO", "c1ccccc1"]

    X, names = chemeleon_batch_from_smiles(smiles, cfg=cfg)

    assert X.shape == (2, 64)
    assert X.dtype == np.float32
    assert len(names) == 64
    assert names[0] == "chemeleon_0"
    assert names[-1] == "chemeleon_63"


def test_chemeleon_is_deterministic_in_eval_mode():
    cfg = CheMeleonConfig(device="cpu", reduce_dim=32)
    fp = CheMeleonFingerprint(cfg)

    smiles = ["CCO", "c1ccccc1"]
    X1 = fp(smiles)
    X2 = fp(smiles)

    assert X1.shape == (2, 32)
    assert X2.shape == (2, 32)
    assert X1.dtype == np.float32
    assert X2.dtype == np.float32

    # Should be identical in eval mode; allow tiny tolerance for platform differences.
    assert np.allclose(X1, X2, rtol=0.0, atol=1e-6)


def test_chemeleon_invalid_smiles_raises():
    """Test case when invalid smiles are parsed"""
    cfg = CheMeleonConfig(device="cpu")
    fp = CheMeleonFingerprint(cfg)

    with pytest.raises(ValueError):
        fp(["not_a_smiles"])