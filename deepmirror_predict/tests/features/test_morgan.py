# tests/test_morgan.py
import numpy as np

from deepmirror_predict.features.morgan import (
    MorganFPConfig,
    morgan_bits_from_smiles,
    morgan_bits_feature_names,
)


def test_morgan_bits_valid_shape_dtype():
    """Test to see if the shape and type is correct"""
    cfg = MorganFPConfig(radius=2, n_bits=1024)
    v = morgan_bits_from_smiles("CC(=O)O", cfg)
    assert v is not None
    assert v.shape == (1024,)
    assert v.dtype == np.float32


def test_morgan_bits_invalid_smiles_returns_none():
    """Test invalid smiles case"""
    cfg = MorganFPConfig(radius=2, n_bits=1024)
    v = morgan_bits_from_smiles("not_a_smiles", cfg)
    assert v is None


def test_morgan_bits_is_binary():
    """test if the morgan FP is binary"""
    cfg = MorganFPConfig(radius=2, n_bits=1024)
    v = morgan_bits_from_smiles("c1ccccc1", cfg)
    assert v is not None
    uniq = np.unique(v)
    assert set(uniq.tolist()).issubset({0.0, 1.0})


def test_morgan_feature_names_length_and_uniqueness():
    """test the fp bit lenght and names"""
    cfg = MorganFPConfig(radius=2, n_bits=1024)
    names = morgan_bits_feature_names(cfg)
    assert len(names) == 1024
    assert len(set(names)) == 1024
    assert names[0].startswith("morgan_bits_r2_1024_chiral_")