# tests/test_avalon.py
import numpy as np

from deepmirror_predict.features.avalon import (
    AvalonFPConfig,
    avalon_bits_from_smiles,
    avalon_bits_feature_names,
)


def test_avalon_bits_valid_shape_dtype():
    """Test to see if the shape and type is correct"""
    cfg = AvalonFPConfig(n_bits=1024)
    v = avalon_bits_from_smiles("CC(=O)O", cfg)
    assert v is not None
    assert v.shape == (1024,)
    assert v.dtype == np.float32


def test_avalon_bits_invalid_smiles_returns_none():
    """Test invalid smiles case"""
    cfg = AvalonFPConfig(n_bits=1024)
    v = avalon_bits_from_smiles("not_a_smiles", cfg)
    assert v is None


def test_avalon_bits_is_binary():
    """Test if the avalon FP is binary"""
    cfg = AvalonFPConfig(n_bits=1024)
    v = avalon_bits_from_smiles("c1ccccc1", cfg)
    assert v is not None
    uniq = np.unique(v)
    assert set(uniq.tolist()).issubset({0.0, 1.0})


def test_avalon_feature_names_length_and_uniqueness():
    """Test the fp bit lenght and names"""
    cfg = AvalonFPConfig(n_bits=1024)
    names = avalon_bits_feature_names(cfg)
    assert len(names) == 1024
    assert len(set(names)) == 1024
    assert names[0].startswith("avalon_bits_1024_")