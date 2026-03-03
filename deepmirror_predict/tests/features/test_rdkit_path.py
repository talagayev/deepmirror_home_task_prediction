# tests/test_rdkit_path.py
import numpy as np

from deepmirror_predict.features.rdkit_path import (
    RDKitPathFPConfig,
    rdkit_path_bits_from_smiles,
    rdkit_path_bits_feature_names,
)


def test_rdkit_path_bits_valid_shape_dtype():
    """Test to see if the shape and type is correct"""
    cfg = RDKitPathFPConfig(n_bits=1024, min_path=1, max_path=7)
    v = rdkit_path_bits_from_smiles("CC(=O)O", cfg)
    assert v is not None
    assert v.shape == (1024,)
    assert v.dtype == np.float32


def test_rdkit_path_bits_invalid_smiles_returns_none():
    """Test invalid smiles case"""
    cfg = RDKitPathFPConfig(n_bits=1024, min_path=1, max_path=7)
    v = rdkit_path_bits_from_smiles("not_a_smiles", cfg)
    assert v is None


def test_rdkit_path_bits_is_binary():
    """Test if the rdkit path FP is binary"""
    cfg = RDKitPathFPConfig(n_bits=1024, min_path=1, max_path=7)
    v = rdkit_path_bits_from_smiles("c1ccccc1", cfg)
    assert v is not None
    uniq = np.unique(v)
    assert set(uniq.tolist()).issubset({0.0, 1.0})


def test_rdkit_path_feature_names_length_and_uniqueness():
    """Test the fp bit lenght and names"""
    cfg = RDKitPathFPConfig(n_bits=1024, min_path=2, max_path=6)
    names = rdkit_path_bits_feature_names(cfg)
    assert len(names) == 1024
    assert len(set(names)) == 1024
    assert names[0].startswith("rdkit_path_bits_1024_p2-6_")