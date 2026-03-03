from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools


@dataclass(frozen=True)
class AvalonFPConfig:
    """
    Avalon bit vector fingerprint config.
    
    Arguments:
        - number of bits

    """
    n_bits: int = 1024  # common choices: 512, 1024, 2048


def avalon_bits_from_mol(mol: Chem.Mol, cfg: AvalonFPConfig = AvalonFPConfig()) -> np.ndarray:
    """Compute Avalon bit vector from an RDKit Mol"""
    bv = pyAvalonTools.GetAvalonFP(mol, nBits=cfg.n_bits)
    arr = np.zeros((cfg.n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def avalon_bits_from_smiles(smiles: str, cfg: AvalonFPConfig = AvalonFPConfig()) -> Optional[np.ndarray]:
    """Compute Avalon bit vector from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return avalon_bits_from_mol(mol, cfg)


def avalon_bits_feature_names(cfg: AvalonFPConfig = AvalonFPConfig()) -> list[str]:
    return [f"avalon_bits_{cfg.n_bits}_{i}" for i in range(cfg.n_bits)]