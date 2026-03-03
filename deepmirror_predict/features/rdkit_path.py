from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs


@dataclass(frozen=True)
class RDKitPathFPConfig:
    """
    RDKit path-based vector fingerprint config.

    Arguments:
        - number of bits
        - min path
        - max path
    """
    n_bits: int = 1024
    min_path: int = 1
    max_path: int = 7


def rdkit_path_bits_from_mol(mol: Chem.Mol, cfg: RDKitPathFPConfig = RDKitPathFPConfig()) -> np.ndarray:
    """Compute RDKit path bit vector from an RDKit Mol."""
    bv = Chem.RDKFingerprint(
        mol,
        fpSize=cfg.n_bits,
        minPath=cfg.min_path,
        maxPath=cfg.max_path,
    )
    arr = np.zeros((cfg.n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def rdkit_path_bits_from_smiles(smiles: str, cfg: RDKitPathFPConfig = RDKitPathFPConfig()) -> Optional[np.ndarray]:
    """Compute RDKit path bit vector from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdkit_path_bits_from_mol(mol, cfg)


def rdkit_path_bits_feature_names(cfg: RDKitPathFPConfig = RDKitPathFPConfig()) -> list[str]:
    return [f"rdkit_path_bits_{cfg.n_bits}_p{cfg.min_path}-{cfg.max_path}_{i}" for i in range(cfg.n_bits)]