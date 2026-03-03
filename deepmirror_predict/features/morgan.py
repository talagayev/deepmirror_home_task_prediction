from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors


@dataclass(frozen=True)
class MorganFPConfig:
    """
    Morgan bit vector fingerprint config.

    Arguments:
        - radius: Morgan FP radius
        - n_bits: number of bits
    """
    radius: int = 3
    n_bits: int = 1024


def morgan_bits_from_mol(mol: Chem.Mol, cfg: MorganFPConfig = MorganFPConfig()) -> np.ndarray:
    """Compute Morgan bit vector from an RDKit Mol."""
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        cfg.radius,
        nBits=cfg.n_bits,
        useChirality=True,  # keep default behavior: include chirality
    )
    arr = np.zeros((cfg.n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def morgan_bits_from_smiles(smiles: str, cfg: MorganFPConfig = MorganFPConfig()) -> Optional[np.ndarray]:
    """Compute Morgan bit vector from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_bits_from_mol(mol, cfg)


def morgan_bits_feature_names(cfg: MorganFPConfig = MorganFPConfig()) -> list[str]:
    return [f"morgan_bits_r{cfg.radius}_{cfg.n_bits}_chiral_{i}" for i in range(cfg.n_bits)]