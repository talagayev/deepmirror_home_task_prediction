# src/automl_molops/featurize/mordred2d.py
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors


# Build calculator for 2D, due to other features being 2D
_CALC = Calculator(descriptors, ignore_3D=True)


def mordred2d_feature_names() -> list[str]:
    """get the names of the descriptors"""
    return [str(d) for d in _CALC.descriptors]


def _to_float(x) -> float:
    """helper function to convert to float"""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def mordred2d_from_mol(mol: Chem.Mol) -> np.ndarray:
    """
    Compute Mordred 2D descriptors for a single RDKit Mol.
    """
    res = _CALC(mol)
    vals = [_to_float(res[d]) for d in _CALC.descriptors]
    return np.asarray(vals, dtype=np.float32)


def mordred2d_from_smiles(smiles: str) -> Optional[np.ndarray]:
    """Compute Mordred 2D descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mordred2d_from_mol(mol)


def mordred2d_batch_from_smiles(smiles_list: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    """
    Batch compute Mordred 2D descriptors.
    """
    smiles_list = list(smiles_list)
    names = mordred2d_feature_names()
    rows = []

    for s in smiles_list:
        v = mordred2d_from_smiles(s)
        if v is None:
            raise ValueError(f"Invalid SMILES encountered in processed data: {s!r}")
        rows.append(v)

    X = np.vstack(rows).astype(np.float32, copy=False)
    return X, names


def prune_mordred_matrix(
    X: np.ndarray,
    names: list[str],
    *,
    max_nan_frac: float = 0.2,
    drop_constant: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """
    Prune Mordred descriptor matrix:
      - drop columns with NaN fraction > max_nan_frac
      - optionally drop constant (zero-variance) columns (ignoring NaNs)

    Returns pruned (X2, names2).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if X.shape[1] != len(names):
        raise ValueError("names length must match X columns")

    nan_frac = np.mean(np.isnan(X), axis=0)
    keep = nan_frac <= max_nan_frac

    Xk = X[:, keep]
    names_k = [nm for nm, k in zip(names, keep) if k]

    if drop_constant and Xk.shape[1] > 0:
        v = np.nanvar(Xk, axis=0)
        keep2 = np.isfinite(v) & (v > 0.0)
        Xk = Xk[:, keep2]
        names_k = [nm for nm, k in zip(names_k, keep2) if k]

    return Xk.astype(np.float32, copy=False), names_k