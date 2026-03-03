# src/automl_molops/featurize/chemeleon.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np
from rdkit.Chem import Mol, MolFromSmiles

import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN


@dataclass(frozen=True)
class CheMeleonConfig:
    """
    CheMeleon learned embedding (foundation message-passing) configuration.

    - ckpt_path: if None, uses ~/.chemprop/chemeleon_mp.pt (downloaded if missing)
    - ckpt_url: Zenodo URL for the message-passing weights file
    - device: torch device string ("cpu", "cuda") or None to keep torch default

    Dimensionality:
    - The checkpoint output dim is fixed (typically 2048).
    - If you want a smaller embedding, set reduce_dim to apply a deterministic projection.
    """
    ckpt_path: Optional[Path] = None
    ckpt_url: str = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    device: Optional[Union[str, torch.device]] = None

    reduce_dim: Optional[int] = None  # None keeps native dim; e.g. 512 projects to 512


def _default_ckpt_path() -> Path:
    """Path to store the CheMeleon checkpoint."""
    ckpt_dir = Path.home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir / "chemeleon_mp.pt"


def _safe_torch_load(path: Path):
    """torch.load compatibility across versions."""
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def _deterministic_projection_matrix(in_dim: int, out_dim: int, seed: int = 0) -> np.ndarray:
    """
    Deterministic random Gaussian projection matrix with fixed seed.
    Produces a stable mapping across runs for MLOps reproducibility.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    return W


class CheMeleonFingerprint:
    """
    Generates CheMeleon learned embeddings for SMILES or RDKit Mol objects.

    Output shape: (n_mols, output_dim)
      - output_dim == native_dim (typically 2048) if cfg.reduce_dim is None
      - output_dim == cfg.reduce_dim if set (via deterministic projection)
    """

    def __init__(self, cfg: CheMeleonConfig = CheMeleonConfig()):
        self.cfg = cfg

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()

        mp_path = cfg.ckpt_path if cfg.ckpt_path is not None else _default_ckpt_path()
        if not mp_path.exists():
            urlretrieve(cfg.ckpt_url, mp_path)

        chemeleon_mp = _safe_torch_load(mp_path)
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])

        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        self.model.eval()

        if cfg.device is not None:
            self.model.to(device=cfg.device)

        self.native_dim: int = int(mp.output_dim)

        # Optional deterministic projection for reduce_dim
        if cfg.reduce_dim is None:
            self.output_dim: int = self.native_dim
            self._proj: Optional[np.ndarray] = None
        else:
            if cfg.reduce_dim <= 0 or cfg.reduce_dim > self.native_dim:
                raise ValueError(f"reduce_dim must be in [1, {self.native_dim}] (got {cfg.reduce_dim})")
            self.output_dim = int(cfg.reduce_dim)
            self._proj = _deterministic_projection_matrix(self.native_dim, self.output_dim, seed=0)

    def __call__(self, molecules: list[Union[str, Mol]]) -> np.ndarray:
        mols: list[Mol] = []
        for m in molecules:
            if isinstance(m, str):
                rm = MolFromSmiles(m)
                if rm is None:
                    raise ValueError(f"Invalid SMILES passed to CheMeleonFingerprint: {m!r}")
                mols.append(rm)
            else:
                mols.append(m)

        graphs = [self.featurizer(m) for m in mols]
        bmg = BatchMolGraph(graphs)

        device = next(self.model.parameters()).device
        bmg.to(device=device)

        with torch.no_grad():
            fp = self.model.fingerprint(bmg).detach().cpu().numpy().astype(np.float32, copy=False)

        if self._proj is not None:
            fp = fp @ self._proj  # (n, native_dim) -> (n, output_dim)

        return fp


def chemeleon_feature_names(embedding_dim: int) -> list[str]:
    return [f"chemeleon_{i}" for i in range(embedding_dim)]


def chemeleon_batch_from_smiles(
    smiles_list: Iterable[str],
    *,
    cfg: CheMeleonConfig = CheMeleonConfig(),
) -> Tuple[np.ndarray, list[str]]:
    
    smiles_list = list(smiles_list)
    featurizer = CheMeleonFingerprint(cfg)
    X = featurizer(smiles_list)
    names = chemeleon_feature_names(featurizer.output_dim)

    
    return X, names