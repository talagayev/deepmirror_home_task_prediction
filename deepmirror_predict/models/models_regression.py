from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, RegressorMixin

from chemprop import data, featurizers, models, nn

from deepmirror_predict.analysis.applicability_domain import parse_feature_token
from deepmirror_predict.features.avalon import AvalonFPConfig, avalon_bits_from_smiles
from deepmirror_predict.features.mordred import mordred2d_batch_from_smiles, prune_mordred_matrix
from deepmirror_predict.features.morgan import MorganFPConfig, morgan_bits_from_smiles
from deepmirror_predict.features.rdkit_path import RDKitPathFPConfig, rdkit_path_bits_from_smiles


@dataclass(frozen=True)
class ChempropConfig:
    # Message passing (BondMessagePassing)
    message_hidden_dim: int = 300
    message_depth: int = 3
    message_dropout: float = 0.0

    # Predictor (RegressionFFN)
    ffn_hidden_dim: int = 300
    ffn_layers: int = 1
    ffn_dropout: float = 0.0
    ffn_activation: str = "relu"

    # MPNN wrapper
    batch_norm: bool = True

    # Optimizer / scheduler
    warmup_epochs: int = 2
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4

    # Training
    max_epochs: int = 20
    batch_size: int = 64
    num_workers: int = 0

    # Lightning
    accelerator: str = "auto"
    devices: int = 1
    enable_progress_bar: bool = False

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # Checkpointing
    checkpoint_dir: Optional[str] = "checkpoints"
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"

    # Foundation initialization
    from_foundation: Optional[str] = None  # None | "chemeleon"
    foundation_ckpt_path: Optional[str] = None
    foundation_ckpt_url: str = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

    # Extra datapoint descriptors generated from feature tokens
    # Examples:
    #   ("morgan_r2_b1024",)
    #   ("avalon_b1024", "rdkit_path_min1_max7_b1024")
    #   ("morgan_r3_b2048", "avalon_b1024", "mordred")
    extra_descriptor_tokens: tuple[str, ...] = ()

    # Mordred preprocessing for extra descriptors
    mordred_max_nan_frac: float = 0.2
    mordred_drop_constant: bool = True


def _as_smiles_list(X: Sequence[object]) -> list[str]:
    return [str(x) for x in X]


def _as_targets(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"ChempropRegressor expects single-target regression (got shape {y.shape})")
    return y


def _default_chemeleon_ckpt_path() -> Path:
    ckpt_dir = Path.home() / ".chemprop"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / "chemeleon_mp.pt"


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def _load_chemeleon_message_passing(cfg: ChempropConfig):
    ckpt_path = Path(cfg.foundation_ckpt_path) if cfg.foundation_ckpt_path else _default_chemeleon_ckpt_path()
    if not ckpt_path.exists():
        urlretrieve(cfg.foundation_ckpt_url, ckpt_path)

    state = _safe_torch_load(ckpt_path)
    mp = nn.BondMessagePassing(**state["hyper_parameters"])
    mp.load_state_dict(state["state_dict"])
    return mp


def _stack_features(rows: list[np.ndarray], name: str) -> np.ndarray:
    if not rows:
        raise ValueError(f"No descriptor rows produced for {name}")
    return np.vstack(rows).astype(np.float32, copy=False)


def _build_extra_descriptors_from_tokens(
    smiles_list: Sequence[str],
    cfg: ChempropConfig,
) -> Optional[np.ndarray]:
    if not cfg.extra_descriptor_tokens:
        return None

    blocks: list[np.ndarray] = []

    for token in cfg.extra_descriptor_tokens:
        spec = parse_feature_token(token)

        if spec.kind == "morgan":
            fp_cfg = MorganFPConfig(
                radius=spec.params["radius"],
                n_bits=spec.params["n_bits"],
            )
            rows = [morgan_bits_from_smiles(s, fp_cfg) for s in smiles_list]
            if any(v is None for v in rows):
                bad = [s for s, v in zip(smiles_list, rows) if v is None][:5]
                raise ValueError(f"Invalid SMILES during Morgan descriptor generation: {bad}")
            Xi = _stack_features(rows, token)

        elif spec.kind == "avalon":
            fp_cfg = AvalonFPConfig(n_bits=spec.params["n_bits"])
            rows = [avalon_bits_from_smiles(s, fp_cfg) for s in smiles_list]
            if any(v is None for v in rows):
                bad = [s for s, v in zip(smiles_list, rows) if v is None][:5]
                raise ValueError(f"Invalid SMILES during Avalon descriptor generation: {bad}")
            Xi = _stack_features(rows, token)

        elif spec.kind == "rdkit_path":
            fp_cfg = RDKitPathFPConfig(
                min_path=spec.params["min_path"],
                max_path=spec.params["max_path"],
                n_bits=spec.params["n_bits"],
            )
            rows = [rdkit_path_bits_from_smiles(s, fp_cfg) for s in smiles_list]
            if any(v is None for v in rows):
                bad = [s for s, v in zip(smiles_list, rows) if v is None][:5]
                raise ValueError(f"Invalid SMILES during RDKit-path descriptor generation: {bad}")
            Xi = _stack_features(rows, token)

        elif spec.kind == "mordred":
            Xi, names = mordred2d_batch_from_smiles(smiles_list)
            Xi, _ = prune_mordred_matrix(
                Xi,
                names,
                max_nan_frac=cfg.mordred_max_nan_frac,
                drop_constant=cfg.mordred_drop_constant,
            )
            Xi = Xi.astype(np.float32, copy=False)

        elif spec.kind == "chemeleon":
            raise ValueError(
                "CheMeleon cannot be used as an extra Chemprop descriptor token. "
                "Use it as the Chemprop backbone via from_foundation='chemeleon'."
            )

        else:
            raise ValueError(f"Unsupported extra descriptor token for Chemprop: {token}")

        blocks.append(Xi)

    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


class ChempropRegressor(RegressorMixin, BaseEstimator):
    """
    sklearn-compatible wrapper around Chemprop v2 MPNN.

    Supports:
    - vanilla Chemprop
    - Chemprop initialized from CheMeleon foundation weights
    - extra datapoint descriptors generated from feature tokens
    - Chemprop + CheMeleon + extra descriptors

    fit(X_smiles, y, eval_set=[(X_val_smiles, y_val)]) trains with Lightning.
    predict(X_smiles) returns float32 vector.
    """

    def __init__(self, cfg: ChempropConfig = ChempropConfig(), *, random_state: int = 0):
        self.cfg = cfg
        self.random_state = int(random_state)
        self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.model_: Optional[models.MPNN] = None
        self.best_ckpt_path_: Optional[str] = None

    def __sklearn_is_fitted__(self) -> bool:
        return bool(getattr(self, "is_fitted_", False))

    def _build_model(self, scaler, x_d_scaler=None, x_d_dim: int = 0) -> models.MPNN:
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        if self.cfg.from_foundation is None:
            mp = nn.BondMessagePassing(
                d_h=self.cfg.message_hidden_dim,
                depth=self.cfg.message_depth,
                dropout=self.cfg.message_dropout,
            )
        else:
            foundation = self.cfg.from_foundation.lower()
            if foundation != "chemeleon":
                raise ValueError(
                    f"Unsupported foundation model '{self.cfg.from_foundation}'. "
                    "Currently only 'chemeleon' is supported."
                )
            mp = _load_chemeleon_message_passing(self.cfg)

        agg = nn.MeanAggregation()

        ffn = nn.RegressionFFN(
            input_dim=int(mp.output_dim) + int(x_d_dim),
            hidden_dim=self.cfg.ffn_hidden_dim,
            n_layers=self.cfg.ffn_layers,
            dropout=self.cfg.ffn_dropout,
            activation=self.cfg.ffn_activation,
            output_transform=output_transform,
        )

        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

        kwargs = {}
        if x_d_scaler is not None:
            kwargs["X_d_transform"] = nn.ScaleTransform.from_standard_scaler(x_d_scaler)

        return models.MPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
            batch_norm=self.cfg.batch_norm,
            metrics=metric_list,
            warmup_epochs=self.cfg.warmup_epochs,
            init_lr=self.cfg.init_lr,
            max_lr=self.cfg.max_lr,
            final_lr=self.cfg.final_lr,
            **kwargs,
        )

    def _make_dataset(
        self,
        X: Sequence[object],
        y: Optional[np.ndarray] = None,
    ):
        smiles = _as_smiles_list(X)
        x_d = _build_extra_descriptors_from_tokens(smiles, self.cfg)

        datapoints = []
        if y is None:
            if x_d is None:
                datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]
            else:
                datapoints = [data.MoleculeDatapoint.from_smi(smi, x_d=xd) for smi, xd in zip(smiles, x_d)]
        else:
            y = _as_targets(y)
            if x_d is None:
                datapoints = [data.MoleculeDatapoint.from_smi(smi, yi) for smi, yi in zip(smiles, y)]
            else:
                datapoints = [
                    data.MoleculeDatapoint.from_smi(smi, yi, x_d=xd)
                    for smi, yi, xd in zip(smiles, y, x_d)
                ]

        dset = data.MoleculeDataset(datapoints, featurizer=self._featurizer)
        return dset, x_d

    def fit(
        self,
        X: Sequence[object],
        y: np.ndarray,
        *,
        eval_set: Optional[Sequence[Tuple[Sequence[object], np.ndarray]]] = None,
    ) -> "ChempropRegressor":
        pl.seed_everything(self.random_state, workers=True)

        if not eval_set:
            raise ValueError("ChempropRegressor.fit requires eval_set=[(X_val, y_val)] for val_loss monitoring.")

        X_val_raw, y_val_raw = eval_set[0]

        train_dset, Xd_train = self._make_dataset(X, y)
        val_dset, Xd_val = self._make_dataset(X_val_raw, y_val_raw)

        target_scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(target_scaler)

        x_d_scaler = None
        x_d_dim = 0
        if Xd_train is not None:
            x_d_dim = int(Xd_train.shape[1])
            x_d_scaler = train_dset.normalize_inputs("X_d")
            val_dset.normalize_inputs("X_d", x_d_scaler)

        train_loader = data.build_dataloader(
            train_dset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            seed=self.random_state,
        )
        val_loader = data.build_dataloader(
            val_dset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

        mpnn = self._build_model(target_scaler, x_d_scaler=x_d_scaler, x_d_dim=x_d_dim)

        callbacks = []
        ckpt_dir = Path(self.cfg.checkpoint_dir or "checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpointing = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch}-{val_loss:.4f}",
            monitor=self.cfg.checkpoint_monitor,
            mode=self.cfg.checkpoint_mode,
            save_last=True,
        )
        callbacks.append(checkpointing)

        if self.cfg.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.cfg.checkpoint_monitor,
                    mode=self.cfg.checkpoint_mode,
                    patience=self.cfg.early_stopping_patience,
                    min_delta=self.cfg.early_stopping_min_delta,
                )
            )

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=self.cfg.enable_progress_bar,
            accelerator=self.cfg.accelerator,
            devices=self.cfg.devices,
            max_epochs=self.cfg.max_epochs,
            callbacks=callbacks,
        )

        trainer.fit(mpnn, train_loader, val_loader)

        self.best_ckpt_path_ = checkpointing.best_model_path or None
        if self.best_ckpt_path_:
            try:
                self.model_ = models.MPNN.load_from_checkpoint(self.best_ckpt_path_)
            except Exception:
                self.model_ = mpnn
        else:
            self.model_ = mpnn

        self.model_.eval()
        self.is_fitted_ = True
        return self

    def predict(self, X: Sequence[object]) -> np.ndarray:
        if self.model_ is None or not getattr(self, "is_fitted_", False):
            raise RuntimeError("ChempropRegressor.predict called before fit().")

        pred_dset, _ = self._make_dataset(X, None)
        pred_loader = data.build_dataloader(
            pred_dset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=self.cfg.enable_progress_bar,
                accelerator=self.cfg.accelerator,
                devices=self.cfg.devices,
            )
            preds = trainer.predict(self.model_, pred_loader)

        y = np.concatenate([p.detach().cpu().numpy() for p in preds], axis=0)
        return y.reshape(-1).astype(np.float32, copy=False)
