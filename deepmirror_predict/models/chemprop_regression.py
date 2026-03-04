from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, RegressorMixin

from chemprop import data, featurizers, models, nn


@dataclass(frozen=True)
class ChempropConfig:
    # Message passing (BondMessagePassing)
    message_hidden_dim: int = 300  # d_h
    message_depth: int = 3         # depth
    message_dropout: float = 0.0   # dropout

    # Predictor (RegressionFFN)
    ffn_hidden_dim: int = 300
    ffn_layers: int = 1
    ffn_dropout: float = 0.0
    ffn_activation: str = "relu"

    # MPNN wrapper
    batch_norm: bool = True

    # Optimizer / scheduler (Noam-like)
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

    # Early stopping (monitor val_loss)
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # Checkpointing (best checkpoint used for inference)
    checkpoint_dir: Optional[str] = "checkpoints"
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"


def _as_smiles_list(X: Sequence[object]) -> list[str]:
    return [str(x) for x in X]


def _as_targets(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"ChempropRegressor expects single-target regression (got shape {y.shape})")
    return y


class ChempropRegressor(RegressorMixin, BaseEstimator):
    """
    sklearn-compatible wrapper around Chemprop v2 MPNN.

    - fit(X_smiles, y, eval_set=[(X_val_smiles, y_val)]) trains with Lightning
    - predict(X_smiles) returns float32 vector
    """

    def __init__(self, cfg: ChempropConfig = ChempropConfig(), *, random_state: int = 0):
        self.cfg = cfg
        self.random_state = int(random_state)

        self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.model_: Optional[models.MPNN] = None
        self.best_ckpt_path_: Optional[str] = None
        # sklearn "fitted" flag is created in fit()

    def __sklearn_is_fitted__(self) -> bool:
        return bool(getattr(self, "is_fitted_", False))

    def _build_model(self, scaler) -> models.MPNN:
        # UnscaleTransform ensures predictions are back in original target units
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

        mp = nn.BondMessagePassing(
            d_h=self.cfg.message_hidden_dim,
            depth=self.cfg.message_depth,
            dropout=self.cfg.message_dropout,
        )

        agg = nn.MeanAggregation()

        ffn = nn.RegressionFFN(
            input_dim=mp.output_dim,
            hidden_dim=self.cfg.ffn_hidden_dim,
            n_layers=self.cfg.ffn_layers,
            dropout=self.cfg.ffn_dropout,
            activation=self.cfg.ffn_activation,
            output_transform=output_transform,
        )

        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

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
        )

    def fit(
        self,
        X: Sequence[object],
        y: np.ndarray,
        *,
        eval_set: Optional[Sequence[Tuple[Sequence[object], np.ndarray]]] = None,
    ) -> "ChempropRegressor":
        pl.seed_everything(self.random_state, workers=True)

        X = _as_smiles_list(X)
        y = _as_targets(y)

        # Require a validation set for checkpointing/early stopping
        if not eval_set:
            raise ValueError(
                "ChempropRegressor.fit requires eval_set=[(X_val, y_val)] for val_loss monitoring."
            )

        X_val_raw, y_val_raw = eval_set[0]
        X_val = _as_smiles_list(X_val_raw)
        y_val = _as_targets(y_val_raw)

        train_data = [data.MoleculeDatapoint.from_smi(smi, yi) for smi, yi in zip(X, y)]
        val_data = [data.MoleculeDatapoint.from_smi(smi, yi) for smi, yi in zip(X_val, y_val)]

        train_dset = data.MoleculeDataset(train_data, featurizer=self._featurizer)
        scaler = train_dset.normalize_targets()

        val_dset = data.MoleculeDataset(val_data, featurizer=self._featurizer)
        val_dset.normalize_targets(scaler)

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

        mpnn = self._build_model(scaler)

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
            self.model_ = models.MPNN.load_from_checkpoint(self.best_ckpt_path_)
        else:
            # fallback: use last trained in-memory model
            self.model_ = mpnn

        self.model_.eval()
        self.is_fitted_ = True
        return self

    def predict(self, X: Sequence[object]) -> np.ndarray:
        if self.model_ is None or not getattr(self, "is_fitted_", False):
            raise RuntimeError("ChempropRegressor.predict called before fit().")

        X = _as_smiles_list(X)
        pred_data = [data.MoleculeDatapoint.from_smi(smi) for smi in X]

        pred_dset = data.MoleculeDataset(pred_data, featurizer=self._featurizer)
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
        y = y.reshape(-1).astype(np.float32, copy=False)
        return y