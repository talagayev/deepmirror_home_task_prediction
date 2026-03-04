# deepmirror_predict/tests/models/test_chemprop_regressor.py
import numpy as np
import pytest

# If someone runs tests without the optional deps installed, skip cleanly.
pytest.importorskip("chemprop")
pytest.importorskip("lightning")
pytest.importorskip("torch")

from deepmirror_predict.models.chemprop_regression import ChempropConfig, ChempropRegressor


def _make_smiles_regression_data():
    # Small, fast-to-train toy set (valid RDKit SMILES)
    smiles = [
        "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
        "CCO", "CCCO", "CCCCO", "CCN", "CCCN",
        "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
        "CC(=O)O", "CC(=O)N", "CCS",
        "COC", "CCCl", "CCBr", "CCF",
    ]
    # Simple target: length + a tiny deterministic offset
    y = np.array([len(s) + (i % 3) * 0.1 for i, s in enumerate(smiles)], dtype=np.float32)
    return np.array(smiles, dtype=object), y


def _tiny_cfg(tmp_path):
    # Keep training extremely small/fast and CPU-only for CI.
    return ChempropConfig(
        message_hidden_dim=64,
        message_depth=2,
        message_dropout=0.0,
        ffn_hidden_dim=64,
        ffn_layers=1,
        ffn_dropout=0.0,
        batch_norm=False,
        warmup_epochs=1,
        init_lr=1e-4,
        max_lr=3e-4,
        final_lr=1e-4,
        max_epochs=2,
        batch_size=8,
        num_workers=0,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        early_stopping=False,  # keep deterministic & fast
        checkpoint_dir=str(tmp_path),
    )


def test_predict_before_fit_raises(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    reg = ChempropRegressor(cfg=cfg, random_state=0)
    with pytest.raises(RuntimeError):
        reg.predict(["CCO"])


def test_fit_requires_eval_set(tmp_path):
    X, y = _make_smiles_regression_data()
    cfg = _tiny_cfg(tmp_path)
    reg = ChempropRegressor(cfg=cfg, random_state=0)

    with pytest.raises(ValueError):
        reg.fit(X[:10], y[:10])  # missing eval_set


def test_fit_rejects_multi_target_y(tmp_path):
    X, y = _make_smiles_regression_data()
    y2 = np.stack([y, y], axis=1)  # (n, 2) invalid

    cfg = _tiny_cfg(tmp_path)
    reg = ChempropRegressor(cfg=cfg, random_state=0)

    with pytest.raises(ValueError):
        reg.fit(X[:12], y2[:12], eval_set=[(X[12:16], y[12:16])])


def test_fit_and_predict_outputs_finite_and_correct_shape(tmp_path):
    X, y = _make_smiles_regression_data()

    # fixed split
    X_train, y_train = X[:14], y[:14]
    X_val, y_val = X[14:17], y[14:17]
    X_test, y_test = X[17:], y[17:]

    cfg = _tiny_cfg(tmp_path)
    reg = ChempropRegressor(cfg=cfg, random_state=123)

    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    assert reg.model_ is not None

    # checkpoint may or may not be "best_model_path" depending on logging,
    # but if it exists it must be a real file.
    if reg.best_ckpt_path_:
        import os

        assert os.path.exists(reg.best_ckpt_path_)

    y_pred = reg.predict(X_test)

    assert y_pred.shape == (len(y_test),)
    assert y_pred.dtype == np.float32
    assert np.isfinite(y_pred).all()