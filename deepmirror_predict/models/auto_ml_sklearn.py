# src/automl_molops/modeling/autosklearn2_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from auto_sklearn2 import AutoSklearnRegressor


@dataclass(frozen=True)
class AutoSklearn2Config:
    time_limit: int = 120  # seconds
    random_state: int = 0


def fit_predict_autosklearn2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    *,
    cfg: AutoSklearn2Config = AutoSklearn2Config(),
) -> Tuple[np.ndarray, AutoSklearnRegressor]:
    model = AutoSklearnRegressor(time_limit=cfg.time_limit, random_state=cfg.random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    return np.asarray(y_pred, dtype=np.float32), model