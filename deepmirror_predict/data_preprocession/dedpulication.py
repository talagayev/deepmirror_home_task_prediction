from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd


DedupAgg = Literal["mean", "median", "min", "max"]


def _first_nonnull(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan


def deduplicate_smiles(
    df: pd.DataFrame,
    *,
    key_cols: Iterable[str] = ("smiles_std",),
    target_col: str = "y",
    method: DedupAgg = "mean",
    keep_cols: Iterable[str] = (),
    drop_missing_keys: bool = True,
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    key_cols = list(key_cols)
    keep_cols = list(keep_cols)

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing key_cols in df: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target_col='{target_col}' in df")

    work = df.copy()

    if drop_missing_keys:
        for c in key_cols:
            work = work[work[c].notna()]

    if drop_missing_target:
        work = work[work[target_col].notna()]

    # Keep metadata columns using "first non-null" per group
    rep_aggs = {
        c: (c, _first_nonnull)
        for c in keep_cols
        if c in work.columns and c not in key_cols
    }

    base_aggs = {
        "n_reps": (target_col, "count"),
        "y_std": (target_col, "std"),
        "y_min": (target_col, "min"),
        "y_max": (target_col, "max"),
    }

    if method == "mean":
        y_spec = (target_col, "mean")
    elif method == "median":
        y_spec = (target_col, "median")
    elif method == "min":
        y_spec = (target_col, "min")
    elif method == "max":
        y_spec = (target_col, "max")
    else:
        raise ValueError(f"Unsupported method: {method}")

    out = (
        work.groupby(key_cols, dropna=False)
        .agg(**rep_aggs, y=y_spec, **base_aggs)
        .reset_index()
    )
    return out