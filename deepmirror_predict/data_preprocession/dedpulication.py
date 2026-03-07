from __future__ import annotations

from typing import Iterable, Literal, Any

import numpy as np
import pandas as pd


DedupAgg = Literal["mean", "median", "min", "max"]


def _coerce_prefer_value(value: Any) -> Any:
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "none":
            return None
    return value


def _apply_preference_filter(
    df: pd.DataFrame,
    *,
    key_cols: list[str],
    prefer_col: str,
    prefer_value: Any,
) -> pd.DataFrame:
    if prefer_col not in df.columns:
        raise ValueError(f"Missing prefer_col='{prefer_col}' in df")

    prefer_value = _coerce_prefer_value(prefer_value)

    selected_groups = []

    for _, g in df.groupby(key_cols, dropna=False):
        is_preferred = g[prefer_col].eq(prefer_value)

        # Only filter when both preferred and non-preferred values are present
        if is_preferred.any() and (~is_preferred).any():
            selected_groups.append(g.loc[is_preferred])
        else:
            selected_groups.append(g)

    return pd.concat(selected_groups, axis=0, ignore_index=True)


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
    prefer_col: str | None = None,
    prefer_value: Any | None = None,
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

    if prefer_col is not None:
        if prefer_value is None:
            raise ValueError("prefer_value must be provided when prefer_col is set")
        work = _apply_preference_filter(
            work,
            key_cols=key_cols,
            prefer_col=prefer_col,
            prefer_value=prefer_value,
        )

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
        target_spec = (target_col, "mean")
    elif method == "median":
        target_spec = (target_col, "median")
    elif method == "min":
        target_spec = (target_col, "min")
    elif method == "max":
        target_spec = (target_col, "max")
    else:
        raise ValueError(f"Unsupported method: {method}")

    deduplicated_target_col = f"{target_col}_deduplicated"

    out = (
        work.groupby(key_cols, dropna=False)
        .agg(**rep_aggs, **{deduplicated_target_col: target_spec}, **base_aggs)
        .reset_index()
    )
    return out
