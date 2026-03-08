from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

from deepmirror_predict.features.avalon import AvalonFPConfig, avalon_bits_from_smiles
from deepmirror_predict.features.chemeleon import CheMeleonConfig, chemeleon_batch_from_smiles
from deepmirror_predict.features.mordred import mordred2d_batch_from_smiles, prune_mordred_matrix
from deepmirror_predict.features.morgan import MorganFPConfig, morgan_bits_from_smiles
from deepmirror_predict.features.rdkit_path import RDKitPathFPConfig, rdkit_path_bits_from_smiles


EmbeddingMethod = Literal["pca", "umap"]


@dataclass(frozen=True)
class FeatureSpec:
    raw_name: str
    kind: str
    family: Literal["binary", "dense"]
    params: dict


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    family: Literal["binary", "dense"]
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class DualFeatureBlock:
    name: str
    family: Literal["binary", "dense"]
    train_all: np.ndarray
    train_finetune: np.ndarray
    train_other: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class ADResult:
    feature_set_name: str
    train_nn_similarity: np.ndarray
    test_max_similarity: np.ndarray
    test_mean_topk_similarity: np.ndarray
    embedding_coords: np.ndarray
    n_train: int
    n_test: int


@dataclass(frozen=True)
class DualADResult:
    feature_set_name: str
    train_all_nn_similarity: np.ndarray
    train_finetune_nn_similarity: np.ndarray
    test_max_similarity_to_train_all: np.ndarray
    test_mean_topk_similarity_to_train_all: np.ndarray
    test_max_similarity_to_train_finetune: np.ndarray
    test_mean_topk_similarity_to_train_finetune: np.ndarray
    embedding_coords: np.ndarray
    n_train_other: int
    n_train_finetune: int
    n_test: int


def _tanimoto_similarity_matrix(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    Xa = Xa.astype(np.float32, copy=False)
    Xb = Xb.astype(np.float32, copy=False)

    inter = Xa @ Xb.T
    a_sum = Xa.sum(axis=1, keepdims=True)
    b_sum = Xb.sum(axis=1, keepdims=True).T
    union = a_sum + b_sum - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def _ensure_list(values, default: list[int]) -> list[int]:
    if values is None:
        return default
    if isinstance(values, int):
        return [int(values)]
    return sorted({int(v) for v in values}) or default


def expand_feature_sets(
    feature_sets: list[str],
    *,
    morgan_radius: list[int] | int | None = None,
    morgan_bits: list[int] | int | None = None,
    avalon_bits: list[int] | int | None = None,
    rdkit_path_min: list[int] | int | None = None,
    rdkit_path_max: list[int] | int | None = None,
    rdkit_path_bits: list[int] | int | None = None,
) -> list[str]:
    allowed = {"morgan", "avalon", "rdkit_path", "mordred", "chemeleon"}

    morgan_radius = _ensure_list(morgan_radius, [3])
    morgan_bits = _ensure_list(morgan_bits, [1024])
    avalon_bits = _ensure_list(avalon_bits, [1024])
    rdkit_path_min = _ensure_list(rdkit_path_min, [1])
    rdkit_path_max = _ensure_list(rdkit_path_max, [7])
    rdkit_path_bits = _ensure_list(rdkit_path_bits, [1024])

    expanded_feature_sets: list[str] = []

    for feature_set in feature_sets:
        parts = [p.strip() for p in feature_set.split("+") if p.strip()]
        if not parts:
            continue

        for p in parts:
            if p not in allowed:
                raise ValueError(
                    f"Unsupported feature name '{p}' in feature_set '{feature_set}'. "
                    f"Allowed: {sorted(allowed)}"
                )

        expanded_parts: list[list[str]] = []
        for p in parts:
            if p == "morgan":
                variants = [f"morgan_r{r}_b{b}" for r in morgan_radius for b in morgan_bits]
            elif p == "avalon":
                variants = [f"avalon_b{b}" for b in avalon_bits]
            elif p == "rdkit_path":
                variants = []
                for mn in rdkit_path_min:
                    for mx in rdkit_path_max:
                        if mn <= mx:
                            for b in rdkit_path_bits:
                                variants.append(f"rdkit_path_min{mn}_max{mx}_b{b}")
            else:
                variants = [p]
            expanded_parts.append(variants)

        combos = [""]
        for variants in expanded_parts:
            new_combos: list[str] = []
            for prefix in combos:
                for v in variants:
                    new_combos.append(v if prefix == "" else f"{prefix}+{v}")
            combos = new_combos

        expanded_feature_sets.extend(combos)

    seen: set[str] = set()
    deduped: list[str] = []
    for fs in expanded_feature_sets:
        if fs not in seen:
            seen.add(fs)
            deduped.append(fs)
    return deduped


def parse_feature_token(token: str) -> FeatureSpec:
    token = token.strip()
    if not token:
        raise ValueError("Empty feature token")

    if token == "mordred":
        return FeatureSpec(raw_name=token, kind="mordred", family="dense", params={})

    if token == "chemeleon":
        return FeatureSpec(raw_name=token, kind="chemeleon", family="dense", params={})

    if token == "morgan":
        return FeatureSpec(
            raw_name=token,
            kind="morgan",
            family="binary",
            params={"radius": 3, "n_bits": 1024},
        )

    if token.startswith("morgan_r"):
        parts = token.split("_")
        radius = None
        n_bits = None
        for p in parts[1:]:
            if p.startswith("r"):
                radius = int(p[1:])
            elif p.startswith("b"):
                n_bits = int(p[1:])
        if radius is None or n_bits is None:
            raise ValueError(f"Invalid Morgan token '{token}'. Expected morgan_r2_b2048")
        return FeatureSpec(
            raw_name=token,
            kind="morgan",
            family="binary",
            params={"radius": radius, "n_bits": n_bits},
        )

    if token == "avalon":
        return FeatureSpec(
            raw_name=token,
            kind="avalon",
            family="binary",
            params={"n_bits": 1024},
        )

    if token.startswith("avalon_b"):
        n_bits = int(token.split("_b")[1])
        return FeatureSpec(
            raw_name=token,
            kind="avalon",
            family="binary",
            params={"n_bits": n_bits},
        )

    if token == "rdkit_path":
        return FeatureSpec(
            raw_name=token,
            kind="rdkit_path",
            family="binary",
            params={"min_path": 1, "max_path": 7, "n_bits": 1024},
        )

    if token.startswith("rdkit_path_min"):
        parts = token.split("_")
        min_path = None
        max_path = None
        n_bits = None
        for p in parts[2:]:
            if p.startswith("min"):
                min_path = int(p[3:])
            elif p.startswith("max"):
                max_path = int(p[3:])
            elif p.startswith("b"):
                n_bits = int(p[1:])
        if min_path is None or max_path is None or n_bits is None:
            raise ValueError(
                f"Invalid RDKit path token '{token}'. Expected rdkit_path_min1_max6_b2048"
            )
        return FeatureSpec(
            raw_name=token,
            kind="rdkit_path",
            family="binary",
            params={"min_path": min_path, "max_path": max_path, "n_bits": n_bits},
        )

    raise ValueError(
        f"Unsupported feature token '{token}'. "
        "Allowed base kinds: morgan, avalon, rdkit_path, mordred, chemeleon"
    )


def parse_feature_set(feature_set: str) -> list[FeatureSpec]:
    tokens = [x.strip() for x in feature_set.split("+") if x.strip()]
    if not tokens:
        raise ValueError("feature_set must contain at least one feature token")
    return [parse_feature_token(tok) for tok in tokens]


def _binary_block(smiles: list[str], spec: FeatureSpec) -> np.ndarray:
    rows = []

    if spec.kind == "morgan":
        cfg = MorganFPConfig(radius=spec.params["radius"], n_bits=spec.params["n_bits"])
        fn = lambda s: morgan_bits_from_smiles(s, cfg)
    elif spec.kind == "avalon":
        cfg = AvalonFPConfig(n_bits=spec.params["n_bits"])
        fn = lambda s: avalon_bits_from_smiles(s, cfg)
    elif spec.kind == "rdkit_path":
        cfg = RDKitPathFPConfig(
            min_path=spec.params["min_path"],
            max_path=spec.params["max_path"],
            n_bits=spec.params["n_bits"],
        )
        fn = lambda s: rdkit_path_bits_from_smiles(s, cfg)
    else:
        raise ValueError(f"Unsupported binary feature kind: {spec.kind}")

    for s in smiles:
        v = fn(s)
        if v is None:
            raise ValueError(f"Invalid SMILES encountered: {s!r}")
        rows.append(v)

    return np.vstack(rows).astype(np.float32, copy=False)


def _mordred_block(train_smiles: list[str], test_smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    all_smiles = list(train_smiles) + list(test_smiles)
    X_all, names = mordred2d_batch_from_smiles(all_smiles)
    X_all, names = prune_mordred_matrix(X_all, names, max_nan_frac=0.2, drop_constant=True)
    X_all = SimpleImputer(strategy="median").fit_transform(X_all).astype(np.float32, copy=False)
    n_train = len(train_smiles)
    return X_all[:n_train], X_all[n_train:]


def _chemeleon_block(train_smiles: list[str], test_smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    all_smiles = list(train_smiles) + list(test_smiles)
    X_all, _ = chemeleon_batch_from_smiles(all_smiles, cfg=CheMeleonConfig())
    X_all = X_all.astype(np.float32, copy=False)
    n_train = len(train_smiles)
    return X_all[:n_train], X_all[n_train:]


def build_feature_blocks(
    train_smiles: list[str],
    test_smiles: list[str],
    feature_set: str,
) -> list[FeatureBlock]:
    blocks: list[FeatureBlock] = []

    for spec in parse_feature_set(feature_set):
        if spec.family == "binary":
            X_train = _binary_block(train_smiles, spec)
            X_test = _binary_block(test_smiles, spec)
            blocks.append(FeatureBlock(name=spec.raw_name, family="binary", train=X_train, test=X_test))
        elif spec.kind == "mordred":
            X_train, X_test = _mordred_block(train_smiles, test_smiles)
            blocks.append(FeatureBlock(name=spec.raw_name, family="dense", train=X_train, test=X_test))
        elif spec.kind == "chemeleon":
            X_train, X_test = _chemeleon_block(train_smiles, test_smiles)
            blocks.append(FeatureBlock(name=spec.raw_name, family="dense", train=X_train, test=X_test))
        else:
            raise ValueError(f"Unsupported feature kind: {spec.kind}")

    return blocks


def build_dual_feature_blocks(
    train_other_smiles: list[str],
    train_finetune_smiles: list[str],
    test_smiles: list[str],
    feature_set: str,
) -> list[DualFeatureBlock]:
    blocks: list[DualFeatureBlock] = []
    train_all_smiles = list(train_other_smiles) + list(train_finetune_smiles)
    n_other = len(train_other_smiles)
    n_finetune = len(train_finetune_smiles)

    for spec in parse_feature_set(feature_set):
        if spec.family == "binary":
            X_train_all = _binary_block(train_all_smiles, spec)
            X_train_other = X_train_all[:n_other]
            X_train_finetune = X_train_all[n_other:n_other + n_finetune]
            X_test = _binary_block(test_smiles, spec)
            blocks.append(
                DualFeatureBlock(
                    name=spec.raw_name,
                    family="binary",
                    train_all=X_train_all,
                    train_finetune=X_train_finetune,
                    train_other=X_train_other,
                    test=X_test,
                )
            )
        elif spec.kind == "mordred":
            X_train_all, X_test = _mordred_block(train_all_smiles, test_smiles)
            X_train_other = X_train_all[:n_other]
            X_train_finetune = X_train_all[n_other:n_other + n_finetune]
            blocks.append(
                DualFeatureBlock(
                    name=spec.raw_name,
                    family="dense",
                    train_all=X_train_all,
                    train_finetune=X_train_finetune,
                    train_other=X_train_other,
                    test=X_test,
                )
            )
        elif spec.kind == "chemeleon":
            X_train_all, X_test = _chemeleon_block(train_all_smiles, test_smiles)
            X_train_other = X_train_all[:n_other]
            X_train_finetune = X_train_all[n_other:n_other + n_finetune]
            blocks.append(
                DualFeatureBlock(
                    name=spec.raw_name,
                    family="dense",
                    train_all=X_train_all,
                    train_finetune=X_train_finetune,
                    train_other=X_train_other,
                    test=X_test,
                )
            )
        else:
            raise ValueError(f"Unsupported feature kind: {spec.kind}")

    return blocks


def _combine_blocks_for_embedding(blocks: list[FeatureBlock]) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.hstack([b.train for b in blocks]).astype(np.float32, copy=False)
    X_test = np.hstack([b.test for b in blocks]).astype(np.float32, copy=False)
    return X_train, X_test


def _combine_dual_blocks_for_embedding(
    blocks: list[DualFeatureBlock],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_other = np.hstack([b.train_other for b in blocks]).astype(np.float32, copy=False)
    X_finetune = np.hstack([b.train_finetune for b in blocks]).astype(np.float32, copy=False)
    X_test = np.hstack([b.test for b in blocks]).astype(np.float32, copy=False)
    return X_other, X_finetune, X_test


def _compute_similarity_matrix_for_block(block: FeatureBlock) -> np.ndarray:
    if block.family == "binary":
        return _tanimoto_similarity_matrix(block.test, block.train)
    return cosine_similarity(block.test, block.train).astype(np.float32, copy=False)


def _compute_dual_similarity_matrix_for_block(
    block: DualFeatureBlock,
    reference: Literal["all", "finetune"],
) -> np.ndarray:
    ref = block.train_all if reference == "all" else block.train_finetune
    if ref.shape[0] == 0:
        raise ValueError("Reference training subset has zero rows")
    if block.family == "binary":
        return _tanimoto_similarity_matrix(block.test, ref)
    return cosine_similarity(block.test, ref).astype(np.float32, copy=False)


def _compute_train_nn(block_matrix: np.ndarray, family: Literal["binary", "dense"]) -> np.ndarray:
    if block_matrix.shape[0] == 0:
        raise ValueError("Reference training subset has zero rows")
    if block_matrix.shape[0] == 1:
        return np.array([1.0], dtype=np.float32)

    if family == "binary":
        sim_train = _tanimoto_similarity_matrix(block_matrix, block_matrix)
    else:
        sim_train = cosine_similarity(block_matrix, block_matrix).astype(np.float32, copy=False)

    np.fill_diagonal(sim_train, -np.inf)
    return sim_train.max(axis=1)


def compute_embedding(
    X_all: np.ndarray,
    *,
    feature_set: str,
    embedding_method: EmbeddingMethod,
) -> np.ndarray:
    if embedding_method == "pca":
        reducer = PCA(n_components=2, random_state=0)
        return reducer.fit_transform(X_all)

    if embedding_method == "umap":
        import umap

        specs = parse_feature_set(feature_set)
        all_binary = all(spec.family == "binary" for spec in specs)
        metric = "jaccard" if all_binary else "cosine"

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric=metric,
            random_state=0,
        )
        return reducer.fit_transform(X_all)

    raise ValueError(f"Unsupported embedding method: {embedding_method}")


def compute_applicability_domain(
    train_smiles: list[str],
    test_smiles: list[str],
    *,
    feature_set: str = "morgan",
    embedding_method: EmbeddingMethod = "pca",
    top_k: int = 5,
) -> ADResult:
    blocks = build_feature_blocks(train_smiles, test_smiles, feature_set)

    sim_blocks = [_compute_similarity_matrix_for_block(b) for b in blocks]
    sim = np.mean(sim_blocks, axis=0).astype(np.float32, copy=False)

    train_nn_blocks = [_compute_train_nn(b.train, b.family) for b in blocks]
    train_nn_similarity = np.mean(train_nn_blocks, axis=0).astype(np.float32, copy=False)

    test_max_similarity = sim.max(axis=1)
    k = min(top_k, sim.shape[1])
    topk = np.partition(sim, -k, axis=1)[:, -k:]
    test_mean_topk_similarity = topk.mean(axis=1)

    X_train, X_test = _combine_blocks_for_embedding(blocks)
    X_all = np.vstack([X_train, X_test]).astype(np.float32, copy=False)
    embedding_coords = compute_embedding(X_all, feature_set=feature_set, embedding_method=embedding_method)

    return ADResult(
        feature_set_name=feature_set,
        train_nn_similarity=train_nn_similarity,
        test_max_similarity=test_max_similarity,
        test_mean_topk_similarity=test_mean_topk_similarity,
        embedding_coords=embedding_coords,
        n_train=len(train_smiles),
        n_test=len(test_smiles),
    )


def compute_dual_applicability_domain(
    train_other_smiles: list[str],
    train_finetune_smiles: list[str],
    test_smiles: list[str],
    *,
    feature_set: str = "morgan",
    embedding_method: EmbeddingMethod = "pca",
    top_k: int = 5,
) -> DualADResult:
    if len(train_finetune_smiles) == 0:
        raise ValueError("Fine-tune subset has zero rows")

    blocks = build_dual_feature_blocks(
        train_other_smiles=train_other_smiles,
        train_finetune_smiles=train_finetune_smiles,
        test_smiles=test_smiles,
        feature_set=feature_set,
    )

    sim_all_blocks = [_compute_dual_similarity_matrix_for_block(b, "all") for b in blocks]
    sim_all = np.mean(sim_all_blocks, axis=0).astype(np.float32, copy=False)

    sim_ft_blocks = [_compute_dual_similarity_matrix_for_block(b, "finetune") for b in blocks]
    sim_ft = np.mean(sim_ft_blocks, axis=0).astype(np.float32, copy=False)

    train_all_nn_blocks = [_compute_train_nn(b.train_all, b.family) for b in blocks]
    train_all_nn_similarity = np.mean(train_all_nn_blocks, axis=0).astype(np.float32, copy=False)

    train_ft_nn_blocks = [_compute_train_nn(b.train_finetune, b.family) for b in blocks]
    train_finetune_nn_similarity = np.mean(train_ft_nn_blocks, axis=0).astype(np.float32, copy=False)

    test_max_similarity_to_train_all = sim_all.max(axis=1)
    test_max_similarity_to_train_finetune = sim_ft.max(axis=1)

    k_all = min(top_k, sim_all.shape[1])
    topk_all = np.partition(sim_all, -k_all, axis=1)[:, -k_all:]
    test_mean_topk_similarity_to_train_all = topk_all.mean(axis=1)

    k_ft = min(top_k, sim_ft.shape[1])
    topk_ft = np.partition(sim_ft, -k_ft, axis=1)[:, -k_ft:]
    test_mean_topk_similarity_to_train_finetune = topk_ft.mean(axis=1)

    X_other, X_finetune, X_test = _combine_dual_blocks_for_embedding(blocks)
    X_all_embed = np.vstack([X_other, X_finetune, X_test]).astype(np.float32, copy=False)
    embedding_coords = compute_embedding(X_all_embed, feature_set=feature_set, embedding_method=embedding_method)

    return DualADResult(
        feature_set_name=feature_set,
        train_all_nn_similarity=train_all_nn_similarity,
        train_finetune_nn_similarity=train_finetune_nn_similarity,
        test_max_similarity_to_train_all=test_max_similarity_to_train_all,
        test_mean_topk_similarity_to_train_all=test_mean_topk_similarity_to_train_all,
        test_max_similarity_to_train_finetune=test_max_similarity_to_train_finetune,
        test_mean_topk_similarity_to_train_finetune=test_mean_topk_similarity_to_train_finetune,
        embedding_coords=embedding_coords,
        n_train_other=len(train_other_smiles),
        n_train_finetune=len(train_finetune_smiles),
        n_test=len(test_smiles),
    )


def combined_user_cutoff(
    feature_set: str,
    *,
    tanimoto_cutoff: float | None,
    cosine_cutoff: float | None,
) -> float | None:
    cutoffs: list[float] = []
    for spec in parse_feature_set(feature_set):
        if spec.family == "binary":
            if tanimoto_cutoff is None:
                return None
            cutoffs.append(float(tanimoto_cutoff))
        else:
            if cosine_cutoff is None:
                return None
            cutoffs.append(float(cosine_cutoff))
    return float(np.mean(cutoffs)) if cutoffs else None


def build_test_ad_table(
    df_test: pd.DataFrame,
    *,
    feature_set: str,
    test_max_similarity: np.ndarray,
    test_mean_topk_similarity: np.ndarray,
    user_cutoff: float | None = None,
    train_p5_cutoff: float | None = None,
) -> pd.DataFrame:
    out = df_test.copy()
    out["feature_set"] = feature_set
    out["max_similarity_to_train"] = test_max_similarity
    out["mean_topk_similarity_to_train"] = test_mean_topk_similarity

    if user_cutoff is not None:
        out["inside_ad_user_cutoff"] = out["max_similarity_to_train"] >= user_cutoff
        out["outside_ad_user_cutoff"] = ~out["inside_ad_user_cutoff"]

    if train_p5_cutoff is not None:
        out["inside_ad_train_p5"] = out["max_similarity_to_train"] >= train_p5_cutoff
        out["outside_ad_train_p5"] = ~out["inside_ad_train_p5"]

    return out


def build_dual_test_ad_table(
    df_test: pd.DataFrame,
    *,
    feature_set: str,
    test_max_similarity_to_train_all: np.ndarray,
    test_mean_topk_similarity_to_train_all: np.ndarray,
    test_max_similarity_to_train_finetune: np.ndarray,
    test_mean_topk_similarity_to_train_finetune: np.ndarray,
    user_cutoff: float | None = None,
    train_all_p5_cutoff: float | None = None,
    train_finetune_p5_cutoff: float | None = None,
) -> pd.DataFrame:
    out = df_test.copy()
    out["feature_set"] = feature_set

    out["max_similarity_to_train_all"] = test_max_similarity_to_train_all
    out["mean_topk_similarity_to_train_all"] = test_mean_topk_similarity_to_train_all
    out["max_similarity_to_train_finetune"] = test_max_similarity_to_train_finetune
    out["mean_topk_similarity_to_train_finetune"] = test_mean_topk_similarity_to_train_finetune
    out["delta_similarity_finetune_minus_all"] = (
        out["max_similarity_to_train_finetune"] - out["max_similarity_to_train_all"]
    )

    if user_cutoff is not None:
        out["inside_ad_train_all_user_cutoff"] = out["max_similarity_to_train_all"] >= user_cutoff
        out["outside_ad_train_all_user_cutoff"] = ~out["inside_ad_train_all_user_cutoff"]
        out["inside_ad_train_finetune_user_cutoff"] = out["max_similarity_to_train_finetune"] >= user_cutoff
        out["outside_ad_train_finetune_user_cutoff"] = ~out["inside_ad_train_finetune_user_cutoff"]

    if train_all_p5_cutoff is not None:
        out["inside_ad_train_all_p5"] = out["max_similarity_to_train_all"] >= train_all_p5_cutoff
        out["outside_ad_train_all_p5"] = ~out["inside_ad_train_all_p5"]

    if train_finetune_p5_cutoff is not None:
        out["inside_ad_train_finetune_p5"] = out["max_similarity_to_train_finetune"] >= train_finetune_p5_cutoff
        out["outside_ad_train_finetune_p5"] = ~out["inside_ad_train_finetune_p5"]

    return out


def plot_applicability_domain(
    *,
    embedding_coords: np.ndarray,
    n_train: int,
    test_max_similarity: np.ndarray,
    out_path: str,
    title_prefix: str,
    embedding_method: EmbeddingMethod = "pca",
    user_cutoff: float | None = None,
    train_p5_cutoff: float | None = None,
) -> None:
    train_xy = embedding_coords[:n_train]
    test_xy = embedding_coords[n_train:]

    median_train = float(np.median(test_max_similarity))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(train_xy[:, 0], train_xy[:, 1], s=18, alpha=0.6, label="training_set")
    ax.scatter(test_xy[:, 0], test_xy[:, 1], s=24, alpha=0.85, label="test_set")
    ax.set_title(f"{title_prefix}: {embedding_method.upper()}")
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.legend()

    ax = axes[1]
    ax.hist(test_max_similarity, bins=30)
    ax.axvline(median_train, linestyle="-.", label=f"median train={median_train:.3f}")
    if user_cutoff is not None:
        ax.axvline(user_cutoff, linestyle="--", label=f"user cutoff={user_cutoff:.3f}")
    if train_p5_cutoff is not None:
        ax.axvline(train_p5_cutoff, linestyle=":", label=f"train p5={train_p5_cutoff:.3f}")
    ax.set_title(f"{title_prefix}: test vs train")
    ax.set_xlabel("max similarity")
    ax.set_ylabel("count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dual_applicability_domain(
    *,
    embedding_coords: np.ndarray,
    n_train_other: int,
    n_train_finetune: int,
    test_max_similarity_to_train_all: np.ndarray,
    test_max_similarity_to_train_finetune: np.ndarray,
    out_path: str,
    title_prefix: str,
    embedding_method: EmbeddingMethod = "pca",
    user_cutoff: float | None = None,
    train_all_p5_cutoff: float | None = None,
    train_finetune_p5_cutoff: float | None = None,
) -> None:
    idx_other_end = n_train_other
    idx_ft_end = n_train_other + n_train_finetune

    other_xy = embedding_coords[:idx_other_end]
    finetune_xy = embedding_coords[idx_other_end:idx_ft_end]
    test_xy = embedding_coords[idx_ft_end:]

    median_train = float(np.median(test_max_similarity_to_train_all))
    median_finetune = float(np.median(test_max_similarity_to_train_finetune))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    if len(other_xy) > 0:
        ax.scatter(other_xy[:, 0], other_xy[:, 1], s=16, alpha=0.45, label="training_set_other")
    if len(finetune_xy) > 0:
        ax.scatter(finetune_xy[:, 0], finetune_xy[:, 1], s=22, alpha=0.75, label="training_set_finetune")
    ax.scatter(test_xy[:, 0], test_xy[:, 1], s=24, alpha=0.85, label="test_set")
    ax.set_title(f"{title_prefix}: {embedding_method.upper()}")
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.legend()

    ax = axes[1]
    ax.hist(test_max_similarity_to_train_all, bins=30)
    ax.axvline(median_train, linestyle="-.", label=f"median train={median_train:.3f}")
    if user_cutoff is not None:
        ax.axvline(user_cutoff, linestyle="--", label=f"user cutoff={user_cutoff:.3f}")
    if train_all_p5_cutoff is not None:
        ax.axvline(train_all_p5_cutoff, linestyle=":", label=f"all train p5={train_all_p5_cutoff:.3f}")
    ax.set_title(f"{title_prefix}: test vs all train")
    ax.set_xlabel("max similarity")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[2]
    ax.hist(test_max_similarity_to_train_finetune, bins=30)
    ax.axvline(median_finetune, linestyle="-.", label=f"median finetune={median_finetune:.3f}")
    if user_cutoff is not None:
        ax.axvline(user_cutoff, linestyle="--", label=f"user cutoff={user_cutoff:.3f}")
    if train_finetune_p5_cutoff is not None:
        ax.axvline(train_finetune_p5_cutoff, linestyle=":", label=f"finetune p5={train_finetune_p5_cutoff:.3f}")
    ax.set_title(f"{title_prefix}: test vs fine-tune subset")
    ax.set_xlabel("max similarity")
    ax.set_ylabel("count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_applicability_domain_batch(
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    smiles_column: str,
    feature_sets: list[str],
    output_dir: str,
    embedding_method: EmbeddingMethod,
    top_k: int,
    tanimoto_cutoff: float | None,
    cosine_cutoff: float | None,
    use_train_p5_cutoff: bool,
    best_by: Literal["user_cutoff", "train_p5"] = "user_cutoff",
) -> pd.DataFrame:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_smiles = df_train[smiles_column].astype(str).tolist()
    test_smiles = df_test[smiles_column].astype(str).tolist()

    summary_rows = []

    for feature_set in feature_sets:
        result = compute_applicability_domain(
            train_smiles=train_smiles,
            test_smiles=test_smiles,
            feature_set=feature_set,
            embedding_method=embedding_method,
            top_k=top_k,
        )

        feature_dir = output_root / feature_set.replace("+", "__")
        feature_dir.mkdir(parents=True, exist_ok=True)

        user_cutoff = combined_user_cutoff(
            feature_set,
            tanimoto_cutoff=tanimoto_cutoff,
            cosine_cutoff=cosine_cutoff,
        )

        train_p5_cutoff = None
        if use_train_p5_cutoff:
            train_p5_cutoff = float(np.percentile(result.train_nn_similarity, 5))

        df_test_out = build_test_ad_table(
            df_test,
            feature_set=feature_set,
            test_max_similarity=result.test_max_similarity,
            test_mean_topk_similarity=result.test_mean_topk_similarity,
            user_cutoff=user_cutoff,
            train_p5_cutoff=train_p5_cutoff,
        )
        df_test_out.to_csv(feature_dir / "test_ad.csv", index=False)

        plot_applicability_domain(
            embedding_coords=result.embedding_coords,
            n_train=result.n_train,
            test_max_similarity=result.test_max_similarity,
            out_path=str(feature_dir / "ad_plot.png"),
            title_prefix=f"Applicability Domain ({feature_set})",
            embedding_method=embedding_method,
            user_cutoff=user_cutoff,
            train_p5_cutoff=train_p5_cutoff,
        )

        row = {
            "feature_set": feature_set,
            "n_train": len(df_train),
            "n_test": len(df_test),
            "user_cutoff": user_cutoff,
            "train_p5_cutoff": train_p5_cutoff,
            "median_similarity_to_train": float(np.median(result.test_max_similarity)),
            "output_subdir": str(feature_dir),
        }

        if user_cutoff is not None:
            row["n_outside_ad_user_cutoff"] = int(df_test_out["outside_ad_user_cutoff"].sum())
            row["pct_outside_ad_user_cutoff"] = 100.0 * row["n_outside_ad_user_cutoff"] / len(df_test_out)

        if train_p5_cutoff is not None:
            row["n_outside_ad_train_p5"] = int(df_test_out["outside_ad_train_p5"].sum())
            row["pct_outside_ad_train_p5"] = 100.0 * row["n_outside_ad_train_p5"] / len(df_test_out)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    rank_col = "n_outside_ad_user_cutoff" if best_by == "user_cutoff" else "n_outside_ad_train_p5"
    if rank_col in summary.columns:
        summary = summary.sort_values([rank_col, "median_similarity_to_train"], ascending=[True, False]).reset_index(drop=True)
        summary["is_best"] = False
        if len(summary) > 0:
            summary.loc[0, "is_best"] = True
    else:
        summary["is_best"] = False

    summary.to_csv(output_root / "ad_summary.csv", index=False)
    return summary


def run_dual_applicability_domain_batch(
    *,
    df_train_other: pd.DataFrame,
    df_train_finetune: pd.DataFrame,
    df_test: pd.DataFrame,
    smiles_column: str,
    feature_sets: list[str],
    output_dir: str,
    embedding_method: EmbeddingMethod,
    top_k: int,
    tanimoto_cutoff: float | None,
    cosine_cutoff: float | None,
    use_train_p5_cutoff: bool,
    best_by: Literal["all_train_p5", "finetune_p5", "user_cutoff_all", "user_cutoff_finetune"] = "finetune_p5",
    finetune_subset_label: str | None = None,
) -> pd.DataFrame:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_other_smiles = df_train_other[smiles_column].astype(str).tolist()
    train_finetune_smiles = df_train_finetune[smiles_column].astype(str).tolist()
    test_smiles = df_test[smiles_column].astype(str).tolist()

    summary_rows = []

    for feature_set in feature_sets:
        result = compute_dual_applicability_domain(
            train_other_smiles=train_other_smiles,
            train_finetune_smiles=train_finetune_smiles,
            test_smiles=test_smiles,
            feature_set=feature_set,
            embedding_method=embedding_method,
            top_k=top_k,
        )

        feature_dir = output_root / feature_set.replace("+", "__")
        feature_dir.mkdir(parents=True, exist_ok=True)

        user_cutoff = combined_user_cutoff(
            feature_set,
            tanimoto_cutoff=tanimoto_cutoff,
            cosine_cutoff=cosine_cutoff,
        )

        train_all_p5_cutoff = None
        train_finetune_p5_cutoff = None
        if use_train_p5_cutoff:
            train_all_p5_cutoff = float(np.percentile(result.train_all_nn_similarity, 5))
            train_finetune_p5_cutoff = float(np.percentile(result.train_finetune_nn_similarity, 5))

        df_test_out = build_dual_test_ad_table(
            df_test,
            feature_set=feature_set,
            test_max_similarity_to_train_all=result.test_max_similarity_to_train_all,
            test_mean_topk_similarity_to_train_all=result.test_mean_topk_similarity_to_train_all,
            test_max_similarity_to_train_finetune=result.test_max_similarity_to_train_finetune,
            test_mean_topk_similarity_to_train_finetune=result.test_mean_topk_similarity_to_train_finetune,
            user_cutoff=user_cutoff,
            train_all_p5_cutoff=train_all_p5_cutoff,
            train_finetune_p5_cutoff=train_finetune_p5_cutoff,
        )
        df_test_out.to_csv(feature_dir / "test_ad.csv", index=False)

        plot_dual_applicability_domain(
            embedding_coords=result.embedding_coords,
            n_train_other=result.n_train_other,
            n_train_finetune=result.n_train_finetune,
            test_max_similarity_to_train_all=result.test_max_similarity_to_train_all,
            test_max_similarity_to_train_finetune=result.test_max_similarity_to_train_finetune,
            out_path=str(feature_dir / "ad_plot.png"),
            title_prefix=f"Applicability Domain ({feature_set})",
            embedding_method=embedding_method,
            user_cutoff=user_cutoff,
            train_all_p5_cutoff=train_all_p5_cutoff,
            train_finetune_p5_cutoff=train_finetune_p5_cutoff,
        )

        mean_topk_all = float(np.mean(result.test_mean_topk_similarity_to_train_all))
        mean_topk_finetune = float(np.mean(result.test_mean_topk_similarity_to_train_finetune))

        row = {
            "feature_set": feature_set,
            "n_train_other": len(df_train_other),
            "n_train_finetune": len(df_train_finetune),
            "n_test": len(df_test),
            "finetune_subset_label": finetune_subset_label,
            "user_cutoff": user_cutoff,
            "train_all_p5_cutoff": train_all_p5_cutoff,
            "train_finetune_p5_cutoff": train_finetune_p5_cutoff,
            "median_similarity_to_train_all": float(np.median(result.test_max_similarity_to_train_all)),
            "median_similarity_to_train_finetune": float(np.median(result.test_max_similarity_to_train_finetune)),
            "mean_topk_similarity_to_train_all": mean_topk_all,
            "mean_topk_similarity_to_train_finetune": mean_topk_finetune,
            "delta_mean_topk_finetune_minus_all": mean_topk_finetune - mean_topk_all,
            "fine_tuning_preferred": bool(mean_topk_finetune > mean_topk_all),
            "output_subdir": str(feature_dir),
        }

        if user_cutoff is not None:
            row["n_outside_ad_user_cutoff_all"] = int(df_test_out["outside_ad_train_all_user_cutoff"].sum())
            row["pct_outside_ad_user_cutoff_all"] = 100.0 * row["n_outside_ad_user_cutoff_all"] / len(df_test_out)
            row["n_outside_ad_user_cutoff_finetune"] = int(df_test_out["outside_ad_train_finetune_user_cutoff"].sum())
            row["pct_outside_ad_user_cutoff_finetune"] = 100.0 * row["n_outside_ad_user_cutoff_finetune"] / len(df_test_out)

        if train_all_p5_cutoff is not None:
            row["n_outside_ad_train_all_p5"] = int(df_test_out["outside_ad_train_all_p5"].sum())
            row["pct_outside_ad_train_all_p5"] = 100.0 * row["n_outside_ad_train_all_p5"] / len(df_test_out)

        if train_finetune_p5_cutoff is not None:
            row["n_outside_ad_train_finetune_p5"] = int(df_test_out["outside_ad_train_finetune_p5"].sum())
            row["pct_outside_ad_train_finetune_p5"] = 100.0 * row["n_outside_ad_train_finetune_p5"] / len(df_test_out)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    best_map = {
        "all_train_p5": "n_outside_ad_train_all_p5",
        "finetune_p5": "n_outside_ad_train_finetune_p5",
        "user_cutoff_all": "n_outside_ad_user_cutoff_all",
        "user_cutoff_finetune": "n_outside_ad_user_cutoff_finetune",
    }
    rank_col = best_map[best_by]

    if rank_col in summary.columns:
        tie_break = "median_similarity_to_train_finetune" if "finetune" in best_by else "median_similarity_to_train_all"
        summary = summary.sort_values([rank_col, tie_break], ascending=[True, False]).reset_index(drop=True)
        summary["is_best"] = False
        if len(summary) > 0:
            summary.loc[0, "is_best"] = True
    else:
        summary["is_best"] = False

    summary.to_csv(output_root / "ad_summary.csv", index=False)
    return summary
