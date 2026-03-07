from __future__ import annotations

from dataclasses import dataclass
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


FeatureKind = Literal["morgan", "avalon", "rdkit_path", "mordred", "chemeleon"]
SimilarityMetric = Literal["auto", "tanimoto", "cosine"]
EmbeddingMethod = Literal["pca", "umap"]


@dataclass(frozen=True)
class ADResult:
    train_features: np.ndarray
    test_features: np.ndarray
    train_nn_similarity: np.ndarray
    test_max_similarity: np.ndarray
    test_mean_topk_similarity: np.ndarray
    embedding_coords: np.ndarray


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


def _featurize_binary_smiles(smiles: list[str], feature_kind: FeatureKind) -> np.ndarray:
    rows = []

    if feature_kind == "morgan":
        cfg = MorganFPConfig()
        fn = lambda s: morgan_bits_from_smiles(s, cfg)
    elif feature_kind == "avalon":
        cfg = AvalonFPConfig()
        fn = lambda s: avalon_bits_from_smiles(s, cfg)
    elif feature_kind == "rdkit_path":
        cfg = RDKitPathFPConfig()
        fn = lambda s: rdkit_path_bits_from_smiles(s, cfg)
    else:
        raise ValueError(f"Unsupported binary feature kind: {feature_kind}")

    for s in smiles:
        v = fn(s)
        if v is None:
            raise ValueError(f"Invalid SMILES encountered: {s!r}")
        rows.append(v)

    return np.vstack(rows).astype(np.float32, copy=False)


def featurize_train_test(
    train_smiles: list[str],
    test_smiles: list[str],
    feature_kind: FeatureKind,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_kind in {"morgan", "avalon", "rdkit_path"}:
        X_train = _featurize_binary_smiles(train_smiles, feature_kind)
        X_test = _featurize_binary_smiles(test_smiles, feature_kind)
        return X_train, X_test

    if feature_kind == "mordred":
        all_smiles = list(train_smiles) + list(test_smiles)
        X_all, names = mordred2d_batch_from_smiles(all_smiles)
        X_all, names = prune_mordred_matrix(X_all, names, max_nan_frac=0.2, drop_constant=True)
        X_all = SimpleImputer(strategy="median").fit_transform(X_all).astype(np.float32, copy=False)
        n_train = len(train_smiles)
        return X_all[:n_train], X_all[n_train:]

    if feature_kind == "chemeleon":
        all_smiles = list(train_smiles) + list(test_smiles)
        X_all, _ = chemeleon_batch_from_smiles(all_smiles, cfg=CheMeleonConfig())
        X_all = X_all.astype(np.float32, copy=False)
        n_train = len(train_smiles)
        return X_all[:n_train], X_all[n_train:]

    raise ValueError(f"Unsupported feature kind: {feature_kind}")


def compute_embedding(
    X_all: np.ndarray,
    *,
    feature_kind: FeatureKind,
    embedding_method: EmbeddingMethod,
) -> np.ndarray:
    if embedding_method == "pca":
        reducer = PCA(n_components=2, random_state=0)
        return reducer.fit_transform(X_all)

    if embedding_method == "umap":
        import umap

        if feature_kind in {"morgan", "avalon", "rdkit_path"}:
            metric = "jaccard"
        else:
            metric = "cosine"

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric=metric,
            random_state=0,
        )
        return reducer.fit_transform(X_all)

    raise ValueError(f"Unsupported embedding method: {embedding_method}")


def _compute_train_nn_similarity(
    X_train: np.ndarray,
    *,
    similarity_metric: SimilarityMetric,
) -> np.ndarray:
    if X_train.shape[0] == 1:
        return np.array([1.0], dtype=np.float32)

    if similarity_metric == "tanimoto":
        sim_train = _tanimoto_similarity_matrix(X_train, X_train)
    elif similarity_metric == "cosine":
        sim_train = cosine_similarity(X_train, X_train)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    sim_train = sim_train.astype(np.float32, copy=False)
    np.fill_diagonal(sim_train, -np.inf)
    train_nn_similarity = sim_train.max(axis=1)
    return train_nn_similarity


def compute_applicability_domain(
    train_smiles: list[str],
    test_smiles: list[str],
    *,
    feature_kind: FeatureKind = "morgan",
    similarity_metric: SimilarityMetric = "auto",
    embedding_method: EmbeddingMethod = "pca",
    top_k: int = 5,
) -> ADResult:
    X_train, X_test = featurize_train_test(train_smiles, test_smiles, feature_kind)

    if similarity_metric == "auto":
        similarity_metric = "tanimoto" if feature_kind in {"morgan", "avalon", "rdkit_path"} else "cosine"

    if similarity_metric == "tanimoto":
        sim = _tanimoto_similarity_matrix(X_test, X_train)
    elif similarity_metric == "cosine":
        sim = cosine_similarity(X_test, X_train)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    train_nn_similarity = _compute_train_nn_similarity(
        X_train,
        similarity_metric=similarity_metric,
    )

    test_max_similarity = sim.max(axis=1)

    k = min(top_k, sim.shape[1])
    topk = np.partition(sim, -k, axis=1)[:, -k:]
    test_mean_topk_similarity = topk.mean(axis=1)

    X_all = np.vstack([X_train, X_test]).astype(np.float32, copy=False)
    embedding_coords = compute_embedding(
        X_all,
        feature_kind=feature_kind,
        embedding_method=embedding_method,
    )

    return ADResult(
        train_features=X_train,
        test_features=X_test,
        train_nn_similarity=train_nn_similarity,
        test_max_similarity=test_max_similarity,
        test_mean_topk_similarity=test_mean_topk_similarity,
        embedding_coords=embedding_coords,
    )


def build_test_ad_table(
    df_test: pd.DataFrame,
    *,
    test_max_similarity: np.ndarray,
    test_mean_topk_similarity: np.ndarray,
    user_cutoff: float | None = None,
    train_p5_cutoff: float | None = None,
) -> pd.DataFrame:
    out = df_test.copy()
    out["max_similarity_to_train"] = test_max_similarity
    out["mean_topk_similarity_to_train"] = test_mean_topk_similarity

    if user_cutoff is not None:
        out["inside_ad_user_cutoff"] = out["max_similarity_to_train"] >= user_cutoff
        out["outside_ad_user_cutoff"] = ~out["inside_ad_user_cutoff"]

    if train_p5_cutoff is not None:
        out["inside_ad_train_p5"] = out["max_similarity_to_train"] >= train_p5_cutoff
        out["outside_ad_train_p5"] = ~out["inside_ad_train_p5"]

    return out


def plot_applicability_domain(
    *,
    embedding_coords: np.ndarray,
    n_train: int,
    test_max_similarity: np.ndarray,
    out_path: str,
    title_prefix: str = "Applicability Domain",
    embedding_method: EmbeddingMethod = "pca",
    user_cutoff: float | None = None,
    train_p5_cutoff: float | None = None,
) -> None:
    train_xy = embedding_coords[:n_train]
    test_xy = embedding_coords[n_train:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(train_xy[:, 0], train_xy[:, 1], s=18, alpha=0.6, label="train")
    ax.scatter(test_xy[:, 0], test_xy[:, 1], s=22, alpha=0.8, label="test")
    ax.set_title(f"{title_prefix}: {embedding_method.upper()}")
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.legend()

    ax = axes[1]
    ax.hist(test_max_similarity, bins=30)
    if user_cutoff is not None:
        ax.axvline(user_cutoff, linestyle="--", label=f"user cutoff={user_cutoff:.3f}")
    if train_p5_cutoff is not None:
        ax.axvline(train_p5_cutoff, linestyle=":", label=f"train p5={train_p5_cutoff:.3f}")
    ax.set_title(f"{title_prefix}: test max similarity to train")
    ax.set_xlabel("max similarity to train")
    ax.set_ylabel("count")
    if user_cutoff is not None or train_p5_cutoff is not None:
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)