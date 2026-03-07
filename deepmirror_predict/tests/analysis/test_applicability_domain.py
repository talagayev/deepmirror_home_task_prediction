import numpy as np
import pandas as pd
import pytest

from deepmirror_predict.analysis.applicability_domain import (
    build_test_ad_table,
    compute_applicability_domain,
    compute_embedding,
)


@pytest.mark.parametrize(
    "feature_kind,similarity_metric",
    [
        ("morgan", "auto"),
        ("avalon", "auto"),
        ("rdkit_path", "auto"),
    ],
)
def test_compute_applicability_domain_binary_features(feature_kind, similarity_metric):
    train_smiles = ["CCO", "CCN", "c1ccccc1"]
    test_smiles = ["CCO", "CCCl"]

    result = compute_applicability_domain(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        feature_kind=feature_kind,
        similarity_metric=similarity_metric,
        embedding_method="pca",
        top_k=2,
    )

    assert result.train_features.shape[0] == 3
    assert result.test_features.shape[0] == 2
    assert result.train_nn_similarity.shape == (3,)
    assert result.test_max_similarity.shape == (2,)
    assert result.test_mean_topk_similarity.shape == (2,)
    assert result.embedding_coords.shape == (5, 2)

    assert np.all(np.isfinite(result.train_nn_similarity))
    assert np.all(np.isfinite(result.test_max_similarity))
    assert np.all(np.isfinite(result.test_mean_topk_similarity))

    assert np.all(result.train_nn_similarity >= 0.0)
    assert np.all(result.train_nn_similarity <= 1.0)
    assert np.all(result.test_max_similarity >= 0.0)
    assert np.all(result.test_max_similarity <= 1.0)

    # exact match should be highly similar
    assert result.test_max_similarity[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize(
    "feature_kind,similarity_metric",
    [
        ("mordred", "auto"),
        ("chemeleon", "auto"),
    ],
)
def test_compute_applicability_domain_dense_features(feature_kind, similarity_metric):
    train_smiles = ["CCO", "CCN", "c1ccccc1"]
    test_smiles = ["CCO", "CCCl"]

    result = compute_applicability_domain(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        feature_kind=feature_kind,
        similarity_metric=similarity_metric,
        embedding_method="pca",
        top_k=2,
    )

    assert result.train_features.shape[0] == 3
    assert result.test_features.shape[0] == 2
    assert result.train_nn_similarity.shape == (3,)
    assert result.test_max_similarity.shape == (2,)
    assert result.test_mean_topk_similarity.shape == (2,)
    assert result.embedding_coords.shape == (5, 2)

    assert np.all(np.isfinite(result.train_nn_similarity))
    assert np.all(np.isfinite(result.test_max_similarity))
    assert np.all(np.isfinite(result.test_mean_topk_similarity))


def test_build_test_ad_table_with_cutoffs():
    df_test = pd.DataFrame(
        {
            "mol_id": ["m1", "m2", "m3"],
            "SMILES_standardized": ["CCO", "CCN", "CCC"],
        }
    )

    out = build_test_ad_table(
        df_test,
        test_max_similarity=np.array([0.8, 0.35, 0.1], dtype=np.float32),
        test_mean_topk_similarity=np.array([0.75, 0.3, 0.08], dtype=np.float32),
        user_cutoff=0.4,
        train_p5_cutoff=0.2,
    )

    assert "max_similarity_to_train" in out.columns
    assert "mean_topk_similarity_to_train" in out.columns
    assert "inside_ad_user_cutoff" in out.columns
    assert "outside_ad_user_cutoff" in out.columns
    assert "inside_ad_train_p5" in out.columns
    assert "outside_ad_train_p5" in out.columns

    assert out["inside_ad_user_cutoff"].tolist() == [True, False, False]
    assert out["outside_ad_user_cutoff"].tolist() == [False, True, True]
    assert out["inside_ad_train_p5"].tolist() == [True, True, False]
    assert out["outside_ad_train_p5"].tolist() == [False, False, True]


def test_compute_embedding_pca_shape():
    X_all = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    coords = compute_embedding(
        X_all,
        feature_kind="morgan",
        embedding_method="pca",
    )

    assert coords.shape == (4, 2)
    assert np.all(np.isfinite(coords))


def test_compute_embedding_umap_shape():
    pytest.importorskip("umap")

    X_all = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    coords = compute_embedding(
        X_all,
        feature_kind="morgan",
        embedding_method="umap",
    )

    assert coords.shape == (4, 2)
    assert np.all(np.isfinite(coords))


def test_single_train_sample():
    result = compute_applicability_domain(
        train_smiles=["CCO"],
        test_smiles=["CCN"],
        feature_kind="morgan",
        similarity_metric="auto",
        embedding_method="pca",
        top_k=1,
    )

    assert result.train_nn_similarity.shape == (1,)
    assert result.train_nn_similarity[0] == pytest.approx(1.0, rel=1e-6)
    assert result.test_max_similarity.shape == (1,)


def test_invalid_feature_kind_raises():
    with pytest.raises(ValueError, match="Unsupported feature kind"):
        compute_applicability_domain(
            train_smiles=["CCO"],
            test_smiles=["CCN"],
            feature_kind="not_valid",
            similarity_metric="auto",
            embedding_method="pca",
            top_k=1,
        )


def test_invalid_embedding_method_raises():
    X_all = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported embedding method"):
        compute_embedding(
            X_all,
            feature_kind="morgan",
            embedding_method="not_valid",
        )