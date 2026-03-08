import numpy as np
import pandas as pd
import pytest

from deepmirror_predict.analysis import applicability_domain as ad


def test_expand_feature_sets_expands_parameterized_bases():
    out = ad.expand_feature_sets(
        ["morgan", "avalon", "rdkit_path", "chemeleon", "morgan+rdkit_path"],
        morgan_radius=[2, 3],
        morgan_bits=[1024],
        avalon_bits=[512, 1024],
        rdkit_path_min=[1, 2],
        rdkit_path_max=[3],
        rdkit_path_bits=[1024],
    )

    assert "morgan_r2_b1024" in out
    assert "morgan_r3_b1024" in out
    assert "avalon_b512" in out
    assert "avalon_b1024" in out
    assert "rdkit_path_min1_max3_b1024" in out
    assert "rdkit_path_min2_max3_b1024" in out
    assert "chemeleon" in out
    assert "morgan_r2_b1024+rdkit_path_min1_max3_b1024" in out
    assert len(out) == len(set(out))


def test_parse_feature_token_variants():
    morgan = ad.parse_feature_token("morgan_r2_b2048")
    assert morgan.kind == "morgan"
    assert morgan.family == "binary"
    assert morgan.params["radius"] == 2
    assert morgan.params["n_bits"] == 2048

    avalon = ad.parse_feature_token("avalon_b4096")
    assert avalon.kind == "avalon"
    assert avalon.params["n_bits"] == 4096

    rdkit_path = ad.parse_feature_token("rdkit_path_min2_max7_b1024")
    assert rdkit_path.kind == "rdkit_path"
    assert rdkit_path.params["min_path"] == 2
    assert rdkit_path.params["max_path"] == 7
    assert rdkit_path.params["n_bits"] == 1024

    with pytest.raises(ValueError):
        ad.parse_feature_token("not_a_feature")


def test_combined_user_cutoff_binary_only():
    cutoff = ad.combined_user_cutoff(
        "morgan_r2_b1024+avalon_b512",
        tanimoto_cutoff=0.4,
        cosine_cutoff=0.8,
    )
    assert cutoff == pytest.approx(0.4)


def test_combined_user_cutoff_mixed_binary_dense():
    cutoff = ad.combined_user_cutoff(
        "morgan_r2_b1024+chemeleon",
        tanimoto_cutoff=0.4,
        cosine_cutoff=0.8,
    )
    assert cutoff == pytest.approx(0.6)


def test_build_test_ad_table_columns():
    df = pd.DataFrame({"mol_id": ["a", "b"]})

    out = ad.build_test_ad_table(
        df,
        feature_set="morgan_r2_b1024",
        test_max_similarity=np.array([0.9, 0.1], dtype=np.float32),
        test_mean_topk_similarity=np.array([0.8, 0.2], dtype=np.float32),
        user_cutoff=0.5,
        train_p5_cutoff=0.3,
    )

    assert "feature_set" in out.columns
    assert "max_similarity_to_train" in out.columns
    assert "mean_topk_similarity_to_train" in out.columns
    assert "inside_ad_user_cutoff" in out.columns
    assert "outside_ad_user_cutoff" in out.columns
    assert "inside_ad_train_p5" in out.columns
    assert "outside_ad_train_p5" in out.columns

    assert out["inside_ad_user_cutoff"].tolist() == [True, False]
    assert out["outside_ad_train_p5"].tolist() == [False, True]


def test_build_dual_test_ad_table_columns():
    df = pd.DataFrame({"mol_id": ["a", "b"]})

    out = ad.build_dual_test_ad_table(
        df,
        feature_set="morgan_r2_b1024",
        test_max_similarity_to_train_all=np.array([0.9, 0.1], dtype=np.float32),
        test_mean_topk_similarity_to_train_all=np.array([0.8, 0.2], dtype=np.float32),
        test_max_similarity_to_train_finetune=np.array([0.7, 0.3], dtype=np.float32),
        test_mean_topk_similarity_to_train_finetune=np.array([0.6, 0.4], dtype=np.float32),
        user_cutoff=0.5,
        train_all_p5_cutoff=0.25,
        train_finetune_p5_cutoff=0.35,
    )

    expected = {
        "max_similarity_to_train_all",
        "mean_topk_similarity_to_train_all",
        "max_similarity_to_train_finetune",
        "mean_topk_similarity_to_train_finetune",
        "delta_similarity_finetune_minus_all",
        "inside_ad_train_all_user_cutoff",
        "outside_ad_train_all_user_cutoff",
        "inside_ad_train_finetune_user_cutoff",
        "outside_ad_train_finetune_user_cutoff",
        "inside_ad_train_all_p5",
        "outside_ad_train_all_p5",
        "inside_ad_train_finetune_p5",
        "outside_ad_train_finetune_p5",
    }
    assert expected.issubset(set(out.columns))
    assert out["delta_similarity_finetune_minus_all"].tolist() == pytest.approx([-0.2, 0.2])
    assert out["inside_ad_train_finetune_user_cutoff"].tolist() == [True, False]


def test_plot_applicability_domain_writes_file(tmp_path):
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.2, 1.0], [0.8, 1.2]],
        dtype=np.float32,
    )
    out_path = tmp_path / "ad_plot.png"

    ad.plot_applicability_domain(
        embedding_coords=coords,
        n_train=2,
        test_max_similarity=np.array([0.7, 0.4], dtype=np.float32),
        out_path=str(out_path),
        title_prefix="Test AD",
        embedding_method="pca",
        user_cutoff=0.5,
        train_p5_cutoff=0.3,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_dual_applicability_domain_writes_file(tmp_path):
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.2, 1.0], [0.8, 1.2]],
        dtype=np.float32,
    )
    out_path = tmp_path / "dual_ad_plot.png"

    ad.plot_dual_applicability_domain(
        embedding_coords=coords,
        n_train_other=2,
        n_train_finetune=1,
        test_max_similarity_to_train_all=np.array([0.7, 0.4], dtype=np.float32),
        test_max_similarity_to_train_finetune=np.array([0.8, 0.3], dtype=np.float32),
        out_path=str(out_path),
        title_prefix="Dual AD",
        embedding_method="pca",
        user_cutoff=0.5,
        train_all_p5_cutoff=0.3,
        train_finetune_p5_cutoff=0.35,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_compute_embedding_pca_shape():
    X = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    coords = ad.compute_embedding(X, feature_set="morgan_r2_b1024", embedding_method="pca")
    assert coords.shape == (4, 2)
    assert np.isfinite(coords).all()


def test_compute_embedding_umap_shape():
    pytest.importorskip("umap")

    X = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    coords = ad.compute_embedding(X, feature_set="morgan_r2_b1024", embedding_method="umap")
    assert coords.shape == (4, 2)
    assert np.isfinite(coords).all()


def test_compute_applicability_domain_morgan_real_features():
    train_smiles = ["CCO", "CCN", "c1ccccc1"]
    test_smiles = ["CCO", "CCCl"]

    result = ad.compute_applicability_domain(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        feature_set="morgan_r2_b1024",
        embedding_method="pca",
        top_k=2,
    )

    assert result.test_max_similarity.shape == (2,)
    assert result.test_mean_topk_similarity.shape == (2,)
    assert result.train_nn_similarity.shape == (3,)
    assert result.embedding_coords.shape == (5, 2)
    assert np.all(np.isfinite(result.test_max_similarity))
    assert np.all(result.test_max_similarity >= 0.0)
    assert np.all(result.test_max_similarity <= 1.0)
    assert result.test_max_similarity[0] == pytest.approx(1.0)


def test_compute_applicability_domain_avalon_real_features():
    train_smiles = ["CCO", "CCN", "c1ccccc1"]
    test_smiles = ["CCO", "CCCl"]

    result = ad.compute_applicability_domain(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        feature_set="avalon_b512",
        embedding_method="pca",
        top_k=2,
    )

    assert result.test_max_similarity.shape == (2,)
    assert result.train_nn_similarity.shape == (3,)
    assert result.embedding_coords.shape == (5, 2)
    assert np.all(np.isfinite(result.test_max_similarity))


def test_compute_applicability_domain_rdkit_path_real_features():
    train_smiles = ["CCO", "CCN", "c1ccccc1"]
    test_smiles = ["CCO", "CCCl"]

    result = ad.compute_applicability_domain(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        feature_set="rdkit_path_min1_max5_b1024",
        embedding_method="pca",
        top_k=2,
    )

    assert result.test_max_similarity.shape == (2,)
    assert result.train_nn_similarity.shape == (3,)
    assert result.embedding_coords.shape == (5, 2)
    assert np.all(np.isfinite(result.test_max_similarity))


def test_compute_applicability_domain_mordred_real_features():
    pytest.importorskip("mordred")

    train_smiles = ["CCO", "CCN"]
    test_smiles = ["CCCl"]

    result = ad.compute_applicability_domain(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        feature_set="mordred",
        embedding_method="pca",
        top_k=1,
    )

    assert result.test_max_similarity.shape == (1,)
    assert result.train_nn_similarity.shape == (2,)
    assert result.embedding_coords.shape == (3, 2)
    assert np.all(np.isfinite(result.test_max_similarity))


def test_compute_applicability_domain_chemeleon_real_features():
    pytest.importorskip("torch")

    train_smiles = ["CCO", "CCN"]
    test_smiles = ["CCCl"]

    result = ad.compute_applicability_domain(
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        feature_set="chemeleon",
        embedding_method="pca",
        top_k=1,
    )

    assert result.test_max_similarity.shape == (1,)
    assert result.train_nn_similarity.shape == (2,)
    assert result.embedding_coords.shape == (3, 2)
    assert np.all(np.isfinite(result.test_max_similarity))


def test_compute_dual_applicability_domain_morgan_real_features():
    result = ad.compute_dual_applicability_domain(
        train_other_smiles=["CCO", "CCN"],
        train_finetune_smiles=["c1ccccc1"],
        test_smiles=["CCO", "ClCCCl"],
        feature_set="morgan_r2_b1024",
        embedding_method="pca",
        top_k=1,
    )

    assert result.test_max_similarity_to_train_all.shape == (2,)
    assert result.test_max_similarity_to_train_finetune.shape == (2,)
    assert result.test_mean_topk_similarity_to_train_all.shape == (2,)
    assert result.test_mean_topk_similarity_to_train_finetune.shape == (2,)
    assert result.train_all_nn_similarity.shape == (3,)
    assert result.train_finetune_nn_similarity.shape == (1,)
    assert result.embedding_coords.shape == (5, 2)
    assert result.n_train_other == 2
    assert result.n_train_finetune == 1
    assert result.test_max_similarity_to_train_all[0] == pytest.approx(1.0)


def test_run_applicability_domain_batch_writes_summary(tmp_path):
    df_train = pd.DataFrame({"SMILES_standardized": ["CCO", "CCN"]})
    df_test = pd.DataFrame({"SMILES_standardized": ["CCO", "CCCl"]})

    summary = ad.run_applicability_domain_batch(
        df_train=df_train,
        df_test=df_test,
        smiles_column="SMILES_standardized",
        feature_sets=["morgan_r2_b1024"],
        output_dir=str(tmp_path),
        embedding_method="pca",
        top_k=1,
        tanimoto_cutoff=0.5,
        cosine_cutoff=0.8,
        use_train_p5_cutoff=True,
        best_by="train_p5",
    )

    assert len(summary) == 1
    assert "median_similarity_to_train" in summary.columns
    assert "n_outside_ad_train_p5" in summary.columns
    assert "is_best" in summary.columns
    assert bool(summary.loc[0, "is_best"]) is True

    assert (tmp_path / "ad_summary.csv").exists()
    assert (tmp_path / "morgan_r2_b1024" / "test_ad.csv").exists()
    assert (tmp_path / "morgan_r2_b1024" / "ad_plot.png").exists()


def test_run_dual_applicability_domain_batch_writes_summary(tmp_path):
    df_train_other = pd.DataFrame({"SMILES_standardized": ["CCO"]})
    df_train_finetune = pd.DataFrame({"SMILES_standardized": ["c1ccccc1"]})
    df_test = pd.DataFrame({"SMILES_standardized": ["c1ccccc1", "CCO"]})

    summary = ad.run_dual_applicability_domain_batch(
        df_train_other=df_train_other,
        df_train_finetune=df_train_finetune,
        df_test=df_test,
        smiles_column="SMILES_standardized",
        feature_sets=["morgan_r2_b1024"],
        output_dir=str(tmp_path),
        embedding_method="pca",
        top_k=1,
        tanimoto_cutoff=0.5,
        cosine_cutoff=0.8,
        use_train_p5_cutoff=True,
        best_by="finetune_p5",
        finetune_subset_label="openadmet_expansion_train",
    )

    assert len(summary) == 1
    row = summary.iloc[0]

    assert "mean_topk_similarity_to_train_all" in summary.columns
    assert "mean_topk_similarity_to_train_finetune" in summary.columns
    assert "delta_mean_topk_finetune_minus_all" in summary.columns
    assert "fine_tuning_preferred" in summary.columns

    assert np.isfinite(row["mean_topk_similarity_to_train_all"])
    assert np.isfinite(row["mean_topk_similarity_to_train_finetune"])
    assert isinstance(bool(row["fine_tuning_preferred"]), bool)

    assert (tmp_path / "ad_summary.csv").exists()
    assert (tmp_path / "morgan_r2_b1024" / "test_ad.csv").exists()
    assert (tmp_path / "morgan_r2_b1024" / "ad_plot.png").exists()


def test_compute_dual_applicability_domain_requires_finetune_subset():
    with pytest.raises(ValueError, match="Fine-tune subset has zero rows"):
        ad.compute_dual_applicability_domain(
            train_other_smiles=["CCO"],
            train_finetune_smiles=[],
            test_smiles=["CCN"],
            feature_set="morgan_r2_b1024",
            embedding_method="pca",
            top_k=1,
        )
