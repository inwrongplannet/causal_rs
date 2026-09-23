import numpy as np
import pandas as pd
import pytest
from src.data_pipeline.scm_builder import (
    split_by_impression_id,
    reduce_embedding_columns,
    run_quality_checks,
)


class TestSplitByImpressionId:
    @pytest.fixture
    def sample_df(self):
        rows = []
        for i in range(100):
            rows.append({"impression_id": f"imp_{i}", "A": 1, "Y_click": 0, "Y_diversity": 0.5})
        return pd.DataFrame(rows)

    def test_split_ratios(self, sample_df):
        train, val, test, report = split_by_impression_id(sample_df, (0.7, 0.15, 0.15), 42)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)
        assert report["total_impressions"] == 100

    def test_no_leakage(self, sample_df):
        train, val, test, _ = split_by_impression_id(sample_df, (0.7, 0.15, 0.15), 42)
        train_keys = set(train["impression_id"])
        val_keys = set(val["impression_id"])
        test_keys = set(test["impression_id"])
        assert not (train_keys & val_keys)
        assert not (train_keys & test_keys)
        assert not (val_keys & test_keys)

    def test_invalid_ratios(self, sample_df):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_by_impression_id(sample_df, (0.5, 0.5, 0.5), 42)

    def test_deterministic_with_seed(self, sample_df):
        t1, _, _, _ = split_by_impression_id(sample_df, (0.7, 0.15, 0.15), 42)
        t2, _, _, _ = split_by_impression_id(sample_df, (0.7, 0.15, 0.15), 42)
        assert t1["impression_id"].tolist() == t2["impression_id"].tolist()


class TestReduceEmbeddingColumns:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(42)
        records = []
        for i in range(100):
            records.append({
                "U_history_emb_full": rng.random(10).tolist(),
                "I_entity_emb_full": rng.random(5).tolist(),
                "I_title_emb_full": rng.random(5).tolist(),
            })
        return pd.DataFrame(records)

    def test_pca_columns_added(self, sample_df):
        reduced, report = reduce_embedding_columns(sample_df, 5, 42)
        assert f"U_pca_0" in reduced.columns
        assert f"U_pca_4" in reduced.columns
        assert "I_entity_pca_0" in reduced.columns

    def test_pca_report(self, sample_df):
        reduced, report = reduce_embedding_columns(sample_df, 5, 42)
        assert "U_pca" in report
        assert "I_entity_pca" in report
        assert 0 < report["U_pca"]["explained_variance_ratio_sum"] <= 1.0

    def test_n_components_greater_than_matrix(self, sample_df):
        reduced, report = reduce_embedding_columns(sample_df, 100, 42)
        assert "U_pca_99" in reduced.columns

    def test_empty_dataframe(self):
        with pytest.raises(KeyError):
            reduce_embedding_columns(pd.DataFrame(), 5, 42)

    def test_original_columns_preserved(self, sample_df):
        original_cols = set(sample_df.columns)
        reduced, _ = reduce_embedding_columns(sample_df, 5, 42)
        for col in original_cols:
            assert col in reduced.columns


class TestRunQualityChecks:
    @pytest.fixture
    def good_data(self):
        rows = {"user_id": [], "item_id": [], "impression_id": [], "A": [],
                "Y_click": [], "Y_diversity": [], "U_dwell_mean": [], "I_category": [], "I_sentiment": []}
        for i in range(100):
            rows["user_id"].append(f"U{i}")
            rows["item_id"].append(f"N{i}")
            rows["impression_id"].append(f"imp_{i % 10}")
            rows["A"].append(0 if i < 80 else 1)
            rows["Y_click"].append(0)
            rows["Y_diversity"].append(0.5 + 0.005 * i)
            rows["U_dwell_mean"].append(5.0)
            rows["I_category"].append("news")
            rows["I_sentiment"].append(0.0)
        df = pd.DataFrame(rows)
        train = df[df["impression_id"].isin([f"imp_{i}" for i in range(7)])]
        val = df[df["impression_id"].isin([f"imp_{i}" for i in range(7, 9)])]
        test = df[df["impression_id"].isin([f"imp_{i}" for i in range(9, 10)])]
        return train, val, test

    def test_good_data_passes(self, good_data):
        train, val, test = good_data
        required = ["user_id", "item_id", "impression_id", "A", "Y_click", "Y_diversity", "U_dwell_mean", "I_category", "I_sentiment"]
        report = run_quality_checks(train, val, test, required, 4)
        assert report["missing_columns"] == []
        assert report["ratio_check_pass"] is True
        assert report["split_leakage_check_pass"] is True

    def test_missing_column_fails(self, good_data):
        train, val, test = good_data
        required = ["user_id", "nonexistent_col"]
        with pytest.raises(AssertionError, match="Missing required columns"):
            run_quality_checks(train, val, test, required, 4)
