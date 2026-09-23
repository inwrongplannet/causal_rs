import numpy as np
import pandas as pd

from src.data_pipeline.features import (
    build_session_user_features,
    compute_news_features,
    prepare_behaviors,
    sample_negative_items,
)


class TestSampleNegativeItems:
    def test_basic_sampling(self):
        pool = np.array(["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8"])
        shown = ["N1", "N2"]
        rng = np.random.default_rng(42)
        sampled = sample_negative_items(pool, shown, 2, rng)
        assert len(sampled) == 2
        assert all(s not in shown for s in sampled)
        assert all(s in pool for s in sampled)

    def test_zero_count(self):
        pool = np.array(["N1", "N2"])
        sampled = sample_negative_items(pool, [], 0, np.random.default_rng(42))
        assert sampled == []

    def test_negative_count(self):
        pool = np.array(["N1", "N2"])
        sampled = sample_negative_items(pool, [], -1, np.random.default_rng(42))
        assert sampled == []

    def test_all_items_shown(self):
        pool = np.array(["N1", "N2"])
        sampled = sample_negative_items(pool, ["N1", "N2"], 1, np.random.default_rng(42))
        assert sampled == []

    def test_with_replacement(self):
        pool = np.array(["N1"])
        shown = []
        sampled = sample_negative_items(pool, shown, 3, np.random.default_rng(42))
        assert len(sampled) == 3
        assert all(s == "N1" for s in sampled)


class TestPrepareBehaviors:
    def test_basic_preparation(self):
        df = pd.DataFrame({
            "ImpressionID": ["1"],
            "UserID": ["U1"],
            "Time": ["t1"],
            "History": ["N1 N2"],
            "Impressions": ["N3-1 N4-0"],
        })
        result = prepare_behaviors(df)
        assert len(result) == 1
        assert result["history_ids"].iloc[0] == ["N1", "N2"]
        assert result["impression_count"].iloc[0] == 2

    def test_empty_impressions_removed(self):
        df = pd.DataFrame({
            "ImpressionID": ["1", "2"],
            "UserID": ["U1", "U2"],
            "Time": ["t1", "t2"],
            "History": ["N1", ""],
            "Impressions": ["", ""],
        })
        result = prepare_behaviors(df)
        assert len(result) == 0

    def test_max_rows(self):
        df = pd.DataFrame({
            "ImpressionID": ["1", "2"],
            "UserID": ["U1", "U2"],
            "Time": ["t1", "t2"],
            "History": ["N1", "N2"],
            "Impressions": ["N3-1", "N4-0"],
        })
        result = prepare_behaviors(df, max_rows=1)
        assert len(result) == 1


class TestComputeNewsFeatures:
    def test_basic_feature_computation(self):
        news_df = pd.DataFrame({
            "NewsID": ["N1", "N2"],
            "Category": ["news", "sports"],
            "SubCategory": ["newsworld", "sportsfootball"],
            "Title": ["Article A", "Article B"],
            "TitleEntities": ["[]", "[]"],
        })
        def dummy_encoder(texts):
            return np.random.randn(len(texts), 3).astype(np.float32)

        result = compute_news_features(news_df, dummy_encoder, None, 5)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "item_id"
        assert len(result) == 2
        assert "I_category" in result.columns
        assert "I_sentiment" in result.columns
        assert "I_title_emb_full" in result.columns
        assert result.loc["N1", "I_category"] == "news"
        assert result.loc["N2", "I_category"] == "sports"


class TestBuildSessionUserFeatures:
    def test_basic(self):
        behaviors = pd.DataFrame({
            "ImpressionID": ["1"],
            "UserID": ["U1"],
            "Time": ["t1"],
            "History": ["N1"],
            "Impressions": ["N1-1"],
            "history_ids": [["N1"]],
            "parsed_impressions": [[("N1", 1)]],
            "impression_count": [1],
        })
        news_features = pd.DataFrame({
            "I_title_emb_full": [np.zeros(3, dtype=np.float32).tolist()],
        }, index=pd.Index(["N1"], name="item_id"))
        result = build_session_user_features(behaviors, news_features, 3)
        assert len(result) == 1
        assert result["user_id"].iloc[0] == "U1"
        assert result["impression_id"].iloc[0] == "1"
