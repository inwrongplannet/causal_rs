import numpy as np
import pandas as pd

from src.baselines.logistic_cf import (
    build_feature_matrix,
    train_logistic_cf,
    score_candidates,
)


def _make_fake_scm_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "U_pca_0": rng.normal(size=n),
        "U_pca_1": rng.normal(size=n),
        "I_entity_pca_0": rng.normal(size=n),
        "I_title_pca_0": rng.normal(size=n),
        "U_dwell_mean": rng.normal(size=n),
        "I_sentiment": rng.uniform(-1, 1, size=n),
        "I_category": rng.choice(["sports", "news", "finance"], size=n),
    })
    # Y_click is weakly dependent on I_sentiment so the model has signal to learn.
    prob = 1 / (1 + np.exp(-df["I_sentiment"].to_numpy()))
    df["Y_click"] = rng.binomial(1, prob)
    return df


class TestBuildFeatureMatrix:
    def test_returns_matrix_and_columns(self):
        df = _make_fake_scm_df()
        X, columns = build_feature_matrix(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(columns)
        assert not np.isnan(X).any()

    def test_test_set_reindexes_to_train_columns(self):
        train_df = _make_fake_scm_df(n=100, seed=1)
        test_df = _make_fake_scm_df(n=50, seed=2)
        _, train_columns = build_feature_matrix(train_df)
        X_test, test_columns = build_feature_matrix(test_df, fit_columns=train_columns)
        assert test_columns == train_columns
        assert X_test.shape[1] == len(train_columns)


class TestTrainAndScore:
    def test_train_returns_model_and_columns(self):
        df = _make_fake_scm_df()
        fitted = train_logistic_cf(df, seed=42)
        assert "model" in fitted
        assert "columns" in fitted

    def test_score_candidates_returns_probabilities_in_unit_interval(self):
        train_df = _make_fake_scm_df(n=200, seed=1)
        test_df = _make_fake_scm_df(n=50, seed=2)
        fitted = train_logistic_cf(train_df, seed=42)
        scores = score_candidates(fitted, test_df)
        assert len(scores) == len(test_df)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_deterministic_with_same_seed(self):
        train_df = _make_fake_scm_df(n=200, seed=1)
        test_df = _make_fake_scm_df(n=50, seed=2)
        fitted_a = train_logistic_cf(train_df, seed=42)
        fitted_b = train_logistic_cf(train_df, seed=42)
        scores_a = score_candidates(fitted_a, test_df)
        scores_b = score_candidates(fitted_b, test_df)
        np.testing.assert_array_almost_equal(scores_a, scores_b)
