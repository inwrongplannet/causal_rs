import numpy as np
import pytest

from src.evaluation.metrics import (
    aggregate_metrics,
    homogeneity_trend,
    ild,
    ndcg_at_k,
    precision_at_k,
    significance_test,
)


class TestNDCG:
    def test_perfect_ranking(self):
        rec = ["N1", "N2", "N3"]
        clicked = {"N1", "N2", "N3"}
        assert ndcg_at_k(rec, clicked, 3) == pytest.approx(1.0)

    def test_no_matches(self):
        rec = ["N1", "N2", "N3"]
        clicked = {"N4"}
        assert ndcg_at_k(rec, clicked, 3) == pytest.approx(0.0)

    def test_partial_match(self):
        rec = ["N1", "N2", "N3", "N4"]
        clicked = {"N2", "N4"}
        ndcg = ndcg_at_k(rec, clicked, 4)
        assert 0.0 < ndcg < 1.0

    def test_empty_clicked(self):
        assert ndcg_at_k(["N1"], set(), 1) == 0.0

    def test_k_larger_than_list(self):
        rec = ["N1"]
        clicked = {"N1"}
        assert ndcg_at_k(rec, clicked, 10) == pytest.approx(1.0)


class TestPrecision:
    def test_perfect(self):
        rec = ["N1", "N2"]
        clicked = {"N1", "N2"}
        assert precision_at_k(rec, clicked, 2) == 1.0

    def test_half(self):
        rec = ["N1", "N2", "N3", "N4"]
        clicked = {"N1", "N3"}
        assert precision_at_k(rec, clicked, 4) == 0.5

    def test_no_matches(self):
        assert precision_at_k(["N1"], {"N2"}, 1) == 0.0


class TestILD:
    def test_identical_embeddings(self):
        embs = np.array([[1.0, 0.0], [1.0, 0.0]])
        assert ild(embs) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert ild(embs) == pytest.approx(1.0)

    def test_single_item(self):
        assert ild(np.array([[1.0, 0.0]])) == 0.0

    def test_three_items(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]])
        val = ild(embs)
        assert 0.0 < val < 1.0


class TestHomogeneityTrend:
    def test_similar_returns_high(self):
        h = homogeneity_trend([np.array([1.0, 0.0])], [np.array([0.99, 0.01])])
        assert h[0] > 0.9

    def test_different_returns_low(self):
        h = homogeneity_trend([np.array([1.0, 0.0])], [np.array([0.0, 1.0])])
        assert h[0] < 0.1

    def test_multiple_steps(self):
        history = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]
        recs = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
        h = homogeneity_trend(history, recs)
        assert len(h) == 2
        assert h[0] > h[1]  # decreasing = healthy


class TestAggregateMetrics:
    def test_single_result(self):
        agg = aggregate_metrics([{"ndcg": 0.5, "precision": 0.3, "ild": 0.7}])
        assert agg["n_sessions"] == 1
        assert agg["ndcg_mean"] == 0.5

    def test_multiple_results(self):
        agg = aggregate_metrics(
            [
                {"ndcg": 0.5, "precision": 0.3, "ild": 0.7},
                {"ndcg": 0.7, "precision": 0.5, "ild": 0.9},
            ]
        )
        assert agg["n_sessions"] == 2
        assert agg["ndcg_mean"] == 0.6
        assert agg["precision_mean"] == 0.4
        assert agg["ild_mean"] == 0.8


class TestSignificanceTest:
    def test_identical_arrays(self):
        assert significance_test([0.5, 0.6], [0.5, 0.6], "test") is False

    def test_different_arrays(self):
        result = significance_test([0.6, 0.7, 0.8], [0.1, 0.2, 0.3], "test")
        assert result is True
