import numpy as np
import pytest

from src.rl_agent.environment import (
    compute_session_diversity,
    cosine_similarity,
    ema_update,
)


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = cosine_similarity(a, b)
        assert not np.isnan(sim)


class TestEMAUpdate:
    def test_alpha_one(self):
        current = np.array([1.0, 2.0])
        new = np.array([3.0, 4.0])
        result = ema_update(current, new, alpha=1.0)
        np.testing.assert_array_almost_equal(result, new)

    def test_alpha_zero(self):
        current = np.array([1.0, 2.0])
        new = np.array([3.0, 4.0])
        result = ema_update(current, new, alpha=0.0)
        np.testing.assert_array_almost_equal(result, current)

    def test_default_alpha(self):
        current = np.array([1.0, 0.0])
        new = np.array([0.0, 1.0])
        result = ema_update(current, new)
        expected = 0.9 * current + 0.1 * new
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeSessionDiversity:
    def test_returns_float(self):
        history = np.array([1.0, 0.0, 0.0])
        session = type(
            "Session",
            (),
            {"candidates": [[type("C", (), {"title_emb": np.array([0.0, 1.0, 0.0])})()]]},
        )()
        d = compute_session_diversity(history, session)
        assert isinstance(d, float)
        assert 0.0 <= d <= 1.0

    def test_empty_candidates_default(self):
        history = np.array([1.0, 0.0])
        session = type("Session", (), {"candidates": [[]]})()
        d = compute_session_diversity(history, session)
        assert d == 0.5
