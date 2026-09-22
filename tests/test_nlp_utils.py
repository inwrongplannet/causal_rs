import numpy as np
import pytest
from src.data_pipeline.nlp_utils import (
    l2_normalize,
    cosine_diversity,
    stable_hash_vector,
    mean_embeddings,
    score_sentiment,
    batch_l2_normalize,
    batch_cosine_diversity,
)


class TestL2Normalize:
    def test_normal_vector(self):
        v = np.array([3.0, 4.0])
        result = l2_normalize(v)
        assert np.allclose(result, np.array([0.6, 0.8], dtype=np.float32))

    def test_zero_vector(self):
        v = np.zeros(5)
        result = l2_normalize(v)
        assert np.allclose(result, np.zeros(5, dtype=np.float32))

    def test_nan_vector(self):
        v = np.array([np.nan, 1.0])
        result = l2_normalize(v)
        assert np.allclose(result, np.zeros(2, dtype=np.float32))

    def test_inf_vector(self):
        v = np.array([np.inf, 1.0])
        result = l2_normalize(v)
        assert np.allclose(result, np.zeros(2, dtype=np.float32))

    def test_output_dtype(self):
        v = np.array([1.0, 2.0, 3.0])
        result = l2_normalize(v)
        assert result.dtype == np.float32


class TestCosineDiversity:
    def test_orthogonal_vectors(self):
        u = np.array([1.0, 0.0])
        i = np.array([0.0, 1.0])
        d = cosine_diversity(u, i)
        assert d == pytest.approx(1.0)

    def test_same_vector(self):
        u = np.array([1.0, 2.0, 3.0])
        i = np.array([1.0, 2.0, 3.0])
        d = cosine_diversity(u, i)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vector(self):
        u = np.array([1.0, 0.0])
        i = np.array([-1.0, 0.0])
        d = cosine_diversity(u, i)
        assert d == pytest.approx(1.0)  # clipped to [0, 1]

    def test_empty_user_emb(self):
        d = cosine_diversity(np.array([]), np.array([1.0, 0.0]))
        assert d == 0.0

    def test_empty_item_emb(self):
        d = cosine_diversity(np.array([1.0, 0.0]), np.array([]))
        assert d == 0.0

    def test_clip_range(self):
        u = np.array([1.0, 0.0])
        i = np.array([-1.0, 0.0])
        d = cosine_diversity(u, i)
        assert 0.0 <= d <= 1.0

    def test_output_type(self):
        u = np.array([1.0, 2.0])
        i = np.array([3.0, 4.0])
        assert isinstance(cosine_diversity(u, i), float)


class TestStableHashVector:
    def test_output_shape(self):
        vec = stable_hash_vector("test", 100)
        assert vec.shape == (100,)
        assert vec.dtype == np.float32

    def test_deterministic(self):
        a = stable_hash_vector("hello", 100)
        b = stable_hash_vector("hello", 100)
        assert np.allclose(a, b)

    def test_different_inputs_different(self):
        a = stable_hash_vector("hello", 100)
        b = stable_hash_vector("world", 100)
        assert not np.allclose(a, b)

    def test_normalized(self):
        vec = stable_hash_vector("test", 100)
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_dimension_tiling(self):
        vec = stable_hash_vector("a", 1000)
        assert vec.shape == (1000,)


class TestMeanEmbeddings:
    def test_single_vector(self):
        v = [np.array([1.0, 0.0], dtype=np.float32)]
        result = mean_embeddings(v, 2)
        assert np.allclose(result, l2_normalize(np.array([1.0, 0.0])))

    def test_multiple_vectors(self):
        v = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        result = mean_embeddings(v, 2)
        expected = l2_normalize(np.array([0.5, 0.5]))
        assert np.allclose(result, expected)

    def test_empty_list(self):
        result = mean_embeddings([], 10)
        assert np.allclose(result, np.zeros(10, dtype=np.float32))

    def test_output_dtype(self):
        v = [np.array([1.0, 0.0], dtype=np.float32)]
        result = mean_embeddings(v, 2)
        assert result.dtype == np.float32


class TestScoreSentiment:
    def test_returns_zero_when_analyzer_none(self):
        assert score_sentiment("hello world", None) == 0.0

    def test_returns_zero_on_empty_string(self):
        assert score_sentiment("", None) == 0.0


class TestBatchL2Normalize:
    def test_matches_per_row(self):
        matrix = np.random.randn(8, 64).astype(np.float32)
        result = batch_l2_normalize(matrix)
        for i in range(8):
            expected = l2_normalize(matrix[i])
            np.testing.assert_allclose(result[i], expected, rtol=1e-5)

    def test_zero_row(self):
        matrix = np.zeros((3, 16), dtype=np.float32)
        result = batch_l2_normalize(matrix)
        assert np.allclose(result, np.zeros_like(matrix))


class TestBatchCosineDiversity:
    def test_matches_per_element(self):
        n, dim = 6, 32
        user_mat = np.random.randn(n, dim).astype(np.float32)
        item_mat = np.random.randn(n, dim).astype(np.float32)
        batched = batch_cosine_diversity(user_mat, item_mat)
        for i in range(n):
            expected = cosine_diversity(user_mat[i], item_mat[i])
            assert abs(batched[i] - expected) < 1e-5

    def test_range(self):
        matrix = np.random.randn(10, 64).astype(np.float32)
        div = batch_cosine_diversity(matrix, matrix)
        assert div.min() >= 0.0
        assert div.max() <= 1.0
