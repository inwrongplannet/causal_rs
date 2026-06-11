"""Tests for src.gpu_utils — run correctly with or without a GPU."""

import numpy as np
import pytest

from src.gpu_utils import (
    batch_l2_normalize,
    batch_cosine_similarity,
    batch_cosine_diversity,
    pairwise_cosine_similarity,
    gpu_available,
)


class TestBatchL2Normalize:
    def test_identical_to_per_row_norm(self):
        matrix = np.random.randn(10, 768).astype(np.float32)
        batched = batch_l2_normalize(matrix)
        for i in range(10):
            expected = matrix[i] / (np.linalg.norm(matrix[i]) + 1e-10)
            np.testing.assert_allclose(batched[i], expected, rtol=1e-5)

    def test_zero_row(self):
        matrix = np.zeros((3, 768), dtype=np.float32)
        result = batch_l2_normalize(matrix)
        np.testing.assert_allclose(result, np.zeros_like(matrix))

    def test_output_dtype(self):
        matrix = np.random.randn(5, 64).astype(np.float64)
        result = batch_l2_normalize(matrix)
        assert result.dtype == np.float32

    def test_single_row(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        result = batch_l2_normalize(vec[None, :])
        expected = np.array([[0.6, 0.8]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestBatchCosineSimilarity:
    def test_identical_vectors(self):
        matrix = np.random.randn(5, 768).astype(np.float32)
        sims = batch_cosine_similarity(matrix, matrix)
        np.testing.assert_allclose(sims, np.ones(5), rtol=1e-4)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        sims = batch_cosine_similarity(a, b)
        np.testing.assert_allclose(sims, [0.0], atol=1e-6)

    def test_opposite_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        sims = batch_cosine_similarity(a, b)
        np.testing.assert_allclose(sims, [-1.0], atol=1e-6)

    def test_matches_per_element_dot(self):
        n, dim = 8, 128
        a = np.random.randn(n, dim).astype(np.float32)
        b = np.random.randn(n, dim).astype(np.float32)
        batched = batch_cosine_similarity(a, b)
        for i in range(n):
            a_norm = a[i] / (np.linalg.norm(a[i]) + 1e-10)
            b_norm = b[i] / (np.linalg.norm(b[i]) + 1e-10)
            expected = float(np.dot(a_norm, b_norm))
            assert abs(batched[i] - expected) < 1e-5


class TestBatchCosineDiversity:
    def test_range(self):
        n, dim = 10, 768
        user_mat = np.random.randn(n, dim).astype(np.float32)
        item_mat = np.random.randn(n, dim).astype(np.float32)
        diversities = batch_cosine_diversity(user_mat, item_mat)
        assert diversities.min() >= 0.0
        assert diversities.max() <= 1.0

    def test_identical_vectors_zero_diversity(self):
        matrix = np.ones((4, 64), dtype=np.float32)
        div = batch_cosine_diversity(matrix, matrix)
        np.testing.assert_allclose(div, np.zeros(4), atol=1e-5)

    def test_matches_per_element(self):
        n, dim = 6, 32
        user_mat = np.random.randn(n, dim).astype(np.float32)
        item_mat = np.random.randn(n, dim).astype(np.float32)
        batched = batch_cosine_diversity(user_mat, item_mat)
        for i in range(n):
            u = user_mat[i] / (np.linalg.norm(user_mat[i]) + 1e-10)
            v = item_mat[i] / (np.linalg.norm(item_mat[i]) + 1e-10)
            expected = float(np.clip(1.0 - np.dot(u, v), 0.0, 1.0))
            assert abs(batched[i] - expected) < 1e-5

    def test_output_type(self):
        matrix = np.random.randn(3, 16).astype(np.float32)
        result = batch_cosine_diversity(matrix, matrix)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64 or result.dtype == np.float32


class TestPairwiseCosineSimilarity:
    def test_identity_matrix(self):
        matrix = np.eye(5, dtype=np.float32)
        sim = pairwise_cosine_similarity(matrix)
        np.testing.assert_allclose(sim, matrix, atol=1e-5)

    def test_symmetric(self):
        matrix = np.random.randn(6, 32).astype(np.float32)
        sim = pairwise_cosine_similarity(matrix)
        assert np.allclose(sim, sim.T, rtol=1e-5)

    def test_diagonal_is_one(self):
        matrix = np.random.randn(5, 64).astype(np.float32)
        sim = pairwise_cosine_similarity(matrix)
        np.testing.assert_allclose(np.diag(sim), np.ones(5), rtol=1e-5)

    def test_single_row(self):
        vec = np.array([[3.0, 4.0]], dtype=np.float32)
        sim = pairwise_cosine_similarity(vec)
        assert sim.shape == (1, 1)
        assert abs(sim[0, 0] - 1.0) < 1e-5


class TestGpuAvailable:
    def test_returns_bool(self):
        result = gpu_available()
        assert isinstance(result, bool)
