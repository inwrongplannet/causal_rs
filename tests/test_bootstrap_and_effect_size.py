import pytest

from src.evaluation.metrics import bootstrap_ci, cohens_d_label


class TestCohensDLabel:
    def test_negligible(self):
        assert cohens_d_label(0.05) == "negligible"

    def test_negligible_negative(self):
        assert cohens_d_label(-0.1) == "negligible"

    def test_small(self):
        assert cohens_d_label(0.3) == "small"

    def test_medium(self):
        assert cohens_d_label(0.6) == "medium"

    def test_large(self):
        assert cohens_d_label(1.2) == "large"

    def test_boundary_0_2_is_small(self):
        assert cohens_d_label(0.2) == "small"

    def test_boundary_0_8_is_large(self):
        assert cohens_d_label(0.8) == "large"


class TestBootstrapCI:
    def test_returns_expected_keys(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(values, n_boot=500, seed=1)
        for key in ("mean", "ci_low", "ci_high", "excludes_zero"):
            assert key in result

    def test_mean_is_correct(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(values, n_boot=500, seed=1)
        assert result["mean"] == pytest.approx(3.0)

    def test_ci_low_le_ci_high(self):
        values = [0.1, 0.2, 0.15, 0.18, 0.22, 0.19]
        result = bootstrap_ci(values, n_boot=1000, seed=1)
        assert result["ci_low"] <= result["ci_high"]

    def test_excludes_zero_when_all_positive(self):
        values = [5.0, 6.0, 5.5, 6.5, 5.8]
        result = bootstrap_ci(values, n_boot=1000, seed=1)
        assert result["excludes_zero"] is True

    def test_does_not_exclude_zero_when_centered_on_zero(self):
        values = [-1.0, 1.0, -0.5, 0.5, 0.1, -0.1, 0.0]
        result = bootstrap_ci(values, n_boot=2000, seed=1)
        assert result["excludes_zero"] is False

    def test_deterministic_with_same_seed(self):
        values = [1.0, 2.0, 3.0, 2.5, 1.5]
        r1 = bootstrap_ci(values, n_boot=500, seed=7)
        r2 = bootstrap_ci(values, n_boot=500, seed=7)
        assert r1 == r2
