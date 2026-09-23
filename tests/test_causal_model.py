"""Tests for src.causal_model — previously 0% test coverage despite being
the module implementing this project's central causal-inference claim.

These tests use SYNTHETIC data with a KNOWN true average treatment effect
(ATE), so a passing test is evidence the estimator is actually correct,
not just that it runs without crashing.
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no display needed for tests

import numpy as np
import pandas as pd
import pytest
from dowhy import CausalModel

from src.causal_model.model import (
    create_causal_model,
    identify_effect,
    estimate_ate_ipw,
    estimate_ate_linear,
    positivity_check,
)
from src.causal_model.refutation import run_refutations


def _make_synthetic_confounded_data(n=2000, true_ate=2.0, seed=42):
    """Generate data with a KNOWN true ATE for validating the estimator.

    W is a confounder affecting both treatment assignment and outcome.
    Y = true_ate * A + W + noise, so a correctly backdoor-adjusted
    estimator should recover an ATE close to `true_ate`.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, size=n)
    propensity = 1 / (1 + np.exp(-W))
    A = rng.binomial(1, propensity)
    noise = rng.normal(0, 0.5, size=n)
    Y = true_ate * A + W + noise
    return pd.DataFrame({"W": W, "A": A, "Y": Y})


@pytest.fixture(scope="module")
def synthetic_df():
    return _make_synthetic_confounded_data()


class TestCreateCausalModel:
    def test_returns_causal_model_instance(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        assert isinstance(model, CausalModel)


class TestIdentifyEffect:
    def test_returns_nonnull_estimand(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        assert estimand is not None


class TestEstimateATE:
    def test_ipw_recovers_true_ate_within_tolerance(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_ipw(model, estimand)
        # Generous tolerance (abs=0.3) because IPW on n=2000 synthetic
        # samples still has real estimation noise — this is checking
        # correctness of direction and rough magnitude, not exactness.
        assert estimate.value == pytest.approx(2.0, abs=0.3)

    def test_linear_recovers_true_ate_within_tolerance(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_linear(model, estimand)
        assert estimate.value == pytest.approx(2.0, abs=0.3)


class TestPositivityCheck:
    def test_returns_scores_in_unit_interval(self, synthetic_df):
        ps = positivity_check(synthetic_df, common_causes=["W"], treatment="A")
        assert (ps >= 0.0).all() and (ps <= 1.0).all()


class TestRefutations:
    def test_all_three_refutations_run_without_raising(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_linear(model, estimand)
        results = run_refutations(model, estimand, estimate, num_simulations=5)
        assert "placebo" in results
        assert "subset" in results
        assert "random_common_cause" in results

    def test_placebo_effect_is_near_zero(self, synthetic_df):
        model = create_causal_model(
            synthetic_df, treatment="A", outcome="Y", common_causes=["W"]
        )
        estimand = identify_effect(model)
        estimate = estimate_ate_linear(model, estimand)
        results = run_refutations(model, estimand, estimate, num_simulations=5)
        if results["placebo"] is not None:
            assert abs(results["placebo"].new_effect) < 0.5
