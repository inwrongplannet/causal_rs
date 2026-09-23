"""Tests for src.evaluation.metrics.replay_evaluate — previously untested
and containing a missing-import bug (fixed in Ticket 1.1 of the
research-ready plan). These tests exercise the real function end-to-end
against a tiny synthetic session and a tiny real PPO model (trained for a
handful of timesteps), rather than mocking internal SB3 APIs.
"""
import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.evaluation.metrics import replay_evaluate
from src.rl_agent.environment import NewsRecommendEnv


def _make_candidate(item_id, emb):
    return type(
        "Candidate", (), {"item_id": item_id, "title_emb": np.array(emb, dtype=np.float32)}
    )()


def _make_session():
    candidates = [
        _make_candidate("N1", [1.0, 0.0, 0.0]),
        _make_candidate("N2", [0.0, 1.0, 0.0]),
        _make_candidate("N3", [0.0, 0.0, 1.0]),
    ]
    return type(
        "Session",
        (),
        {
            "user_id": "U1",
            "initial_history_emb": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "candidates": [candidates],
            "clicks": [[1, 0, 0]],
            "clicked_items": {"N1"},
        },
    )()


def _make_news_df():
    return pd.DataFrame(
        {
            "I_title_emb_full": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        },
        index=["N1", "N2", "N3"],
    )


def _train_tiny_policy(session):
    vec_env = DummyVecEnv(
        [lambda: NewsRecommendEnv([session], pd.DataFrame(), {}, w=0.6, K=3, T=1)]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        verbose=0,
        seed=42,
    )
    model.learn(total_timesteps=64)
    return model


class TestReplayEvaluateWrongArgument:
    def test_raises_typeerror_when_passed_model_instead_of_model_policy(self):
        session = _make_session()
        news_df = _make_news_df()
        model = _train_tiny_policy(session)
        with pytest.raises(TypeError):
            replay_evaluate(model, [session], news_df, {}, w=0.6, K=3, T=1)


class TestReplayEvaluateCorrectUsage:
    def test_runs_without_raising_and_returns_expected_keys(self):
        session = _make_session()
        news_df = _make_news_df()
        model = _train_tiny_policy(session)

        result = replay_evaluate(model.policy, [session], news_df, {}, w=0.6, K=3, T=1)

        assert isinstance(result, dict)
        for key in (
            "ndcg_mean", "ndcg_std",
            "precision_mean", "precision_std",
            "ild_mean", "ild_std",
            "n_sessions",
        ):
            assert key in result
        assert result["n_sessions"] == 1
