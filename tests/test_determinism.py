"""Verifies that PPO training is reproducible given a fixed seed — this is
a prerequisite for Phase 4's multi-seed statistical analysis to be
meaningful (if training weren't seed-reproducible, "same seed, different
result" would silently corrupt every downstream statistic).
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
        },
    )()


def _train_once(seed):
    session = _make_session()
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
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=64)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    return action


class TestDeterminism:
    def test_same_seed_gives_same_action(self):
        action_a = _train_once(seed=123)
        action_b = _train_once(seed=123)
        np.testing.assert_array_equal(action_a, action_b)
