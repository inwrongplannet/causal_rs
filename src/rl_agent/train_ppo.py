from pathlib import Path

import numpy as np
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.rl_agent.environment import NewsRecommendEnv


def make_env(sessions, news_df, cdi_cache, w=0.6, K=20, T=10):
    """Factory for creating NewsRecommendEnv instances (for SubprocVecEnv).

    Args:
        sessions: List of replay sessions.
        news_df: News features DataFrame indexed by item_id.
        cdi_cache: Dict mapping (user_id, item_id) -> CDI score.
        w: Click reward weight.
        K: Number of candidates per step.
        T: Episode length in steps.

    Returns:
        A zero-argument callable that returns a NewsRecommendEnv.
    """
    return lambda: NewsRecommendEnv(sessions, news_df, cdi_cache, w=w, K=K, T=T)


def _create_vec_env(env_fns, n_envs):
    """Create a vectorised environment, using DummyVecEnv on Windows."""
    if sys.platform == "win32":
        return DummyVecEnv(env_fns[:1])
    return SubprocVecEnv(env_fns[:n_envs])


def train_ppo(
    train_sessions,
    news_df,
    cdi_cache,
    total_timesteps=5_000_000,
    n_envs=8,
    w=0.6,
    K=20,
    T=10,
    learning_rate=3e-4,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    net_arch=None,
    tensorboard_log="./tb_logs/",
    checkpoint_dir="./checkpoints/",
    verbose=1,
):
    """Train a PPO agent for news recommendation with causal diversity.

    Sets up parallel environments via SubprocVecEnv and trains a stable-
    baselines3 PPO model with the provided hyperparameters. Saves the
    trained model to a checkpoint file.

    Args:
        train_sessions: List of training session objects.
        news_df: News features DataFrame indexed by item_id.
        cdi_cache: Dict mapping (user_id, item_id) -> CDI score.
        total_timesteps: Total environment steps for training.
        n_envs: Number of parallel environments.
        w: Click reward weight (default 0.6).
        K: Number of candidates per step.
        T: Episode length in steps.
        learning_rate: PPO learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping epsilon.
        ent_coef: Entropy coefficient for exploration.
        n_steps: Rollout buffer size per environment.
        batch_size: Minibatch size for gradient updates.
        n_epochs: PPO epochs per rollout.
        net_arch: Policy network architecture [shared_layers].
        tensorboard_log: TensorBoard log directory.
        checkpoint_dir: Directory to save model checkpoints.
        verbose: Verbosity level (0 = silent, 1 = info).

    Returns:
        Tuple of (trained PPO model, checkpoint save path).
    """
    if net_arch is None:
        net_arch = [256, 128]

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    vec_env = _create_vec_env(
        [
            make_env(train_sessions, news_df, cdi_cache, w=w, K=K, T=T)
            for _ in range(n_envs)
        ],
        n_envs,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    save_path = str(checkpoint_path / f"ppo_causal_rs_w{int(w*10):02d}")
    model.save(save_path)
    vec_env.close()

    return model, save_path
