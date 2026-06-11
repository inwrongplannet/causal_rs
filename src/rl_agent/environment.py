import logging

import numpy as np
import gymnasium as gym

from src.gpu_utils import gpu_available, batch_cosine_similarity

logger = logging.getLogger(__name__)

# Lazy import for cupy (only when GPU is available)
_cp = None
if gpu_available():
    try:
        import cupy as _cp
    except ImportError:
        pass


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Uses GPU acceleration via cupy when available, falling back to
    numpy automatically.

    Args:
        a: 1-D vector.
        b: 1-D vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    if _cp is not None:
        try:
            gpu_a = _cp.asarray(a, dtype=_cp.float32)
            gpu_b = _cp.asarray(b, dtype=_cp.float32)
            a_norm = gpu_a / (_cp.linalg.norm(gpu_a) + 1e-10)
            b_norm = gpu_b / (_cp.linalg.norm(gpu_b) + 1e-10)
            return float(_cp.dot(a_norm, b_norm))
        except Exception as exc:
            logger.debug("GPU cosine_similarity failed, falling back: %s", exc)
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def ema_update(current: np.ndarray, new: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average update of a user's history embedding.

    Args:
        current: Current user history embedding.
        new: New item embedding to blend in.
        alpha: Blending factor (0 = no change, 1 = replace).

    Returns:
        Updated embedding: (1 - alpha) * current + alpha * new.
    """
    return (1 - alpha) * current + alpha * new


def compute_session_diversity(history_emb: np.ndarray, session) -> float:
    """Compute initial diversity score for a session.

    Calculates the mean of 1 - cosine_similarity between the user's history
    embedding and all candidate item embeddings in the first impression.
    Batches the computation via GPU matrix multiply when available.

    Args:
        history_emb: User's current history embedding.
        session: Session object with .candidates[0] containing items
                 with .title_emb attributes.

    Returns:
        Diversity score in [0, 1] (higher = more diverse).
    """
    candidates = session.candidates[0]
    if not candidates:
        return 0.5

    # Batch all K candidates into a single GPU matmul call.
    item_matrix = np.vstack([c.title_emb for c in candidates]).astype(np.float32)
    user_matrix = np.tile(history_emb.astype(np.float32), (len(candidates), 1))

    sim = batch_cosine_similarity(user_matrix, item_matrix)
    return float(np.clip(1.0 - np.mean(sim), 0.0, 1.0))


class NewsRecommendEnv(gym.Env):
    """Gymnasium environment for news recommendation with causal diversity.

    The agent observes a 385-dim state (384-dim user history embedding +
    1-dim diversity score) and selects from K candidate articles each step.
    Reward blends click feedback (w) with causal diversity impact (1-w).

    Attributes:
        sessions: List of replay sessions.
        news_df: News features DataFrame indexed by item_id.
        cdi_cache: Dict mapping (user_id, item_id) -> CDI score.
        w: Click reward weight (diversity weight = 1 - w).
        K: Number of candidates per step.
        T: Episode length in steps.
    """

    def __init__(self, sessions, news_df, cdi_cache, w=0.6, K=20, T=10):
        super().__init__()
        self.sessions = sessions
        self.news_df = news_df
        self.cdi_cache = cdi_cache
        self.w = w
        self.K = K
        self.T = T

        sample_sess = sessions[0]
        emb_dim = len(sample_sess.initial_history_emb)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(emb_dim + 1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(K)

    def reset(self, seed=None):
        """Start a new episode by sampling a random session.

        Args:
            seed: Optional RNG seed.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)
        self.session = self.np_random.choice(self.sessions)
        self.step_idx = 0
        self.history_emb = self.session.initial_history_emb.copy()
        self.D = compute_session_diversity(self.history_emb, self.session)
        return self._obs(), {}

    def step(self, action: int):
        """Execute one environment step.

        Args:
            action: Index of the selected candidate article.

        Returns:
            Tuple of (next_obs, reward, terminated, truncated, info).
        """
        candidate = self.session.candidates[self.step_idx][action]
        r_click = self.session.clicks[self.step_idx][action]
        cdi = self.cdi_cache.get(
            (self.session.user_id, candidate.item_id), 0.0
        )
        reward = self.w * r_click + (1 - self.w) * cdi

        if r_click:
            self.history_emb = ema_update(
                self.history_emb, candidate.title_emb
            )
            self.D = 1.0 - cosine_similarity(
                self.history_emb, candidate.title_emb
            )

        self.step_idx += 1
        done = self.step_idx >= self.T
        return self._obs(), reward, done, False, {}

    def _obs(self):
        """Build observation vector from current state."""
        return np.append(self.history_emb, self.D).astype(np.float32)
