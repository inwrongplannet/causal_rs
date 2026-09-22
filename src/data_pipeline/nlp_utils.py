import logging

import numpy as np

from src.gpu_utils import batch_l2_normalize as _batch_l2_normalize
from src.gpu_utils import batch_cosine_diversity as _batch_cosine_diversity
from src.gpu_utils import gpu_available

logger = logging.getLogger(__name__)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0 or not np.isfinite(norm):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def cosine_diversity(user_emb: np.ndarray, item_emb: np.ndarray) -> float:
    u = l2_normalize(user_emb)
    i = l2_normalize(item_emb)
    if u.size == 0 or i.size == 0:
        return 0.0
    cosine_sim = float(np.dot(u, i))
    if not np.isfinite(cosine_sim):
        cosine_sim = 0.0
    return float(np.clip(1.0 - cosine_sim, 0.0, 1.0))


def stable_hash_vector(text: str, dim: int) -> np.ndarray:
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    base = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    repeats = int(np.ceil(dim / base.size))
    vec = np.tile(base, repeats)[:dim]
    vec = (vec - 127.5) / 127.5
    return l2_normalize(vec)


def mean_embeddings(vectors: list[np.ndarray], dim: int) -> np.ndarray:
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    matrix = np.vstack([np.asarray(v, dtype=np.float32) for v in vectors])
    return l2_normalize(matrix.mean(axis=0))


def build_sentiment_analyzer():
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as exc:
        print(f"Sentiment analyzer unavailable, defaulting to 0.0 sentiment. Reason: {exc}")
        return None


def score_sentiment(text: str, analyzer) -> float:
    if analyzer is None:
        return 0.0
    try:
        return float(analyzer.polarity_scores(text if isinstance(text, str) else "")["compound"])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Batched GPU-accelerated variants  (thin wrappers over src.gpu_utils)
# ---------------------------------------------------------------------------

def batch_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of *matrix* using GPU (cupy) when available.

    Args:
        matrix: 2-D array shaped ``(n_rows, dim)``.

    Returns:
        Normalised array with the same shape, dtype float32.
    """
    return _batch_l2_normalize(matrix)


def batch_cosine_diversity(
    user_matrix: np.ndarray,
    item_matrix: np.ndarray,
) -> np.ndarray:
    """Row-wise ``1 - cosine_similarity`` computed on GPU when available.

    Args:
        user_matrix: 2-D array shaped ``(n, dim)``.
        item_matrix: 2-D array shaped ``(n, dim)``.

    Returns:
        1-D diversity array of length *n*.
    """
    return _batch_cosine_diversity(user_matrix, item_matrix)
