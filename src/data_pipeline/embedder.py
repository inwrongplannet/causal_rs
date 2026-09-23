import importlib
import logging
from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from src.config import GPU_DEVICE
from src.gpu_utils import gpu_available

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    """Return ``'cuda:N'`` if GPU is available, else ``'cpu'``."""
    if gpu_available():
        return f"cuda:{GPU_DEVICE}"
    return "cpu"


def build_title_encoder(model_name: str, expected_dim: int) -> tuple[Callable, dict]:
    try:
        st_module = importlib.import_module("sentence_transformers")
        sentence_transformer_cls = st_module.SentenceTransformer
        device = _resolve_device()
        model = sentence_transformer_cls(model_name, device=device)
        logger.info("Title encoder running on %s", device)

        def encode(texts: list[str], batch_size: int = 256) -> np.ndarray:
            embeddings = model.encode(
                texts, batch_size=batch_size, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            if embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Expected {expected_dim}-dim title embeddings, got {embeddings.shape[1]} from {model_name}."
                )
            return embeddings.astype(np.float32)

        return encode, {
            "encoder_type": "sentence_transformers",
            "model": model_name,
            "embedding_dim": str(expected_dim),
        }
    except Exception as exc:  # noqa: BLE001
        vectorizer = HashingVectorizer(n_features=expected_dim, alternate_sign=False, norm=None)

        def encode(texts: list[str]) -> np.ndarray:
            sparse_matrix = vectorizer.transform([
                text if isinstance(text, str) else "" for text in texts
            ]).astype(np.float32)
            dense = sparse_matrix.toarray()
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (dense / norms).astype(np.float32)

        return encode, {
            "encoder_type": "hashing_vectorizer_fallback",
            "model": "hashing",
            "embedding_dim": str(expected_dim),
            "fallback_reason": str(exc),
        }
