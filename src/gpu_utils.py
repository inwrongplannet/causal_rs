"""GPU-accelerated array operations with automatic numpy fallback.

All public functions accept numpy arrays and return numpy arrays.
If cupy is available and GPU_ENABLED is True, computation runs on GPU
transparently.  Fallback to numpy is automatic when cupy is missing,
GPU_ENABLED is False, or a GPU runtime error occurs.

Typical usage::

    from src.gpu_utils import batch_cosine_diversity, gpu_available

    diversities = batch_cosine_diversity(user_matrix, item_matrix)
"""

import logging
import os
from pathlib import Path

import numpy as np

from src.config import GPU_BATCH_SIZE, GPU_ENABLED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA_PATH auto-detection  (helps cupy find nvrtc when no system-wide CUDA
# Toolkit is installed)
# ---------------------------------------------------------------------------
def _resolve_cuda_path() -> str | None:
    """Return a path to CUDA binaries if available, else None."""
    # Already set by user — honour it.
    if os.environ.get("CUDA_PATH"):
        return os.environ["CUDA_PATH"]
    # Check venv-local nvidia wheels (torch / nvidia-cuda-nvrtc).
    site_pkgs = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
    candidates = [
        site_pkgs / "nvidia" / "cuda_runtime" / "bin",
        site_pkgs / "nvidia" / "cuda_nvrtc" / "bin",
        site_pkgs / "torch" / "lib",
    ]
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("nvrtc*.dll")):
            return str(cand.resolve())
    return None


_cuda_bin = _resolve_cuda_path()
if _cuda_bin and "CUDA_PATH" not in os.environ:
    os.environ["CUDA_PATH"] = _cuda_bin

import warnings

warnings.filterwarnings("ignore", message="CUDA path could not be detected")


# ---------------------------------------------------------------------------
# Cupy availability probe  (lazy — first import triggers the check)
# ---------------------------------------------------------------------------
_HAS_CUPY: bool | None = None
_cp = None


def _probe_cupy() -> bool:
    global _HAS_CUPY, _cp
    if _HAS_CUPY is not None:
        return _HAS_CUPY
    try:
        import cupy as cp
        _cp = cp
        _ = cp.cuda.runtime.getDeviceCount()
        _HAS_CUPY = True
    except Exception:  # noqa: BLE001
        _HAS_CUPY = False
        _cp = None
    return _HAS_CUPY


def gpu_available() -> bool:
    """Return True iff cupy is importable and at least one CUDA device is visible."""
    return _probe_cupy() and bool(GPU_ENABLED)


def _to_array(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)


def _to_gpu(arr: np.ndarray):
    if gpu_available():
        try:
            return _cp.asarray(arr)
        except Exception:  # noqa: BLE001, S110
            pass
    return None


def _from_gpu(arr) -> np.ndarray:
    if hasattr(arr, "get"):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _gpu_memory_info() -> str:
    """Return a short string with current GPU memory usage, or empty if unavailable."""
    try:
        if _cp is not None:
            mem = _cp.cuda.runtime.memGetInfo()
            free_mb = mem[0] / 1024 ** 2
            total_mb = mem[1] / 1024 ** 2
            return f"GPU mem: {free_mb:.0f}/{total_mb:.0f} MiB free"
    except Exception:  # noqa: BLE001, S110  # noqa: BLE001
        pass
    return ""


def batch_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of *matrix* in place.

    Args:
        matrix: 2-D array shaped ``(n_rows, dim)``.

    Returns:
        Normalised array with the same shape.
    """
    n = len(matrix)
    if n == 0:
        return matrix.astype(np.float32)

    results = []
    offset = 0
    while offset < n:
        chunk = matrix[offset:offset + GPU_BATCH_SIZE]
        gpu_chunk = _to_gpu(chunk)
        if gpu_chunk is not None:
            try:
                norms = _cp.linalg.norm(gpu_chunk, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                result_chunk = _from_gpu(gpu_chunk / norms)
                results.append(result_chunk.astype(np.float32))
                offset += GPU_BATCH_SIZE
                continue
            except Exception as exc:  # noqa: BLE001  # noqa: BLE001
                mem_info = _gpu_memory_info()
                logger.warning(
                    "GPU batch_l2_normalize failed on rows %d-%d (%s), "
                    "falling back chunk to CPU: %s",
                    offset, min(offset + GPU_BATCH_SIZE, n), mem_info, exc,
                )

        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        results.append((chunk / norms).astype(np.float32))
        offset += GPU_BATCH_SIZE

    return np.vstack(results)


def batch_cosine_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equal-length matrices.

    Args:
        matrix_a: 2-D array shaped ``(n, dim)``.
        matrix_b: 2-D array shaped ``(n, dim)``.

    Returns:
        1-D array of cosine similarities in ``[-1, 1]``, length *n*.
    """
    n = len(matrix_a)
    if n == 0:
        return np.array([], dtype=np.float32)

    results = np.empty(n, dtype=np.float32)
    offset = 0
    while offset < n:
        chunk_end = min(offset + GPU_BATCH_SIZE, n)
        a_chunk = batch_l2_normalize(matrix_a[offset:chunk_end])
        b_chunk = batch_l2_normalize(matrix_b[offset:chunk_end])

        gpu_a = _to_gpu(a_chunk)
        gpu_b = _to_gpu(b_chunk)
        if gpu_a is not None and gpu_b is not None:
            try:
                sim = _cp.sum(gpu_a * gpu_b, axis=1)
                results[offset:chunk_end] = _from_gpu(sim)
                offset = chunk_end
                continue
            except Exception as exc:  # noqa: BLE001  # noqa: BLE001
                mem_info = _gpu_memory_info()
                logger.warning(
                    "GPU batch_cosine_similarity failed on rows %d-%d (%s), "
                    "falling back chunk to CPU: %s",
                    offset, chunk_end, mem_info, exc,
                )

        results[offset:chunk_end] = np.sum(a_chunk * b_chunk, axis=1)
        offset = chunk_end

    return results


def batch_cosine_diversity(
    user_matrix: np.ndarray,
    item_matrix: np.ndarray,
) -> np.ndarray:
    """Compute ``1 - cosine_similarity`` row-wise for two matrices.

    Args:
        user_matrix: 2-D array shaped ``(n, dim)`` — user history embeddings.
        item_matrix: 2-D array shaped ``(n, dim)`` — candidate item embeddings.

    Returns:
        1-D array of diversity scores in ``[0, 1]``, length *n*.
    """
    sim = batch_cosine_similarity(user_matrix, item_matrix)
    return np.clip(1.0 - sim, 0.0, 1.0)


def pairwise_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for all rows in *matrix*.

    Args:
        matrix: 2-D array shaped ``(n, dim)``.

    Returns:
        Symmetric 2-D array shaped ``(n, n)`` with entries in ``[-1, 1]``.
    """
    normed = batch_l2_normalize(matrix)

    gpu_arr = _to_gpu(normed)
    if gpu_arr is not None:
        try:
            sim = _cp.dot(gpu_arr, gpu_arr.T)
            return _from_gpu(sim)
        except Exception as exc:  # noqa: BLE001  # noqa: BLE001
            logger.warning("GPU pairwise_cosine_similarity failed, falling back: %s", exc)

    return np.dot(normed, normed.T)
