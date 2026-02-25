import math
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import kv

from ...utils import CUPY_AVAILABLE, PARALLEL_SETTINGS

try:
    import cupy as cp
except ImportError:
    cp = None


# ---------------------------------------------------------------------------
# Shared distance computation
# ---------------------------------------------------------------------------


def _compute_sq_dist(X1: np.ndarray, X2: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance: ||x-y||^2_S with diagonal S = diag(sigma^2)."""
    X1_norm = X1 / sigma
    X2_norm = X2 / sigma
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T
    result: np.ndarray = X1_sq + X2_sq - 2 * inner_prod
    return result


# ---------------------------------------------------------------------------
# Kernel formulas (pure functions: sq_dist -> kernel matrix)
# These use `xp` (array module) so they work with both numpy and cupy arrays.
# ---------------------------------------------------------------------------


def _get_xp(arr: Any) -> Any:
    """Return cupy if arr is a cupy array, else numpy."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def _rbf_formula(sq_dist: Any, alpha: Optional[float], xp: Any = np) -> Any:
    return xp.exp(-0.5 * sq_dist)


def _rq_formula(sq_dist: Any, alpha: Optional[float], xp: Any = np) -> Any:
    if alpha is None:
        raise ValueError("alpha is required for RQ kernel")
    return (1 + 0.5 * sq_dist / alpha) ** (-alpha)


def _matern_formula(sq_dist: Any, alpha: Optional[float], xp: Any = np) -> Any:
    if alpha is None:
        raise ValueError("alpha (nu) is required for MATERN kernel")
    dist = xp.sqrt(xp.maximum(sq_dist, 0))

    if alpha == 0.5:
        return xp.exp(-dist)

    elif alpha == 1.5:
        sqrt3_dist = xp.sqrt(3) * dist
        return (1 + sqrt3_dist) * xp.exp(-sqrt3_dist)

    elif alpha == 2.5:
        sqrt5_dist = xp.sqrt(5) * dist
        return (1 + sqrt5_dist + 5 * dist**2 / 3) * xp.exp(-sqrt5_dist)

    else:
        # General Matern uses scipy.special.kv — CPU-only
        if xp is not np:
            raise ValueError(
                f"Matern with alpha={alpha} only supports 0.5, 1.5, 2.5 on GPU"
            )
        dist_safe = np.where(dist < 1e-10, 1e-10, dist)
        return (
            (2 ** (1 - alpha) / math.gamma(alpha))
            * (np.sqrt(2 * alpha) * dist_safe) ** alpha
            * kv(alpha, np.sqrt(2 * alpha) * dist_safe)
        )


KERNEL_FORMULAS = {
    "RBF": _rbf_formula,
    "RQ": _rq_formula,
    "MATERN": _matern_formula,
}


# ---------------------------------------------------------------------------
# GPU kernel (one function for all types)
# ---------------------------------------------------------------------------


def _gpu_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    kernel_type: str,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """Compute kernel entirely on GPU, transfer only the final result back."""
    if not CUPY_AVAILABLE or cp is None:
        raise RuntimeError("GPU acceleration not available. Install cupy.")

    try:
        X1_gpu = cp.asarray(X1)
        X2_gpu = cp.asarray(X2)
        sigma_gpu = cp.asarray(sigma)

        X1_norm = X1_gpu / sigma_gpu
        X2_norm = X2_gpu / sigma_gpu
        X1_sq = cp.sum(X1_norm**2, axis=1, keepdims=True)
        X2_sq = cp.sum(X2_norm**2, axis=1)
        sq_dist = X1_sq + X2_sq - 2 * (X1_norm @ X2_norm.T)

        formula = KERNEL_FORMULAS[kernel_type]
        result = formula(sq_dist, alpha, xp=cp)
        return np.asarray(cp.asnumpy(result))

    except Exception as e:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            warnings.warn("Failed to free GPU memory pool during error cleanup")
        raise RuntimeError(f"GPU computation failed: {e}")


# ---------------------------------------------------------------------------
# Parallel kernel (one function for all types)
# ---------------------------------------------------------------------------


def _get_n_jobs() -> int:
    n_jobs = PARALLEL_SETTINGS["n_jobs"]
    if n_jobs is not None and not isinstance(n_jobs, bool):
        return int(n_jobs)
    return min(mp.cpu_count(), 8)


def _should_use_parallel(n1: int, n2: int) -> bool:
    if not PARALLEL_SETTINGS["use_parallel"]:
        return False
    min_size = PARALLEL_SETTINGS["min_size_for_parallel"]
    if min_size is None:
        min_size = 500
    return n1 * n2 >= int(min_size) * int(min_size)


def _chunk_indices(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + int(chunk_size), n)) for i in range(0, n, int(chunk_size))]


def _compute_chunk(
    args: Tuple[np.ndarray, np.ndarray, np.ndarray, str, Optional[float]],
) -> np.ndarray:
    X1_chunk, X2, sigma, kernel_type, alpha = args
    formula = KERNEL_FORMULAS[kernel_type]
    sq_dist = _compute_sq_dist(X1_chunk, X2, sigma)
    return np.asarray(formula(sq_dist, alpha))


def _parallel_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    kernel_type: str,
    alpha: Optional[float] = None,
) -> np.ndarray:
    n1 = X1.shape[0]
    chunk_size_raw = PARALLEL_SETTINGS["chunk_size"]
    chunk_size = (
        int(chunk_size_raw)
        if chunk_size_raw is not None and not isinstance(chunk_size_raw, bool)
        else 1000
    )

    chunks = [
        (X1[i_start:i_end], X2, sigma, kernel_type, alpha)
        for i_start, i_end in _chunk_indices(n1, chunk_size)
    ]

    n_jobs = _get_n_jobs()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_compute_chunk, chunks))

    return np.vstack(results)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def compute_kernel(
    kernel_type: str,
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Compute kernel matrix K[i,j] = k(X1[i], X2[j]).

    Parameters
    ----------
    kernel_type : str
        One of "RBF", "RQ", "MATERN".
    X1 : np.ndarray, shape (n1, d)
    X2 : np.ndarray, shape (n2, d)
    sigma : np.ndarray, shape (d,)
        Per-dimension length scales.
    alpha : float, optional
        Shape parameter for RQ, smoothness for MATERN.

    Returns
    -------
    K : np.ndarray, shape (n1, n2)
    """
    if kernel_type not in KERNEL_FORMULAS:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]

    if PARALLEL_SETTINGS["use_gpu"]:
        try:
            return _gpu_kernel(X1, X2, sigma, kernel_type, alpha)
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")

    if _should_use_parallel(n1, n2):
        return _parallel_kernel(X1, X2, sigma, kernel_type, alpha)

    sq_dist = _compute_sq_dist(X1, X2, sigma)
    formula = KERNEL_FORMULAS[kernel_type]
    return np.asarray(formula(sq_dist, alpha))


def get_parallel_info() -> Dict[str, Any]:
    """Get parallel processing capabilities."""
    info = {
        "parallel_available": PARALLEL_SETTINGS["use_parallel"],
        "cpu_cores": mp.cpu_count(),
        "gpu_available": CUPY_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "settings": PARALLEL_SETTINGS.copy(),
    }

    if CUPY_AVAILABLE and cp is not None:
        try:
            info["gpu_memory"] = cp.cuda.runtime.memGetInfo()[0]

        except Exception:
            info["gpu_memory"] = None

    return info
