import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.kernel_utils import PARALLEL_SETTINGS, CUPY_AVAILABLE
from .cache import _kernel_cache

try:
    import cupy as cp
except ImportError:
    cp = None


def _get_n_jobs() -> int:
    """Number of jobs for parallel processing"""
    n_jobs = PARALLEL_SETTINGS["n_jobs"]
    # Only cast to int if n_jobs is not None and not isinstance(n_jobs, bool)
    if n_jobs is not None and not isinstance(n_jobs, bool):
        return int(n_jobs)
    return min(mp.cpu_count(), 8)  # Capped at 8 to avoid overhead


def _should_use_parallel(n1: int, n2: int) -> bool:
    """Based on matrix size"""
    if not PARALLEL_SETTINGS["use_parallel"]:
        return False
    min_size = PARALLEL_SETTINGS["min_size_for_parallel"]
    if min_size is None:
        min_size = 500
    return n1 * n2 >= int(min_size) * int(min_size)


def _chunk_indices(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    """List of chunk indices"""
    return [(i, min(i + int(chunk_size), n)) for i in range(0, n, int(chunk_size))]


def _compute_kernel_chunk(
    args: Tuple[np.ndarray, np.ndarray, np.ndarray, str, Optional[float]],
) -> np.ndarray:
    """Single chunk kernel matrix"""
    X1_chunk, X2, sigma, kernel_type, alpha = args

    if kernel_type == "RBF":
        from .rbf import _compute_rbf_chunk
        return _compute_rbf_chunk(X1_chunk, X2, sigma)
    elif kernel_type == "RQ":
        from .rq import _compute_rq_chunk
        return _compute_rq_chunk(X1_chunk, X2, sigma, alpha)
    elif kernel_type == "MATERN":
        from .matern import _compute_matern_chunk
        return _compute_matern_chunk(X1_chunk, X2, sigma, alpha)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def _parallel_kernel_computation(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    kernel_type: str,
    alpha: Optional[float] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Parallelized kernel computation with improved error handling"""
    n1 = X1.shape[0]  # Only use n1, ignore n2
    chunk_size_raw = PARALLEL_SETTINGS["chunk_size"]
    chunk_size = (
        int(chunk_size_raw)
        if chunk_size_raw is not None and not isinstance(chunk_size_raw, bool)
        else 1000
    )

    # Check cache first
    if use_cache:
        if kernel_type == "RBF":
            cached_result = _kernel_cache.get(
                X1, X2, sigma, alpha=None, kernel_type="RBF"
            )
        elif kernel_type == "RQ":
            cached_result = _kernel_cache.get(
                X1, X2, sigma, alpha=alpha, kernel_type="RQ"
            )
        elif kernel_type == "MATERN":
            cached_result = _kernel_cache.get(
                X1, X2, sigma, alpha=alpha, kernel_type="MATERN"
            )
        else:
            cached_result = None

        if cached_result is not None:
            if isinstance(cached_result, tuple):
                return np.asarray(cached_result[0])
            return np.asarray(cached_result)

    # Prepare and process chunk
    chunks = []
    for i_start, i_end in _chunk_indices(n1, chunk_size):
        X1_chunk = X1[i_start:i_end]
        chunks.append((X1_chunk, X2, sigma, kernel_type, alpha))

    # Process chunks in parallel
    n_jobs = _get_n_jobs()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_compute_kernel_chunk, chunks))

    # Combine results
    result = np.vstack(results)

    # Cache the result
    if use_cache:
        if kernel_type == "RBF":
            _kernel_cache.set(X1, X2, sigma, alpha=None, kernel_type="RBF", result=result)
        elif kernel_type == "RQ":
            _kernel_cache.set(X1, X2, sigma, alpha=alpha, kernel_type="RQ", result=result)
        elif kernel_type == "MATERN":
            _kernel_cache.set(X1, X2, sigma, alpha=alpha, kernel_type="MATERN", result=result)

    return result


def get_parallel_info() -> Dict[str, Any]:
    """Get parallel processing capabilities"""
    info = {
        "parallel_available": PARALLEL_SETTINGS["use_parallel"],
        "cpu_cores": mp.cpu_count(),
        "gpu_available": CUPY_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "settings": PARALLEL_SETTINGS.copy(),
    }

    if CUPY_AVAILABLE and cp is not None:
        try:
            info["gpu_memory"] = cp.cuda.runtime.memGetInfo()[0]  # Free memory
        except Exception:
            info["gpu_memory"] = None

    return info
