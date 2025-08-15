import warnings

import numpy as np

from ...utils import CUPY_AVAILABLE, PARALLEL_SETTINGS
from .base import (
    _parallel_kernel_computation,
    _should_use_parallel,
)
from .cache import _kernel_cache

try:
    import cupy as cp
except ImportError:
    cp = None


def _compute_rbf_chunk(
    X1_chunk: np.ndarray, X2: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """RBF kernel for a chunk of X1"""
    X1_norm = X1_chunk / sigma
    X2_norm = X2 / sigma

    # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T

    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    return np.asarray(np.exp(-0.5 * sq_dist))


def _cupy_rbf(X1: np.ndarray, X2: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """RBF kernel using GPU acceleration"""
    if not CUPY_AVAILABLE or cp is None:
        raise RuntimeError("GPU acceleration not available. Install cupy.")

    try:
        X1_gpu = cp.asarray(X1)
        X2_gpu = cp.asarray(X2)
        sigma_gpu = cp.asarray(sigma)

        X1_norm = X1_gpu / sigma_gpu
        X2_norm = X2_gpu / sigma_gpu

        # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
        X1_sq = cp.sum(X1_norm**2, axis=1, keepdims=True)
        X2_sq = cp.sum(X2_norm**2, axis=1)
        inner_prod = X1_norm @ X2_norm.T

        sq_dist = X1_sq + X2_sq - 2 * inner_prod
        result = cp.exp(-0.5 * sq_dist)

        # Back to CPU
        return np.asarray(cp.asnumpy(result))

    except Exception as e:
        # Clean up GPU memory if possible
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        raise RuntimeError(f"GPU computation failed: {e}")


def RBF(
    X1: np.ndarray, X2: np.ndarray, sigma: np.ndarray, use_cache: bool = True
) -> np.ndarray:
    """
    k(x, y) = exp(-||x - y||² / (2 * σ²))

    Parameters:
    -----------
    X1 : np.ndarray
        First set of points (n1, d)
    X2 : np.ndarray
        Second set of points (n2, d)
    sigma : np.ndarray
        Length scales for each dimension (d,)
    use_cache : bool
        Whether to use caching

    Returns:
    --------
    K : np.ndarray
        Kernel matrix (n1, n2)
    """
    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]

    if PARALLEL_SETTINGS["use_gpu"]:
        try:
            return _cupy_rbf(X1, X2, sigma)
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")

    if _should_use_parallel(n1, n2):
        return _parallel_kernel_computation(X1, X2, sigma, "RBF", use_cache=use_cache)

    # Sequential
    if use_cache:
        cached_result = _kernel_cache.get(X1, X2, sigma, alpha=None, kernel_type="RBF")
        if cached_result is not None:
            if isinstance(cached_result, tuple):
                return np.asarray(cached_result[0])
            return np.asarray(cached_result)

    # Check for cached intermediate vals
    if use_cache:
        intermediate = _kernel_cache.get_intermediate(X1, X2, sigma)
        if intermediate is not None:
            X1_norm, X2_norm, X1_sq, X2_sq, inner_prod = intermediate
        else:
            X1_norm = X1 / sigma
            X2_norm = X2 / sigma

            # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
            X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
            X2_sq = np.sum(X2_norm**2, axis=1)
            inner_prod = X1_norm @ X2_norm.T

            # Cache intermediate vals
            _kernel_cache.set_intermediate(
                X1, X2, sigma, (X1_norm, X2_norm, X1_sq, X2_sq, inner_prod)
            )
    else:
        X1_norm = X1 / sigma
        X2_norm = X2 / sigma

        # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
        X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2_norm**2, axis=1)
        inner_prod = X1_norm @ X2_norm.T

    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    result = np.asarray(np.exp(-0.5 * sq_dist))

    if use_cache:
        _kernel_cache.set(X1, X2, sigma, alpha=None, kernel_type="RBF", result=result)

    return result
