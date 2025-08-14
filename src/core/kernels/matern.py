import math
import warnings
from typing import Optional

import numpy as np
from scipy.special import kv

from .base import (
    _should_use_parallel,
    _parallel_kernel_computation,
)
from .cache import _kernel_cache
from src.utils.kernel_utils import PARALLEL_SETTINGS, CUPY_AVAILABLE

try:
    import cupy as cp
except ImportError:
    cp = None


def _compute_matern_chunk(
    X1_chunk: np.ndarray, X2: np.ndarray, sigma: np.ndarray, alpha: Optional[float]
) -> np.ndarray:
    """Matérn kernel for a chunk of X1"""
    if alpha is None:
        raise ValueError("Alpha parameter must not be None for MATERN kernel.")
    X1_norm = X1_chunk / sigma
    X2_norm = X2 / sigma

    # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T

    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    dist = np.sqrt(np.maximum(sq_dist, 0))

    # Matern 1/2: k(r) = exp(-r)
    if alpha == 0.5:
        return np.asarray(np.exp(-dist))

    # Matern 3/2: k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    elif alpha == 1.5:
        sqrt3_dist = np.sqrt(3) * dist
        return np.asarray((1 + sqrt3_dist) * np.exp(-sqrt3_dist))

    # Matern 5/2: k(r) = (1 + sqrt(5)*r + 5*r²/3) * exp(-sqrt(5)*r)
    elif alpha == 2.5:
        sqrt5_dist = np.sqrt(5) * dist
        return np.asarray((1 + sqrt5_dist + 5 * dist**2 / 3) * np.exp(-sqrt5_dist))

    # Generalize with scipy's modified Bessel function
    else:
        dist_safe = np.where(dist < 1e-10, 1e-10, dist)  # Avoid division by zero
        return np.asarray(
            (2 ** (1 - alpha) / math.gamma(alpha))
            * (np.sqrt(2 * alpha) * dist_safe) ** alpha
            * kv(alpha, np.sqrt(2 * alpha) * dist_safe)
        )


def _cupy_matern(X1: np.ndarray, X2: np.ndarray, sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Matérn kernel using GPU acceleration"""
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
        dist = cp.sqrt(cp.maximum(sq_dist, 0))

        if alpha == 0.5:
            result = cp.exp(-dist)
        elif alpha == 1.5:
            sqrt3_dist = cp.sqrt(3) * dist
            result = (1 + sqrt3_dist) * cp.exp(-sqrt3_dist)
        elif alpha == 2.5:
            sqrt5_dist = cp.sqrt(5) * dist
            result = (1 + sqrt5_dist + 5 * dist**2 / 3) * cp.exp(-sqrt5_dist)
        else:
            raise ValueError(
                f"Matérn kernel with alpha={alpha} not implemented for GPU"
            )

        # Back to CPU
        return np.asarray(cp.asnumpy(result))

    except Exception as e:
        # Clean up GPU memory if possible
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        raise RuntimeError(f"GPU computation failed: {e}")


def MATERN(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    use_cache: bool = True,
) -> np.ndarray:
    """
    k(x, y) = (2^(1-ν)/Γ(ν)) * (√(2ν) * ||x-y||/σ)^ν * K_ν(√(2ν) * ||x-y||/σ)

    Parameters:
    -----------
    X1 : np.ndarray
        First set of points (n1, d)
    X2 : np.ndarray
        Second set of points (n2, d)
    sigma : np.ndarray
        Length scales for each dimension (d,)
    alpha : float
        Smoothness parameter
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
            return _cupy_matern(X1, X2, sigma, alpha)
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")

    if _should_use_parallel(n1, n2):
        return _parallel_kernel_computation(
            X1, X2, sigma, "MATERN", alpha=alpha, use_cache=use_cache
        )

    # Sequential
    if use_cache:
        cached_result = _kernel_cache.get(
            X1, X2, sigma, alpha=alpha, kernel_type="MATERN"
        )
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
    dist = np.sqrt(np.maximum(sq_dist, 0))

    # Matern 1/2: k(r) = exp(-r)
    if alpha == 0.5:
        result = np.asarray(np.exp(-dist))

    # Matern 3/2: k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    elif alpha == 1.5:
        sqrt3_dist = np.sqrt(3) * dist
        result = np.asarray((1 + sqrt3_dist) * np.exp(-sqrt3_dist))

    # Matern 5/2: k(r) = (1 + sqrt(5)*r + 5*r²/3) * exp(-sqrt(5)*r)
    elif alpha == 2.5:
        sqrt5_dist = np.sqrt(5) * dist
        result = np.asarray((1 + sqrt5_dist + 5 * dist**2 / 3) * np.exp(-sqrt5_dist))

    # General case using scipy's modified Bessel function
    else:
        dist_safe = np.where(dist < 1e-10, 1e-10, dist)  # Avoid division by zero
        result = np.asarray(
            (2 ** (1 - alpha) / math.gamma(alpha))
            * (np.sqrt(2 * alpha) * dist_safe) ** alpha
            * kv(alpha, np.sqrt(2 * alpha) * dist_safe)
        )

    if use_cache:
        _kernel_cache.set(
            X1, X2, sigma, alpha=alpha, kernel_type="MATERN", result=result
        )

    return result
