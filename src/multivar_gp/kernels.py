import hashlib
import math
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import kv

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Default
PARALLEL_SETTINGS = {
    "use_parallel": False,
    "n_jobs": None,  # None for auto-detect
    "chunk_size": 1000,  # Size of chunks for parallel processing
    "use_gpu": False,
    "min_size_for_parallel": 500,  # Minimum matrix size to use parallel processing
}


def load_parallel_conf(config: Any) -> Dict[str, Any]:
    if "PARALLEL" in config:
        parallel_config = config["PARALLEL"]

        # Conf for n_jobs can be 'None' string or an integer
        n_jobs_raw = parallel_config.get("n_jobs", fallback=None)
        if n_jobs_raw is None or n_jobs_raw.lower() == "none":
            n_jobs = None
        else:
            try:
                n_jobs = parallel_config.getint("n_jobs")
            except ValueError:
                n_jobs = None

        use_gpu_config = parallel_config.getboolean("use_gpu", fallback=False)
        gpu_available = use_gpu_config and CUPY_AVAILABLE

        if use_gpu_config and CUPY_AVAILABLE:
            try:
                # Simple GPU test
                test_array = cp.array([1.0, 2.0, 3.0])
                test_result = cp.sum(test_array)
                cp.asnumpy(test_result)
                gpu_available = True
            except Exception as e:
                print(f"GPU test failed: {e}. Disabling GPU acceleration.")
                gpu_available = False

        PARALLEL_SETTINGS.update(
            {
                "use_parallel": parallel_config.getboolean(
                    "use_parallel", fallback=False
                ),
                "n_jobs": n_jobs,
                "chunk_size": parallel_config.getint("chunk_size", fallback=1000),
                "use_gpu": gpu_available,
                "min_size_for_parallel": parallel_config.getint(
                    "min_size_for_parallel", fallback=500
                ),
            }
        )

        if use_gpu_config and not gpu_available:
            warnings.warn(
                "GPU acceleration requested but not available. Falling back to CPU."
            )

    return PARALLEL_SETTINGS


class KernelCache:
    """Simple cache for repeated kernel computation"""

    def __init__(self, max_size: int = 100) -> None:
        self.cache: Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]] = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def _hash_inputs(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        sigma: np.ndarray,
        alpha: Optional[float] = None,
        kernel_type: str = "RBF",
    ) -> str:
        """Create a hash of the inputs for caching"""
        # Convert to bytes for hashing
        X1_bytes = X1.tobytes()
        X2_bytes = X2.tobytes()
        sigma_bytes = np.asarray(sigma).tobytes()

        hash_input = (
            X1_bytes
            + X2_bytes
            + sigma_bytes
            + str(alpha).encode()
            + kernel_type.encode()
        )
        return hashlib.md5(hash_input).hexdigest()

    def _hash_intermediate(
        self, X1: np.ndarray, X2: np.ndarray, sigma: np.ndarray
    ) -> str:
        """Create a hash for intermediate computations (normalized data)"""
        X1_bytes = X1.tobytes()
        X2_bytes = X2.tobytes()
        sigma_bytes = sigma.tobytes()

        hash_input = X1_bytes + X2_bytes + sigma_bytes
        return hashlib.md5(hash_input).hexdigest()

    def get(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        sigma: np.ndarray,
        alpha: Optional[float] = None,
        kernel_type: str = "RBF",
    ) -> Optional[np.ndarray]:
        """Get cached result if available"""
        key = self._hash_inputs(X1, X2, sigma, alpha, kernel_type)
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None

    def get_intermediate(
        self, X1: np.ndarray, X2: np.ndarray, sigma: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Get cached intermediate computations"""
        key = self._hash_intermediate(X1, X2, sigma)
        value = self.cache.get(key, None)

        if (
            value is not None
            and isinstance(value, tuple)
            and len(value) == 5
            and all(isinstance(v, np.ndarray) for v in value)
        ):
            return value
        return None

    def set(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        sigma: np.ndarray,
        alpha: Optional[float],
        kernel_type: str,
        result: np.ndarray,
    ) -> None:
        """Store in cache"""
        key = self._hash_inputs(X1, X2, sigma, alpha, kernel_type)

        # LRU eviction if cache is full - simple FIFO for now
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = result

    def set_intermediate(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        sigma: np.ndarray,
        result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store intermediate computations in cache"""
        key = self._hash_intermediate(X1, X2, sigma)

        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        if (
            isinstance(result, tuple)
            and len(result) == 5
            and all(isinstance(v, np.ndarray) for v in result)
        ):
            self.cache[key] = result

    def get_stats(self) -> Dict[str, Union[int, float]]:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }

    def clear(self) -> None:
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


# Global instance
_kernel_cache = KernelCache()


def _get_n_jobs() -> int:
    """Number of jobs for parallel processing"""
    n_jobs = PARALLEL_SETTINGS["n_jobs"]
    if n_jobs is not None:
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
        return _compute_rbf_chunk(X1_chunk, X2, sigma)
    elif kernel_type == "RQ":
        return _compute_rq_chunk(X1_chunk, X2, sigma, alpha)
    elif kernel_type == "MATERN":
        return _compute_matern_chunk(X1_chunk, X2, sigma, alpha)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


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


def _compute_rq_chunk(
    X1_chunk: np.ndarray, X2: np.ndarray, sigma: np.ndarray, alpha: Optional[float]
) -> np.ndarray:
    """RQ kernel for a chunk of X1"""
    if alpha is None:
        raise ValueError("Alpha parameter must not be None for RQ kernel.")
    X1_norm = X1_chunk / sigma
    X2_norm = X2 / sigma

    # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T

    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    return np.asarray((1 + 0.5 * sq_dist / alpha) ** (-alpha))


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
    chunk_size = int(PARALLEL_SETTINGS["chunk_size"])

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
            return cached_result

    # Prepare and process chunk
    chunks = []
    for i_start, i_end in _chunk_indices(n1, chunk_size):
        X1_chunk = X1[i_start:i_end]
        chunks.append((X1_chunk, X2, sigma, kernel_type, alpha))

    n_jobs = _get_n_jobs()
    result_chunks = []

    # If single chunk, no need for parallel processing
    if len(chunks) == 1:
        result_chunks = [_compute_kernel_chunk(chunks[0])]
    else:
        try:
            # Use a more conservative approach for multiprocessing
            n_jobs_int = int(n_jobs) if n_jobs is not None else 1
            with ProcessPoolExecutor(max_workers=min(n_jobs_int, 4)) as executor:
                # Add timeout to prevent hanging
                result_chunks = list(
                    executor.map(_compute_kernel_chunk, chunks, timeout=300)
                )
        except Exception as e:
            print(
                f"Parallel processing failed: {e}. "
                f"Falling back to sequential computation."
            )
            # Fallback to sequential processing
            result_chunks = [_compute_kernel_chunk(chunk) for chunk in chunks]

    result = np.vstack(result_chunks)

    if use_cache:
        if kernel_type == "RBF":
            _kernel_cache.set(
                X1, X2, sigma, alpha=None, kernel_type="RBF", result=result
            )
        elif kernel_type == "RQ":
            _kernel_cache.set(
                X1, X2, sigma, alpha=alpha, kernel_type="RQ", result=result
            )
        elif kernel_type == "MATERN":
            _kernel_cache.set(
                X1, X2, sigma, alpha=alpha, kernel_type="MATERN", result=result
            )

    return result


def _cupy_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    kernel_type: str,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """CuPy kernel computation with improved error handling"""
    if not CUPY_AVAILABLE:
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

        if kernel_type == "RBF":
            result = cp.exp(-0.5 * sq_dist)
        elif kernel_type == "RQ":
            if alpha is None:
                raise ValueError("Alpha parameter must not be None for RQ kernel.")
            result = (1 + 0.5 * sq_dist / alpha) ** (-alpha)
        elif kernel_type == "MATERN":
            if alpha is None:
                raise ValueError("Alpha parameter must not be None for MATERN kernel.")
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
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Back to CPU
        return cp.asnumpy(result)

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
    Radial Basis Function kernel with optional parallel processing

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
    # Ensure sigma is a numpy array
    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]

    if PARALLEL_SETTINGS["use_gpu"]:
        try:
            return _cupy_kernel(X1, X2, sigma, "RBF")
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")

    if _should_use_parallel(n1, n2):
        return _parallel_kernel_computation(X1, X2, sigma, "RBF", use_cache=use_cache)

    # Sequential implementation
    if use_cache:
        cached_result = _kernel_cache.get(X1, X2, sigma, alpha=None, kernel_type="RBF")
        if cached_result is not None:
            return cached_result

    # Check for cached intermediate computations
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

            # Cache intermediate computations
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


def RQ(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Rational Quadratic kernel with optional parallel processing

    Parameters:
    -----------
    X1 : np.ndarray
        First set of points (n1, d)
    X2 : np.ndarray
        Second set of points (n2, d)
    sigma : np.ndarray
        Length scales for each dimension (d,)
    alpha : float
        Shape parameter
    use_cache : bool
        Whether to use caching

    Returns:
    --------
    K : np.ndarray
        Kernel matrix (n1, n2)
    """
    # Ensure sigma is a numpy array
    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]

    if PARALLEL_SETTINGS["use_gpu"]:
        try:
            return _cupy_kernel(X1, X2, sigma, "RQ", alpha=alpha)
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")

    if _should_use_parallel(n1, n2):
        return _parallel_kernel_computation(
            X1, X2, sigma, "RQ", alpha=alpha, use_cache=use_cache
        )

    # Sequential implementation
    if use_cache:
        cached_result = _kernel_cache.get(X1, X2, sigma, alpha=alpha, kernel_type="RQ")
        if cached_result is not None:
            return cached_result

    # Check for cached intermediate computations
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

            # Cache intermediate computations
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
    result = np.asarray((1 + 0.5 * sq_dist / alpha) ** (-alpha))

    if use_cache:
        _kernel_cache.set(X1, X2, sigma, alpha=alpha, kernel_type="RQ", result=result)

    return result


def MATERN(
    X1: np.ndarray,
    X2: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Matérn kernel with optional parallel processing

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
    # Ensure sigma is a numpy array
    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]

    if PARALLEL_SETTINGS["use_gpu"]:
        try:
            return _cupy_kernel(X1, X2, sigma, "MATERN", alpha=alpha)
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")

    if _should_use_parallel(n1, n2):
        return _parallel_kernel_computation(
            X1, X2, sigma, "MATERN", alpha=alpha, use_cache=use_cache
        )

    # Sequential implementation
    if use_cache:
        cached_result = _kernel_cache.get(
            X1, X2, sigma, alpha=alpha, kernel_type="MATERN"
        )
        if cached_result is not None:
            return cached_result

    # Check for cached intermediate computations
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

            # Cache intermediate computations
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


def configure_parallel_settings(
    use_parallel: bool = False,
    n_jobs: Optional[int] = None,
    chunk_size: int = 1000,
    use_gpu: bool = False,
    min_size_for_parallel: int = 500,
) -> None:
    """
    Configure config for kernel computations

    Parameters:
    -----------
    use_parallel : bool
        Whether to use parallel processing
    n_jobs : int, optional
        Number of parallel jobs (None for auto-detect)
    chunk_size : int
        Size of chunks for parallel processing
    use_gpu : bool
        Whether to use GPU acceleration
    min_size_for_parallel : int
        Minimum matrix size to use parallel processing
    """
    PARALLEL_SETTINGS.update(
        {
            "use_parallel": use_parallel,
            "n_jobs": n_jobs,
            "chunk_size": chunk_size,
            "use_gpu": use_gpu,
            "min_size_for_parallel": min_size_for_parallel,
        }
    )

    print("Parallel kernel settings:")
    print(f"  Use parallel: {use_parallel}")
    print(f"  Jobs: {n_jobs if n_jobs else 'auto'}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Use GPU: {use_gpu}")
    print(f"  Min size for parallel: {min_size_for_parallel}")

    if use_gpu:
        if CUPY_AVAILABLE:
            print("  GPU backend: CuPy")
        else:
            print("  GPU backend: None available")
            PARALLEL_SETTINGS["use_gpu"] = False


def get_parallel_info() -> Dict[str, Any]:
    """Get parallel processing capabilities"""
    info = {
        "parallel_available": PARALLEL_SETTINGS["use_parallel"],
        "cpu_cores": mp.cpu_count(),
        "gpu_available": CUPY_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "settings": PARALLEL_SETTINGS.copy(),
    }

    if CUPY_AVAILABLE:
        try:
            info["gpu_memory"] = cp.cuda.runtime.memGetInfo()[0]  # Free memory
        except Exception:
            info["gpu_memory"] = None

    return info


def get_kernel(
    config: Any, sigma: np.ndarray, use_cache: bool = True, cache_size: int = 100
) -> Any:
    """
    Get kernel function based on config with optional caching and parallel processing

    Parameters:
    -----------
    config : ConfigParser
        Configuration object
    sigma : np.ndarray
        Length scales for each dimension
    use_cache : bool
        Whether to use caching
    cache_size : int
        Maximum size of the cache

    Returns:
    --------
    kernel_func : function
        Vectorized kernel function that takes (X1, X2) and returns kernel matrix
    """
    global _kernel_cache
    load_parallel_conf(config)

    # Configure cache
    if use_cache and _kernel_cache.max_size != cache_size:
        _kernel_cache = KernelCache(max_size=cache_size)

    kernel_type = config.get("KERNEL", "type")
    alpha = config.getfloat("KERNEL", "alpha")

    kernel_functions = {
        "RBF": lambda X1, X2: RBF(X1, X2, sigma, use_cache=use_cache),
        "RQ": lambda X1, X2: RQ(X1, X2, sigma, alpha, use_cache=use_cache),
        "MATERN": lambda X1, X2: MATERN(X1, X2, sigma, alpha, use_cache=use_cache),
    }

    if kernel_type not in kernel_functions:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernel_functions[kernel_type]


def get_cache_stats() -> Dict[str, Union[int, float]]:
    return _kernel_cache.get_stats()


def clear_kernel_cache() -> None:
    _kernel_cache.clear()
