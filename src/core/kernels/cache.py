import hashlib
from typing import Dict, Optional, Tuple, Union

import numpy as np


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
        """Create a hash for intermediate vals (normalized data)"""
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
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
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
        """Get cached intermediate vals"""
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
        """Store intermediate vals in cache"""
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


def get_cache_stats() -> Dict[str, Union[int, float]]:
    return _kernel_cache.get_stats()


def clear_kernel_cache() -> None:
    _kernel_cache.clear()
