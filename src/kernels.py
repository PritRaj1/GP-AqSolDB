import numpy as np
from functools import lru_cache
import hashlib

class KernelCache:
    """Simple cache for repeated kernel computation
    """
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
    
    def _hash_inputs(self, X1, X2, sigma, alpha=None, kernel_type="RBF"):
        """Create a hash of the inputs for caching"""
        # Convert to bytes for hashing
        X1_bytes = X1.tobytes()
        X2_bytes = X2.tobytes()
        sigma_bytes = sigma.tobytes()
        
        hash_input = X1_bytes + X2_bytes + sigma_bytes + str(alpha).encode() + kernel_type.encode()
        return hashlib.md5(hash_input).hexdigest()
    
    def _hash_intermediate(self, X1, X2, sigma):
        """Create a hash for intermediate computations (normalized data)"""
        X1_bytes = X1.tobytes()
        X2_bytes = X2.tobytes()
        sigma_bytes = sigma.tobytes()
        
        hash_input = X1_bytes + X2_bytes + sigma_bytes
        return hashlib.md5(hash_input).hexdigest()
    
    def get(self, X1, X2, sigma, alpha=None, kernel_type="RBF"):
        """Get cached result if available"""
        key = self._hash_inputs(X1, X2, sigma, alpha, kernel_type)
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def get_intermediate(self, X1, X2, sigma):
        """Get cached intermediate computations"""
        key = self._hash_intermediate(X1, X2, sigma)
        if key in self.cache:
            return self.cache[key]
        return None
    
    def set(self, X1, X2, sigma, alpha, kernel_type, result):
        """Store in cache"""
        key = self._hash_inputs(X1, X2, sigma, alpha, kernel_type)
        
        # LRU eviction if cache is full - simple FIFO for now
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result
    
    def set_intermediate(self, X1, X2, sigma, result):
        """Store intermediate computations in cache"""
        key = self._hash_intermediate(X1, X2, sigma)
        
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result
    
    def get_stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
    
    def clear(self):
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

# Global instance
_kernel_cache = KernelCache()

def RBF(X1, X2, sigma, use_cache=True):
    """
    Radial Basis Function kernel
    
    Parameters:
    -----------
    X1 : np.ndarray, shape (n1, d)
        First set of points
    X2 : np.ndarray, shape (n2, d) 
        Second set of points
    sigma : np.ndarray, shape (d,)
        Length scales for each dimension
    use_cache : bool
        Whether to use caching
        
    Returns:
    --------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix
    """
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
            _kernel_cache.set_intermediate(X1, X2, sigma, (X1_norm, X2_norm, X1_sq, X2_sq, inner_prod))
    else:
        X1_norm = X1 / sigma
        X2_norm = X2 / sigma
        
        # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
        X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2_norm**2, axis=1)
        inner_prod = X1_norm @ X2_norm.T
    
    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    result = np.exp(-0.5 * sq_dist)
    
    if use_cache:
        _kernel_cache.set(X1, X2, sigma, alpha=None, kernel_type="RBF", result=result)
    
    return result

def RQ(X1, X2, sigma, alpha, use_cache=True):
    """
    Rational Quadratic kernel 
    
    Parameters:
    -----------
    X1 : np.ndarray, shape (n1, d)
        First set of points
    X2 : np.ndarray, shape (n2, d)
        Second set of points
    sigma : np.ndarray, shape (d,)
        Length scales for each dimension
    alpha : float
        Shape parameter
    use_cache : bool
        Whether to use caching
        
    Returns:
    --------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix
    """
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
            _kernel_cache.set_intermediate(X1, X2, sigma, (X1_norm, X2_norm, X1_sq, X2_sq, inner_prod))
    else:
        X1_norm = X1 / sigma
        X2_norm = X2 / sigma
        
        # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
        X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2_norm**2, axis=1)
        inner_prod = X1_norm @ X2_norm.T
    
    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    result = (1 + 0.5 * sq_dist / alpha)**(-alpha)
    
    if use_cache:
        _kernel_cache.set(X1, X2, sigma, alpha=alpha, kernel_type="RQ", result=result)
    
    return result

def get_kernel(config, sigma, use_cache=True, cache_size=100):
    """
    Get kernel function based on config with optional caching
    
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
    if use_cache and _kernel_cache.max_size != cache_size:
        _kernel_cache = KernelCache(max_size=cache_size)
    
    kernel_type = config.get("KERNEL", "type")
    alpha = config.getfloat("KERNEL", "alpha")

    kernel_functions = {    
        "RBF": lambda X1, X2: RBF(X1, X2, sigma, use_cache=use_cache),
        "RQ": lambda X1, X2: RQ(X1, X2, sigma, alpha, use_cache=use_cache)
    }
    
    if kernel_type not in kernel_functions:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernel_functions[kernel_type]

def get_cache_stats():
    return _kernel_cache.get_stats()

def clear_kernel_cache():
    _kernel_cache.clear()
