import numpy as np
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy.special import kv
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Default
PARALLEL_SETTINGS = {
    'use_parallel': False,
    'n_jobs': None,  # None for auto-detect
    'chunk_size': 1000,  # Size of chunks for parallel processing
    'use_gpu': False, 
    'min_size_for_parallel': 500  # Minimum matrix size to use parallel processing
}

def load_parallel_conf(config):
    global PARALLEL_SETTINGS
    
    if 'PARALLEL' in config:
        parallel_config = config['PARALLEL']
        
        # Conf for n_jobs can be 'None' string or an integer
        n_jobs_raw = parallel_config.get('n_jobs', fallback=None)
        if n_jobs_raw is None or n_jobs_raw.lower() == 'none':
            n_jobs = None
        else:
            try:
                n_jobs = parallel_config.getint('n_jobs')
            except ValueError:
                n_jobs = None
        
        PARALLEL_SETTINGS.update({
            'use_parallel': parallel_config.getboolean('use_parallel', fallback=False),
            'n_jobs': n_jobs,
            'chunk_size': parallel_config.getint('chunk_size', fallback=1000),
            'use_gpu': parallel_config.getboolean('use_gpu', fallback=False),
            'min_size_for_parallel': parallel_config.getint('min_size_for_parallel', fallback=500)
        })
        
        if PARALLEL_SETTINGS['use_gpu'] and not (CUPY_AVAILABLE):
            warnings.warn("GPU acceleration requested but no GPU libraries available. Falling back to CPU.")
            PARALLEL_SETTINGS['use_gpu'] = False
    
    return PARALLEL_SETTINGS

class KernelCache:
    """Simple cache for repeated kernel computation"""
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
        sigma_bytes = np.asarray(sigma).tobytes()
        
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

def _get_n_jobs():
    """Number of jobs for parallel processing"""
    if PARALLEL_SETTINGS['n_jobs'] is not None:
        return PARALLEL_SETTINGS['n_jobs']
    return min(mp.cpu_count(), 8)  # Capped at 8 to avoid overhead

def _should_use_parallel(n1, n2):
    """Based on matrix size"""
    if not PARALLEL_SETTINGS['use_parallel']:
        return False
    min_size = PARALLEL_SETTINGS['min_size_for_parallel']
    return n1 * n2 >= min_size * min_size

def _chunk_indices(n, chunk_size):
    """List of chunk indices"""
    for i in range(0, n, chunk_size):
        yield i, min(i + chunk_size, n)

def _compute_kernel_chunk(args):
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

def _compute_rbf_chunk(X1_chunk, X2, sigma):
    """RBF kernel for a chunk of X1"""
    X1_norm = X1_chunk / sigma
    X2_norm = X2 / sigma
    
    # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T
    
    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    return np.exp(-0.5 * sq_dist)

def _compute_rq_chunk(X1_chunk, X2, sigma, alpha):
    """RQ kernel for a chunk of X1"""
    X1_norm = X1_chunk / sigma
    X2_norm = X2 / sigma
    
    # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T
    
    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    return (1 + 0.5 * sq_dist / alpha)**(-alpha)

def _compute_matern_chunk(X1_chunk, X2, sigma, alpha):
    """Matérn kernel for a chunk of X1"""
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
        return np.exp(-dist)

    # Matern 3/2: k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    elif alpha == 1.5:
        sqrt3_dist = np.sqrt(3) * dist
        return (1 + sqrt3_dist) * np.exp(-sqrt3_dist)

    # Matern 5/2: k(r) = (1 + sqrt(5)*r + 5*r²/3) * exp(-sqrt(5)*r)
    elif alpha == 2.5:
        sqrt5_dist = np.sqrt(5) * dist
        return (1 + sqrt5_dist + 5 * dist**2 / 3) * np.exp(-sqrt5_dist)
    
    # Genealize with scipy's modified Bessel function
    else:
        dist_safe = np.where(dist < 1e-10, 1e-10, dist) # Avoid division by zero
        return (2**(1-alpha) / np.math.gamma(alpha)) * (np.sqrt(2*alpha) * dist_safe)**alpha * kv(alpha, np.sqrt(2*alpha) * dist_safe)

def _parallel_kernel_computation(X1, X2, sigma, kernel_type, alpha=None, use_cache=True):
    """Parallelized kernel computation"""
    n1, n2 = X1.shape[0], X2.shape[0]
    chunk_size = PARALLEL_SETTINGS['chunk_size']
    
    # Check cache first
    if use_cache:
        if kernel_type == "RBF":
            cached_result = _kernel_cache.get(X1, X2, sigma, alpha=None, kernel_type="RBF")
        elif kernel_type == "RQ":
            cached_result = _kernel_cache.get(X1, X2, sigma, alpha=alpha, kernel_type="RQ")
        elif kernel_type == "MATERN":
            cached_result = _kernel_cache.get(X1, X2, sigma, alpha=alpha, kernel_type="MATERN")
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
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            result_chunks = list(executor.map(_compute_kernel_chunk, chunks))
    
    result = np.vstack(result_chunks)
    
    if use_cache:
        if kernel_type == "RBF":
            _kernel_cache.set(X1, X2, sigma, alpha=None, kernel_type="RBF", result=result)
        elif kernel_type == "RQ":
            _kernel_cache.set(X1, X2, sigma, alpha=alpha, kernel_type="RQ", result=result)
        elif kernel_type == "MATERN":
            _kernel_cache.set(X1, X2, sigma, alpha=alpha, kernel_type="MATERN", result=result)
    
    return result

def _cupy_kernel(X1, X2, sigma, kernel_type, alpha=None):
    """CuPy kernel computation"""
    if not CUPY_AVAILABLE:
        raise RuntimeError("GPU acceleration not available. Install cupy.")
    
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
        result = (1 + 0.5 * sq_dist / alpha)**(-alpha)
    elif kernel_type == "MATERN":
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
            raise ValueError(f"Matérn kernel with alpha={alpha} not implemented for GPU")
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Back to CPU
    return cp.asnumpy(result)

def RBF(X1, X2, sigma, use_cache=True):
    """
    Radial Basis Function kernel with optional parallel processing
    
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
    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]
    
    if PARALLEL_SETTINGS['use_gpu']:
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
    Rational Quadratic kernel with optional parallel processing
    
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
    # Ensure sigma is a numpy array
    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]
    
    if PARALLEL_SETTINGS['use_gpu']:
        try:
            return _cupy_kernel(X1, X2, sigma, "RQ", alpha)
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")
    
    if _should_use_parallel(n1, n2):
        return _parallel_kernel_computation(X1, X2, sigma, "RQ", alpha, use_cache=use_cache)
    
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

def MATERN(X1, X2, sigma, alpha, use_cache=True):
    """
    Matérn kernel with optional parallel processing
    
    Parameters:
    -----------
    X1 : np.ndarray, shape (n1, d)
        First set of points
    X2 : np.ndarray, shape (n2, d)
        Second set of points
    sigma : np.ndarray, shape (d,)
        Length scales for each dimension
    alpha : float
        Smoothness parameter (0.5, 1.5, 2.5 are most common)
    use_cache : bool
        Whether to use caching
        
    Returns:
    --------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix
    """
    # Ensure sigma is a numpy array
    sigma = np.asarray(sigma)
    n1, n2 = X1.shape[0], X2.shape[0]
    
    if PARALLEL_SETTINGS['use_gpu']:
        try:
            return _cupy_kernel(X1, X2, sigma, "MATERN", alpha=alpha)
        except Exception as e:
            warnings.warn(f"GPU computation failed, falling back to CPU: {e}")
    
    if _should_use_parallel(n1, n2):
        return _parallel_kernel_computation(X1, X2, sigma, "MATERN", nu=nu, use_cache=use_cache)
    
    # Sequential implementation
    if use_cache:
        cached_result = _kernel_cache.get(X1, X2, sigma, alpha=alpha, kernel_type="MATERN")
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
    dist = np.sqrt(np.maximum(sq_dist, 0))
    
    # Matern 1/2: k(r) = exp(-r)
    if alpha == 0.5:
        result = np.exp(-dist)

    # Matern 3/2: k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    elif alpha == 1.5:
        sqrt3_dist = np.sqrt(3) * dist
        result = (1 + sqrt3_dist) * np.exp(-sqrt3_dist)

    # Matern 5/2: k(r) = (1 + sqrt(5)*r + 5*r²/3) * exp(-sqrt(5)*r)
    elif alpha == 2.5:
        sqrt5_dist = np.sqrt(5) * dist
        result = (1 + sqrt5_dist + 5 * dist**2 / 3) * np.exp(-sqrt5_dist)

    # General case using scipy's modified Bessel function
    else:
        dist_safe = np.where(dist < 1e-10, 1e-10, dist) # Avoid division by zero
        result = (2**(1-alpha) / np.math.gamma(alpha)) * (np.sqrt(2*alpha) * dist_safe)**alpha * kv(alpha, np.sqrt(2*alpha) * dist_safe)
    
    if use_cache:
        _kernel_cache.set(X1, X2, sigma, alpha=alpha, kernel_type="MATERN", result=result)
    
    return result

def configure_parallel_settings(use_parallel=False, n_jobs=None, chunk_size=1000, 
                               use_gpu=False, min_size_for_parallel=500):
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
    global PARALLEL_SETTINGS
    PARALLEL_SETTINGS.update({
        'use_parallel': use_parallel,
        'n_jobs': n_jobs,
        'chunk_size': chunk_size,
        'use_gpu': use_gpu,
        'min_size_for_parallel': min_size_for_parallel
    })
    
    print(f"Parallel kernel settings:")
    print(f"  Use parallel: {use_parallel}")
    print(f"  Jobs: {n_jobs if n_jobs else 'auto'}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Use GPU: {use_gpu}")
    print(f"  Min size for parallel: {min_size_for_parallel}")
    
    if use_gpu:
        if CUPY_AVAILABLE:
            print(f"  GPU backend: CuPy")
        else:
            print(f"  GPU backend: None available")
            PARALLEL_SETTINGS['use_gpu'] = False

def get_parallel_info():
    """Get parallel processing capabilities"""
    info = {
        'parallel_available': PARALLEL_SETTINGS['use_parallel'],
        'cpu_cores': mp.cpu_count(),
        'gpu_available': CUPY_AVAILABLE,
        'cupy_available': CUPY_AVAILABLE,
        'settings': PARALLEL_SETTINGS.copy()
    }
    
    if CUPY_AVAILABLE:
        try:
            info['gpu_memory'] = cp.cuda.runtime.memGetInfo()[0]  # Free memory
        except:
            info['gpu_memory'] = None
    
    return info

def get_kernel(config, sigma, use_cache=True, cache_size=100):
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
        "MATERN": lambda X1, X2: MATERN(X1, X2, sigma, alpha, use_cache=use_cache)
    }
    
    if kernel_type not in kernel_functions:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernel_functions[kernel_type]

def get_cache_stats():
    return _kernel_cache.get_stats()

def clear_kernel_cache():
    _kernel_cache.clear()
