from typing import Any

import numpy as np

from .cache import _kernel_cache
from .rbf import RBF
from .rq import RQ
from .matern import MATERN
from src.utils.kernel_utils import load_parallel_conf


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

    if use_cache and _kernel_cache.max_size != cache_size:
        _kernel_cache = _kernel_cache.__class__(max_size=cache_size)

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
