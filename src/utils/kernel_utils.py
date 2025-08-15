import warnings
from typing import Any, Dict, Optional

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Default parallel settings
PARALLEL_SETTINGS = {
    "use_parallel": False,
    "n_jobs": None,  # None for auto-detect
    "chunk_size": 1000,  # Size of chunks for parallel processing
    "use_gpu": False,
    "min_size_for_parallel": 500,  # Minimum matrix size to use parallel processing
}


def load_parallel_conf(config: Any) -> Dict[str, Any]:
    """Load parallel configuration from config object"""
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

        # Catch GPU errors before long runs
        if use_gpu_config and CUPY_AVAILABLE:
            try:
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
