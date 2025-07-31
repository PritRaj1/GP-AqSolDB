from configparser import ConfigParser
from typing import Any, Dict


def create_default_config(
    n_jobs: int = 2,
    use_gpu: bool = False,
    chunk_size: int = 500,
    min_size_for_parallel: int = 1000,
) -> ConfigParser:
    config = ConfigParser()
    config["KERNEL"] = {
        "type": "RBF",
        "lmbda": "0.1",
        "alpha": "1.0",
        "use_cache": "true",
        "cache_size": "100",
    }
    config["SPARSE"] = {
        "use_sparse": "false",
        "num_inducing": "20",
        "inducing_method": "random",
    }
    config["PARALLEL"] = {
        "use_parallel": "true",
        "n_jobs": str(n_jobs),
        "chunk_size": str(chunk_size),
        "use_gpu": str(use_gpu).lower(),
        "min_size_for_parallel": str(min_size_for_parallel),
    }
    return config


def load_kernel_config(config: ConfigParser) -> Dict[str, Any]:
    if "KERNEL" not in config:
        raise ValueError("KERNEL section not found in config")
    kernel_section = config["KERNEL"]
    return {
        "type": kernel_section.get("type", "RBF"),
        "lmbda": float(kernel_section.get("lmbda", "0.1")),
        "alpha": float(kernel_section.get("alpha", "1.0")),
        "use_cache": kernel_section.get("use_cache", "true") == "true",
        "cache_size": int(kernel_section.get("cache_size", "100")),
    }


def load_sparse_config(config: ConfigParser) -> Dict[str, Any]:
    if "SPARSE" not in config:
        raise ValueError("SPARSE section not found in config")
    sparse_section = config["SPARSE"]
    return {
        "use_sparse": sparse_section.get("use_sparse", "false") == "true",
        "num_inducing": int(sparse_section.get("num_inducing", "20")),
        "inducing_method": sparse_section.get("inducing_method", "random"),
    }


def load_parallel_config(config: ConfigParser) -> Dict[str, Any]:
    if "PARALLEL" not in config:
        raise ValueError("PARALLEL section not found in config")
    parallel_section = config["PARALLEL"]
    return {
        "use_parallel": parallel_section.get("use_parallel", "true") == "true",
        "n_jobs": int(parallel_section.get("n_jobs", "2")),
        "chunk_size": int(parallel_section.get("chunk_size", "500")),
        "use_gpu": parallel_section.get("use_gpu", "false") == "true",
        "min_size_for_parallel": int(
            parallel_section.get("min_size_for_parallel", "1000")
        ),
    }
