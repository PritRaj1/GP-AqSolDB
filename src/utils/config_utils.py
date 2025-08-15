from configparser import ConfigParser
from typing import Any, Dict, Optional, Union


def create_default_config(
    n_features: Optional[int] = None,
    input_size: Optional[int] = None,
    output_size: int = 1,
    learning_rate: float = 0.001,
    num_epochs: int = 30,
    batch_size: int = 32,
    pretrain_iters: int = 10,
    seed: int = 42,
    min_var: float = 0.2,
    device_precision: str = "float32",
    use_gpu: bool = False,
    n_jobs: int = 2,
    chunk_size: int = 500,
    min_size_for_parallel: int = 1000,
    kernel_type: str = "RBF",
    lmbda: float = 0.1,
    alpha: float = 1.0,
    use_cache: bool = True,
    cache_size: int = 100,
    use_sparse: bool = False,
    num_inducing: int = 20,
    inducing_method: str = "random",
) -> ConfigParser:
    config = ConfigParser()
    # Network section (for GP-KAN)
    if n_features is not None or input_size is not None:
        config["NETWORK"] = {
            "input_size": str(input_size if input_size is not None else n_features),
            "output_size": str(output_size),
        }
    # GP section (for GP-KAN)
    config["GP"] = {
        "num_inducing_points": str(num_inducing),
        "z_init_low": "-2.0",
        "z_init_high": "2.0",
        "h_init_low": "-1.0",
        "h_init_high": "1.0",
        "global_length_scale": "0.4",
        "min_length_scale": "0.2",
        "global_covariance_scale": "1.0",
        "min_covariance_scale": "0.1",
        "global_jitter": "0.001",
        "baseline_jitter": "0.01",
    }
    # Kernel section (for GP)
    config["KERNEL"] = {
        "type": kernel_type,
        "lmbda": str(lmbda),
        "alpha": str(alpha),
        "use_cache": str(use_cache).lower(),
        "cache_size": str(cache_size),
    }
    # Sparse section (for GP)
    config["SPARSE"] = {
        "use_sparse": str(use_sparse).lower(),
        "num_inducing": str(num_inducing),
        "inducing_method": inducing_method,
    }
    # Parallel section (for GP)
    config["PARALLEL"] = {
        "use_parallel": "true",
        "n_jobs": str(n_jobs),
        "chunk_size": str(chunk_size),
        "use_gpu": str(use_gpu).lower(),
        "min_size_for_parallel": str(min_size_for_parallel),
    }
    # Normalization section
    config["NORMALIZATION"] = {"min_var": str(min_var)}
    # Training section
    config["TRAINING"] = {
        "seed": str(seed),
        "learning_rate": str(learning_rate),
        "num_epochs": str(num_epochs),
        "batch_size": str(batch_size),
        "pretrain_iters": str(pretrain_iters),
    }
    # Device section
    config["DEVICE"] = {
        "use_gpu": str(use_gpu).lower(),
        "device": "gpu" if use_gpu else "cpu",
        "precision": device_precision,
    }
    return config


def load_network_config(config: ConfigParser) -> Dict[str, int]:
    if "NETWORK" not in config:
        raise ValueError("NETWORK section not found in config")
    section = config["NETWORK"]
    return {
        "input_size": int(section.get("input_size", "1")),
        "output_size": int(section.get("output_size", "1")),
    }


def load_gp_config(config: ConfigParser) -> Dict[str, float]:
    if "GP" not in config:
        raise ValueError("GP section not found in config")
    section = config["GP"]
    return {
        "num_inducing_points": int(section.get("num_inducing_points", "10")),
        "z_init_low": float(section.get("z_init_low", "-2.0")),
        "z_init_high": float(section.get("z_init_high", "2.0")),
        "h_init_low": float(section.get("h_init_low", "-1.0")),
        "h_init_high": float(section.get("h_init_high", "1.0")),
        "global_length_scale": float(section.get("global_length_scale", "0.4")),
        "min_length_scale": float(section.get("min_length_scale", "0.2")),
        "global_covariance_scale": float(section.get("global_covariance_scale", "1.0")),
        "min_covariance_scale": float(section.get("min_covariance_scale", "0.1")),
        "global_jitter": float(section.get("global_jitter", "0.001")),
        "baseline_jitter": float(section.get("baseline_jitter", "0.01")),
    }


def load_kernel_config(config: ConfigParser) -> Dict[str, Any]:
    if "KERNEL" not in config:
        raise ValueError("KERNEL section not found in config")
    section = config["KERNEL"]
    return {
        "type": section.get("type", "RBF"),
        "lmbda": float(section.get("lmbda", "0.1")),
        "alpha": float(section.get("alpha", "1.0")),
        "use_cache": section.get("use_cache", "true") == "true",
        "cache_size": int(section.get("cache_size", "100")),
    }


def load_sparse_config(config: ConfigParser) -> Dict[str, Any]:
    if "SPARSE" not in config:
        raise ValueError("SPARSE section not found in config")
    section = config["SPARSE"]
    return {
        "use_sparse": section.get("use_sparse", "false") == "true",
        "num_inducing": int(section.get("num_inducing", "20")),
        "inducing_method": section.get("inducing_method", "random"),
    }


def load_parallel_config(config: ConfigParser) -> Dict[str, Any]:
    if "PARALLEL" not in config:
        raise ValueError("PARALLEL section not found in config")
    section = config["PARALLEL"]
    return {
        "use_parallel": section.get("use_parallel", "true") == "true",
        "n_jobs": int(section.get("n_jobs", "2")),
        "chunk_size": int(section.get("chunk_size", "500")),
        "use_gpu": section.get("use_gpu", "false") == "true",
        "min_size_for_parallel": int(section.get("min_size_for_parallel", "1000")),
    }


def load_normalization_config(config: ConfigParser) -> Dict[str, float]:
    if "NORMALIZATION" not in config:
        raise ValueError("NORMALIZATION section not found in config")
    section = config["NORMALIZATION"]
    return {"min_var": float(section.get("min_var", "0.2"))}


def load_training_config(config: ConfigParser) -> Dict[str, Union[int, float]]:
    if "TRAINING" not in config:
        raise ValueError("TRAINING section not found in config")
    section = config["TRAINING"]
    return {
        "seed": int(section.get("seed", "42")),
        "learning_rate": float(section.get("learning_rate", "0.001")),
        "num_epochs": int(section.get("num_epochs", "30")),
        "batch_size": int(section.get("batch_size", "32")),
        "pretrain_iters": int(section.get("pretrain_iters", "10")),
    }


def load_device_config(config: ConfigParser) -> Dict[str, Any]:
    if "DEVICE" not in config:
        raise ValueError("DEVICE section not found in config")
    section = config["DEVICE"]
    return {
        "use_gpu": section.get("use_gpu", "false") == "true",
        "device": section.get("device", "cpu"),
        "precision": section.get("precision", "float32"),
    }


def load_kan_conf(config: ConfigParser) -> Dict[str, Union[int, float]]:
    if "NETWORK" not in config:
        raise ValueError("NETWORK section not found in config")
    if "GP" not in config:
        raise ValueError("GP section not found in config")
    if "NORMALIZATION" not in config:
        raise ValueError("NORMALIZATION section not found in config")
    if "TRAINING" not in config:
        raise ValueError("TRAINING section not found in config")
    network_section = config["NETWORK"]
    normalization_section = config["NORMALIZATION"]
    training_section = config["TRAINING"]
    return {
        "input_size": int(network_section.get("input_size", "1")),
        "output_size": int(network_section.get("output_size", "1")),
        "num_inducing_points": int(config["GP"].get("num_inducing_points", "10")),
        "z_init_low": float(config["GP"].get("z_init_low", "-2.0")),
        "z_init_high": float(config["GP"].get("z_init_high", "2.0")),
        "h_init_low": float(config["GP"].get("h_init_low", "-1.0")),
        "h_init_high": float(config["GP"].get("h_init_high", "1.0")),
        "global_length_scale": float(config["GP"].get("global_length_scale", "0.4")),
        "min_length_scale": float(config["GP"].get("min_length_scale", "0.2")),
        "global_covariance_scale": float(
            config["GP"].get("global_covariance_scale", "1.0")
        ),
        "min_covariance_scale": float(config["GP"].get("min_covariance_scale", "0.1")),
        "global_jitter": float(config["GP"].get("global_jitter", "0.001")),
        "baseline_jitter": float(config["GP"].get("baseline_jitter", "0.01")),
        "min_var": float(normalization_section.get("min_var", "0.2")),
        "seed": int(training_section.get("seed", "42")),
        "learning_rate": float(training_section.get("learning_rate", "0.001")),
        "num_epochs": int(training_section.get("num_epochs", "30")),
        "batch_size": int(training_section.get("batch_size", "32")),
        "pretrain_iters": int(training_section.get("pretrain_iters", "10")),
    }
