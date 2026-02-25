from configparser import ConfigParser
from typing import Dict, Optional, Union


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
    use_sparse: bool = False,
    num_inducing: int = 20,
    inducing_method: str = "random",
) -> ConfigParser:
    config = ConfigParser()
    if n_features is not None or input_size is not None:
        config["NETWORK"] = {
            "input_size": str(input_size if input_size is not None else n_features),
            "output_size": str(output_size),
        }
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
    config["KERNEL"] = {
        "type": kernel_type,
        "lmbda": str(lmbda),
        "alpha": str(alpha),
    }
    config["SPARSE"] = {
        "use_sparse": str(use_sparse).lower(),
        "num_inducing": str(num_inducing),
        "inducing_method": inducing_method,
    }
    config["PARALLEL"] = {
        "use_parallel": "true",
        "n_jobs": str(n_jobs),
        "chunk_size": str(chunk_size),
        "use_gpu": str(use_gpu).lower(),
        "min_size_for_parallel": str(min_size_for_parallel),
    }
    config["NORMALIZATION"] = {"min_var": str(min_var)}
    config["TRAINING"] = {
        "seed": str(seed),
        "learning_rate": str(learning_rate),
        "num_epochs": str(num_epochs),
        "batch_size": str(batch_size),
        "pretrain_iters": str(pretrain_iters),
    }
    config["DEVICE"] = {
        "use_gpu": str(use_gpu).lower(),
        "device": "gpu" if use_gpu else "cpu",
        "precision": device_precision,
    }
    return config


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


def load_kan_conf(config: ConfigParser) -> Dict[str, Union[int, float]]:
    for section in ("NETWORK", "GP", "NORMALIZATION", "TRAINING"):
        if section not in config:
            raise ValueError(f"{section} section not found in config")

    network = config["NETWORK"]
    gp = config["GP"]
    norm = config["NORMALIZATION"]
    train = config["TRAINING"]
    return {
        "input_size": int(network.get("input_size", "1")),
        "output_size": int(network.get("output_size", "1")),
        "num_inducing_points": int(gp.get("num_inducing_points", "10")),
        "z_init_low": float(gp.get("z_init_low", "-2.0")),
        "z_init_high": float(gp.get("z_init_high", "2.0")),
        "h_init_low": float(gp.get("h_init_low", "-1.0")),
        "h_init_high": float(gp.get("h_init_high", "1.0")),
        "global_length_scale": float(gp.get("global_length_scale", "0.4")),
        "min_length_scale": float(gp.get("min_length_scale", "0.2")),
        "global_covariance_scale": float(gp.get("global_covariance_scale", "1.0")),
        "min_covariance_scale": float(gp.get("min_covariance_scale", "0.1")),
        "global_jitter": float(gp.get("global_jitter", "0.001")),
        "baseline_jitter": float(gp.get("baseline_jitter", "0.01")),
        "min_var": float(norm.get("min_var", "0.2")),
        "seed": int(train.get("seed", "42")),
        "learning_rate": float(train.get("learning_rate", "0.001")),
        "num_epochs": int(train.get("num_epochs", "30")),
        "batch_size": int(train.get("batch_size", "32")),
        "pretrain_iters": int(train.get("pretrain_iters", "10")),
    }
