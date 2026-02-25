from configparser import ConfigParser
from typing import Any, Dict, Optional, Union


def create_default_config(
    n_features: Optional[int] = None,
    use_gpu: bool = False,
    kernel_type: str = "RBF",
    lmbda: float = 0.1,
    alpha: float = 1.0,
    num_inducing: int = 20,
    **kwargs: Any,
) -> ConfigParser:
    config = ConfigParser()
    input_size = kwargs.pop("input_size", None)
    output_size = kwargs.pop("output_size", 1)
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
        "use_sparse": str(kwargs.pop("use_sparse", False)).lower(),
        "num_inducing": str(num_inducing),
        "inducing_method": str(kwargs.pop("inducing_method", "random")),
    }
    min_var = kwargs.pop("min_var", 0.2)
    config["NORMALIZATION"] = {"min_var": str(min_var)}
    seed = kwargs.pop("seed", 42)
    learning_rate = kwargs.pop("learning_rate", 0.001)
    num_epochs = kwargs.pop("num_epochs", 30)
    batch_size = kwargs.pop("batch_size", 32)
    pretrain_iters = kwargs.pop("pretrain_iters", 10)
    config["TRAINING"] = {
        "seed": str(seed),
        "learning_rate": str(learning_rate),
        "num_epochs": str(num_epochs),
        "batch_size": str(batch_size),
        "pretrain_iters": str(pretrain_iters),
    }
    device_precision = kwargs.pop("device_precision", "float32")
    config["DEVICE"] = {
        "use_gpu": str(use_gpu).lower(),
        "device": "gpu" if use_gpu else "cpu",
        "precision": str(device_precision),
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

    result: Dict[str, Union[int, float]] = load_gp_config(config)

    network = config["NETWORK"]
    norm = config["NORMALIZATION"]
    train = config["TRAINING"]
    result.update(
        {
            "input_size": int(network.get("input_size", "1")),
            "output_size": int(network.get("output_size", "1")),
            "min_var": float(norm.get("min_var", "0.2")),
            "seed": int(train.get("seed", "42")),
            "learning_rate": float(train.get("learning_rate", "0.001")),
            "num_epochs": int(train.get("num_epochs", "30")),
            "batch_size": int(train.get("batch_size", "32")),
            "pretrain_iters": int(train.get("pretrain_iters", "10")),
        }
    )
    return result
