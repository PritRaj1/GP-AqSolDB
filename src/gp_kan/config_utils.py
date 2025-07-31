from configparser import ConfigParser
from typing import Dict, Optional, Union


def create_default_config(
    n_features: Optional[int] = None,
    use_gpu: bool = False,
    input_size: Optional[int] = None,
    output_size: int = 1,
    learning_rate: float = 0.001,
    num_epochs: int = 30,
    batch_size: int = 32,
    pretrain_iters: int = 10,
    seed: int = 42,
    min_var: float = 0.2,
    device_precision: str = "float32",
) -> ConfigParser:
    config = ConfigParser()
    if n_features is not None or input_size is not None:
        config["NETWORK"] = {
            "input_size": str(input_size if input_size is not None else n_features),
            "output_size": str(output_size),
        }
    config["GP"] = {
        "num_inducing_points": "10",
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
    gp_section = config["GP"]
    return {
        "num_inducing_points": int(gp_section.get("num_inducing_points", "10")),
        "z_init_low": float(gp_section.get("z_init_low", "-2.0")),
        "z_init_high": float(gp_section.get("z_init_high", "2.0")),
        "h_init_low": float(gp_section.get("h_init_low", "-1.0")),
        "h_init_high": float(gp_section.get("h_init_high", "1.0")),
        "global_length_scale": float(gp_section.get("global_length_scale", "0.4")),
        "min_length_scale": float(gp_section.get("min_length_scale", "0.2")),
        "global_covariance_scale": float(
            gp_section.get("global_covariance_scale", "1.0")
        ),
        "min_covariance_scale": float(gp_section.get("min_covariance_scale", "0.1")),
        "global_jitter": float(gp_section.get("global_jitter", "0.001")),
        "baseline_jitter": float(gp_section.get("baseline_jitter", "0.01")),
    }


def load_normalization_config(config: ConfigParser) -> Dict[str, float]:
    if "NORMALIZATION" not in config:
        raise ValueError("NORMALIZATION section not found in config")
    normalization_section = config["NORMALIZATION"]
    return {"min_var": float(normalization_section.get("min_var", "0.2"))}


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
