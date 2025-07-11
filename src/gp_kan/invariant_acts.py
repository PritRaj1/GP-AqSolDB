from configparser import ConfigParser
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from src.gp_kan.normal_dist import NormalDist, get_device_config


def load_normalization_config(config: ConfigParser) -> Dict[str, float]:
    if "NORMALIZATION" not in config:
        raise ValueError("NORMALIZATION section not found in config")

    normalization_section = config["NORMALIZATION"]

    return {"min_var": float(normalization_section.get("min_var", "0.2"))}


def create_default_conf() -> ConfigParser:
    config = ConfigParser()
    config["NORMALIZATION"] = {"min_var": "0.2"}
    config["DEVICE"] = {"use_gpu": "false", "device": "cpu", "precision": "float32"}
    return config


class NormaliseGaussian:
    """Normalize - tanh for mean and sigmoid for variance."""

    def __init__(
        self, min_var: float = 0.2, config: Optional[ConfigParser] = None
    ) -> None:
        if config is not None:
            norm_params = load_normalization_config(config)
            self.min_var = norm_params["min_var"]
            self.device_config = get_device_config(config)
        else:
            self.min_var = min_var
            self.device_config = {"use_gpu": False, "device": "cpu"}

        # Precompute sigmoid offset for efficiency
        self.sigmoid_offset = self.inverse_sigmoid(self.min_var)

    @staticmethod
    def inverse_sigmoid(x: float) -> jax.Array:
        t1 = (1 / x) - 1
        t2 = -jnp.log(t1)
        return t2

    def __call__(self, x: NormalDist) -> NormalDist:

        # JIT once
        def _normalize_core(
            x_mean: jax.Array, x_var: jax.Array, sigmoid_offset: float, min_var: float
        ) -> Tuple[jax.Array, jax.Array]:
            out_mean = jnp.tanh(x_mean)
            out_var = jax.nn.sigmoid(x_var - x_mean**2 + sigmoid_offset)
            out_var = jnp.maximum(out_var, min_var)
            return out_mean, out_var

        if not hasattr(self, "_normalize_core_jit"):
            self._normalize_core_jit = jax.jit(_normalize_core)

        out_mean, out_var = self._normalize_core_jit(
            x.mean, x.var, self.sigmoid_offset, self.min_var
        )
        result = NormalDist(out_mean, out_var)  # type: ignore[call-arg]

        if self.device_config["use_gpu"]:
            return result.to_device("gpu")

        return result


class ReshapeGaussian:
    def __init__(
        self, new_shape: List[int], config: Optional[ConfigParser] = None
    ) -> None:
        self.new_shape = new_shape
        if config is not None:
            self.device_config = get_device_config(config)
        else:
            self.device_config = {"use_gpu": False, "device": "cpu"}

    def __call__(self, x: NormalDist) -> NormalDist:
        out_mean = jnp.reshape(x.mean, self.new_shape)
        out_var = jnp.reshape(x.var, self.new_shape)

        if self.device_config["use_gpu"]:
            return NormalDist(out_mean, out_var).to_device(
                "gpu"
            )  # type: ignore[call-arg]

        return NormalDist(out_mean, out_var)  # type: ignore[call-arg]
