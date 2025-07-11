from configparser import ConfigParser
from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc


def get_device_config(config: ConfigParser) -> Dict[str, Union[bool, str]]:
    if "DEVICE" not in config:
        return {"use_gpu": False, "device": "cpu"}

    device_section = config["DEVICE"]
    return {
        "use_gpu": device_section.getboolean("use_gpu", False),
        "device": device_section.get("device", "cpu"),
        "precision": device_section.get("precision", "float32"),
    }


def setup_jax_device(config: ConfigParser) -> None:
    device_config = get_device_config(config)

    if device_config["use_gpu"]:
        jax.config.update("jax_platform_name", "gpu")

        if device_config["precision"] == "float64":
            jax.config.update("jax_enable_x64", True)
        elif device_config["precision"] == "float32":
            jax.config.update("jax_enable_x64", False)
    else:
        jax.config.update("jax_platform_name", "cpu")


@jdc.pytree_dataclass
class NormalDist:
    mean: jax.Array
    var: jax.Array

    def __post_init__(self) -> None:
        if self.mean.shape != self.var.shape:
            raise ValueError(
                f"Mean shape {self.mean.shape} must match "
                f"variance shape {self.var.shape}"
            )

    @classmethod
    def from_array(
        cls, mean: jax.Array, var: Optional[jax.Array] = None
    ) -> "NormalDist":
        """
        Make NormalDist from mean array.

        Parameters:
        -----------
        mean : jax.Array
            Mean values
        var : jax.Array, optional
            Variance values. If None, uses small default variance (1e-6)

        Returns:
        --------
        NormalDist : Normal distribution object
        """
        if var is None:
            var = 1e-6 * jnp.ones_like(mean)
        return cls(mean, var)

    def __add__(self, other: "NormalDist") -> "NormalDist":
        """
        Add two NormalDist.

        Assume independent - the means and variances add.

        Parameters:
        -----------
        other : NormalDist
            Another Normal distribution to add

        Returns:
        --------
        NormalDist : Sum of the two
        """
        if not isinstance(other, NormalDist):
            raise TypeError(f"Cannot add NormalDist with {type(other)}")

        mean = self.mean + other.mean
        var = self.var + other.var
        return NormalDist(mean, var)

    def __repr__(self) -> str:
        return f"NormalDist(mean: {self.mean}, var: {self.var})"

    def sample(
        self, key: jax.Array, shape: Optional[Tuple[int, ...]] = None
    ) -> jax.Array:
        """
        Sample from NormalDist.

        Parameters:
        -----------
        key : jax.random.PRNGKey
            Random key for sampling
        shape : tuple, optional
            Number of samples to generate. If None, uses the shape of mean.
            If shape is provided, output shape will be shape + mean.shape
        Returns:
        --------
        jax.Array : Sampled values
        """
        if shape is None:
            sample_shape = self.mean.shape
        else:
            sample_shape = shape + self.mean.shape

        std_samples = jax.random.normal(key, sample_shape)
        samples = self.mean + jnp.sqrt(self.var) * std_samples
        return samples

    def log_prob(self, x: jax.Array) -> jax.Array:
        """
        Returns normal log_prob(x).

        Parameters:
        -----------
        x : jax.Array
            Points to evaluate log probability at

        Returns:
        --------
        jax.Array : Log probability values
        """
        return -0.5 * (jnp.log(2 * jnp.pi * self.var) + (x - self.mean) ** 2 / self.var)

    @property
    def std(self) -> jax.Array:
        return jnp.sqrt(self.var)

    def to_numpy(self) -> "NormalDist":
        return NormalDist(jnp.array(self.mean), jnp.array(self.var))

    def to_device(self, device: str) -> "NormalDist":
        if device == "gpu":
            mean_gpu = jax.device_put(self.mean, jax.devices("gpu")[0])
            var_gpu = jax.device_put(self.var, jax.devices("gpu")[0])
            return NormalDist(mean_gpu, var_gpu)
        else:
            mean_cpu = jax.device_put(self.mean, jax.devices("cpu")[0])
            var_cpu = jax.device_put(self.var, jax.devices("cpu")[0])
            return NormalDist(mean_cpu, var_cpu)
