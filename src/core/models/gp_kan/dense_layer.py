from configparser import ConfigParser
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp

from ....utils import create_default_config, load_gp_config
from .normal_dist import NormalDist, get_device_config, setup_jax_device

SQRT_2PI = jnp.sqrt(2 * jnp.pi)


class HyperParamsContext:
    GLOBAL_JITTER: float = 1e-3
    BASELINE_JITTER: float = 1e-2

    @classmethod
    def configure(
        cls,
        *,
        global_jitter: Optional[float] = None,
        baseline_jitter: Optional[float] = None,
    ) -> None:
        if global_jitter is not None:
            cls.GLOBAL_JITTER = float(global_jitter)

        if baseline_jitter is not None:
            cls.BASELINE_JITTER = float(baseline_jitter)

    @classmethod
    def to_dict(cls) -> Dict[str, float]:
        return {
            "GLOBAL_JITTER": cls.GLOBAL_JITTER,
            "BASELINE_JITTER": cls.BASELINE_JITTER,
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> None:
        cls.configure(
            global_jitter=values.get("GLOBAL_JITTER"),
            baseline_jitter=values.get("BASELINE_JITTER"),
        )


def normal_pdf(x1: jax.Array, x2: jax.Array, var: jax.Array) -> jax.Array:
    return jnp.exp(-0.5 * (x1 - x2) ** 2 / var) / jnp.sqrt(2 * jnp.pi * var)


def build_kernel_mat(
    x1: jax.Array,
    x2: jax.Array,
    func: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Always returns (..., N1, N2) where N1 = x1.shape[-1], N2 = x2.shape[-1]."""
    x1_expanded = x1[..., :, None]
    x2_expanded = x2[..., None, :]
    return func(x1_expanded, x2_expanded)


class DenseGPLayer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[ConfigParser] = None,
        key: Optional[jax.Array] = None,
    ) -> None:
        self.input_dim = input_size
        self.output_dim = output_size

        if config is None:
            config = create_default_config()

        self.config = config
        gp_params = load_gp_config(config)

        self.device_config = get_device_config(config)
        setup_jax_device(config)

        self.P = gp_params["num_inducing_points"]
        self.num_neurons = input_size * output_size

        self.z_init_low = gp_params["z_init_low"]
        self.z_init_high = gp_params["z_init_high"]
        self.h_init_low = gp_params["h_init_low"]
        self.h_init_high = gp_params["h_init_high"]
        self.global_length_scale = gp_params["global_length_scale"]
        self.min_length_scale = gp_params["min_length_scale"]
        self.global_covariance_scale = gp_params["global_covariance_scale"]
        self.min_covariance_scale = gp_params["min_covariance_scale"]
        self.global_jitter = gp_params["global_jitter"]
        self.baseline_jitter = gp_params["baseline_jitter"]

        HyperParamsContext.configure(
            global_jitter=self.global_jitter, baseline_jitter=self.baseline_jitter
        )

        if key is None:
            key = jax.random.PRNGKey(0)

        h_key = jax.random.split(key, 1)[0]

        # Inducing points
        _z_single_neuron = jnp.linspace(self.z_init_low, self.z_init_high, int(self.P))
        _z = (
            jnp.zeros((self.input_dim, self.output_dim, self.P))
            + _z_single_neuron[jnp.newaxis, jnp.newaxis, :]
        )
        self.z = _z

        # Fcn values @ inducing points
        self.h = jax.random.uniform(
            h_key,
            (self.input_dim, self.output_dim, int(self.P)),
            minval=self.h_init_low,
            maxval=self.h_init_high,
        )

        self.length_scale: jax.Array
        self.s: jax.Array
        self.jitter: jax.Array
        self.reset_gp_hyp()

        if self.device_config["use_gpu"]:
            self._move_to_gpu()

    def _move_to_gpu(self) -> None:
        gpu_device = jax.devices("gpu")[0]
        self.z = jax.device_put(self.z, gpu_device)
        self.h = jax.device_put(self.h, gpu_device)
        self.length_scale = jax.device_put(self.length_scale, gpu_device)
        self.s = jax.device_put(self.s, gpu_device)
        self.jitter = jax.device_put(self.jitter, gpu_device)

    # Getters to ensure consistent transformation applied
    def get_jitter(self) -> jax.Array:
        return jnp.exp(self.jitter) + HyperParamsContext.BASELINE_JITTER

    def get_s(self) -> jax.Array:
        return jnp.exp(self.s) + self.min_covariance_scale

    def get_length_scale(self) -> jax.Array:
        return jnp.exp(self.length_scale) + self.min_length_scale

    def get_z(self) -> jax.Array:
        return jnp.tanh(self.z)

    def reset_gp_hyp(self) -> None:
        z_tanh = self.get_z()
        max_z = jnp.max(z_tanh, axis=-1)
        min_z = jnp.min(z_tanh, axis=-1)
        new_lengthscale = (max_z - min_z) / self.P

        self.length_scale = jnp.log(new_lengthscale)
        self.s = jnp.log(
            jnp.ones((self.input_dim, self.output_dim)) * self.global_covariance_scale
        )
        self.jitter = jnp.log(
            jnp.ones((self.input_dim, self.output_dim))
            * HyperParamsContext.GLOBAL_JITTER
        )

    def get_params(self) -> Dict[str, jax.Array]:
        return {
            "z": self.z,
            "h": self.h,
            "length_scale": self.length_scale,
            "s": self.s,
            "jitter": self.jitter,
        }

    def set_params(self, params: Dict[str, jax.Array]) -> None:
        self.z = params["z"]
        self.h = params["h"]
        self.length_scale = params["length_scale"]
        self.s = params["s"]
        self.jitter = params["jitter"]

    def _reshape_params(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        s = self.get_s().reshape(1, self.input_dim, self.output_dim, 1)
        length_scale = self.get_length_scale().reshape(
            1, self.input_dim, self.output_dim, 1
        )
        jitter = self.get_jitter().reshape(1, self.input_dim, self.output_dim, 1, 1)
        z = self.get_z().reshape(1, self.input_dim, self.output_dim, self.P)
        return s, length_scale, jitter, z

    def _jitter(
        self,
        kernel_matrix: jax.Array,
        jitter: jax.Array,
        s: jax.Array,
        length_scale: jax.Array,
        P: int,
    ) -> jax.Array:
        noise_factor = (jitter**2) / (
            s.reshape(1, self.input_dim, self.output_dim, 1, 1) ** 2
            * jnp.abs(length_scale.reshape(1, self.input_dim, self.output_dim, 1, 1))
            * SQRT_2PI
        )
        identity_matrix = jnp.eye(P)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]
        return kernel_matrix + noise_factor * identity_matrix

    def _scipy_cholesky(
        self, kernel_matrix: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        cholesky_factor = jax.scipy.linalg.cholesky(kernel_matrix, lower=True)
        cholesky_inverse = jax.scipy.linalg.inv(cholesky_factor)
        if kernel_matrix.ndim == 4:
            cholesky_inverse_transpose = jnp.transpose(cholesky_inverse, (0, 1, 3, 2))
        else:
            cholesky_inverse_transpose = jnp.transpose(
                cholesky_inverse, (0, 1, 2, 4, 3)
            )
        return cholesky_factor, cholesky_inverse, cholesky_inverse_transpose

    def _predictive_mean(
        self,
        query_inducing_kernel: jax.Array,
        inducing_kernel_inverse: jax.Array,
        N: int,
        input_dim: int,
        output_dim: int,
        P: int,
    ) -> jax.Array:
        # GP PREDICTION: mu* = K(x*,z) @ K(z,z)^(-1) @ h
        query_inducing_weighted = query_inducing_kernel @ inducing_kernel_inverse
        inducing_function_values = self.h.reshape(1, input_dim, output_dim, P, 1)
        weighted_function_values = query_inducing_weighted @ inducing_function_values
        return weighted_function_values

    def _predictive_variance(
        self,
        query_inducing_kernel: jax.Array,
        cholesky_inverse_transpose: jax.Array,
        s: jax.Array,
        length_scale: jax.Array,
        x_var: jax.Array,
        N: int,
        input_dim: int,
        output_dim: int,
    ) -> jax.Array:
        query_inducing_weighted_for_variance = (
            query_inducing_kernel @ cholesky_inverse_transpose
        )
        query_inducing_weighted_transpose = jnp.transpose(
            query_inducing_weighted_for_variance, (0, 1, 2, 4, 3)
        )

        # s^2 * |l| / sqrt(l^2 + 2 * x_var)
        signal_variance_component = (s**2) * (
            jnp.abs(length_scale) / jnp.sqrt(length_scale**2 + 2 * x_var)
        )
        length_scale_scaling_factor = SQRT_2PI * (s**2) * jnp.abs(length_scale)

        # Uncertainty reduction from inducing points 2*pi*s^4*l^2 q_xh K_hh^(-1) q_hx
        uncertainty_reduction_matrix = (
            query_inducing_weighted_for_variance @ query_inducing_weighted_transpose
        )
        uncertainty_reduction_reshaped = uncertainty_reduction_matrix.reshape(
            N, input_dim, output_dim, 1
        )

        predictive_variance_per_neuron = (
            signal_variance_component
            - length_scale_scaling_factor * uncertainty_reduction_reshaped
            + HyperParamsContext.GLOBAL_JITTER
        )
        return predictive_variance_per_neuron

    def forward(self, x: NormalDist) -> NormalDist:
        mean = x.mean
        var = x.var
        if mean.ndim == 1:
            mean = mean.reshape(-1, self.input_dim)
            var = var.reshape(-1, self.input_dim)
        if mean.ndim != 2 or mean.shape[1] != self.input_dim:
            raise ValueError(
                f"Input mean must have shape (N, {self.input_dim}), got {mean.shape}"
            )
        N = mean.shape[0]
        input_dim = self.input_dim
        output_dim = self.output_dim
        P = self.P
        x_mean = mean.reshape(N, input_dim, 1)
        x_var = var.reshape(N, input_dim, 1, 1)
        s, length_scale, jitter, z = self._reshape_params()

        # INNER PRODUCT: int N(x|mu,s^2) k(x,z) dx = k(mu,z)*exp(s^2/(2l^2))
        def kernel_with_var(x1: jax.Array, x2: jax.Array) -> jax.Array:
            N = x_var.shape[0]
            x_var_reshaped = x_var.reshape(N, input_dim, 1, 1, 1)
            input_variance = jnp.broadcast_to(x_var_reshaped, x1.shape)
            length_scale_b = jnp.broadcast_to(
                length_scale.reshape(1, input_dim, output_dim, 1, 1), x1.shape
            )
            return normal_pdf(x1, x2, input_variance + length_scale_b**2)

        # Kernel for inducing points (no input uncertainty)
        def kernel_inducing_points(x1: jax.Array, x2: jax.Array) -> jax.Array:
            length_scale_reshaped = length_scale.reshape(1, input_dim, output_dim, 1, 1)
            length_scale_b = jnp.broadcast_to(length_scale_reshaped, x1.shape)
            return normal_pdf(x1, x2, length_scale_b**2)

        # Build kernel matrices
        inducing_kernel_matrix = build_kernel_mat(z, z, kernel_inducing_points)
        query_inducing_kernel_matrix = build_kernel_mat(
            jnp.repeat(x_mean, output_dim, axis=2).reshape(N, input_dim, output_dim, 1),
            z,
            kernel_with_var,  # With input uncertainty
        )
        inducing_kernel_with_noise = self._jitter(
            inducing_kernel_matrix, jitter, s, length_scale, int(P)
        )
        cholesky_factor, cholesky_inverse, cholesky_inverse_transpose = (
            self._scipy_cholesky(inducing_kernel_with_noise)
        )
        inducing_kernel_inverse = cholesky_inverse_transpose @ cholesky_inverse
        weighted_function_values = self._predictive_mean(
            query_inducing_kernel_matrix,
            inducing_kernel_inverse,
            N,
            input_dim,
            output_dim,
            int(P),
        )
        out_mean = jnp.sum(weighted_function_values, axis=1).reshape(N, output_dim)
        predictive_variance_per_neuron = self._predictive_variance(
            query_inducing_kernel_matrix,
            cholesky_inverse_transpose,
            s,
            length_scale,
            x_var,
            N,
            input_dim,
            output_dim,
        )
        out_var = jnp.sum(predictive_variance_per_neuron, axis=1).reshape(N, output_dim)
        return NormalDist(out_mean, out_var)

    def loglikelihood(self) -> jax.Array:

        # Top level JIT
        def _loglikelihood_core(
            s: jax.Array,
            length_scale: jax.Array,
            jitter: jax.Array,
            z: jax.Array,
            h: jax.Array,
        ) -> jax.Array:
            n_in, n_out, P = z.shape

            def gaussian_kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
                signal_variance = s.reshape(n_in, n_out, 1, 1)
                length_scale_sq = length_scale.reshape(n_in, n_out, 1, 1) ** 2
                sq_dist = (x1 - x2) ** 2
                return signal_variance**2 * jnp.exp(-sq_dist / (2 * length_scale_sq))

            inducing_kernel_matrix = build_kernel_mat(z, z, gaussian_kernel)
            inducing_kernel_with_noise = (
                inducing_kernel_matrix
                + (jitter.reshape(n_in, n_out, 1, 1) ** 2)
                * jnp.eye(P)[jnp.newaxis, jnp.newaxis, :, :]
            )

            cholesky_factor, cholesky_inverse, cholesky_inverse_transpose = (
                self._scipy_cholesky(inducing_kernel_with_noise)
            )

            inducing_function_values = h.reshape(n_in, n_out, 1, P)
            weighted_function_values = (
                inducing_function_values @ cholesky_inverse_transpose
            )
            weighted_function_values_transpose = jnp.transpose(
                weighted_function_values, (0, 1, 3, 2)
            )

            log_determinant = jnp.log(jnp.linalg.det(cholesky_factor))
            quadratic_term = (
                weighted_function_values @ weighted_function_values_transpose
            )

            log_likelihood_per_neuron = (
                -0.5 * quadratic_term - log_determinant - P * jnp.log(SQRT_2PI)
            )
            total_log_likelihood = jnp.sum(log_likelihood_per_neuron)

            return total_log_likelihood / (n_in * P)

        # JIT once
        if not hasattr(self, "_loglikelihood_core_jit"):
            self._loglikelihood_core_jit = jax.jit(_loglikelihood_core)

        s = self.get_s()
        length_scale = self.get_length_scale()
        jitter = self.get_jitter()
        z = self.get_z()
        h = self.h

        result: jax.Array = self._loglikelihood_core_jit(s, length_scale, jitter, z, h)
        return result

    def __repr__(self) -> str:
        return (
            f"DenseGPLayer(I={self.input_dim}, O={self.output_dim}, "
            f"P={self.P}, GPU={self.device_config['use_gpu']})"
        )
