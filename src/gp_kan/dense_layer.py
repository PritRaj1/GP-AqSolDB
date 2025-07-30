from configparser import ConfigParser
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.gp_kan.normal_dist import NormalDist, get_device_config, setup_jax_device

SQRT_2PI = jnp.sqrt(2 * jnp.pi)


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


def create_default_conf() -> ConfigParser:
    config = ConfigParser()
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
    config["DEVICE"] = {"use_gpu": "false", "device": "cpu", "precision": "float32"}
    return config


def normal_pdf(x1: jax.Array, x2: jax.Array, var: jax.Array) -> jax.Array:
    return jnp.exp(-0.5 * (x1 - x2) ** 2 / var) / jnp.sqrt(2 * jnp.pi * var)


def build_kernel_mat(
    x1: jax.Array,
    x2: jax.Array,
    func: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Always returns (..., N1, N2) where N1 = x1.shape[-1], N2 = x2.shape[-1]."""
    if x1.ndim == 1 and x2.ndim == 1:
        x1_expanded = x1[:, None]
        x2_expanded = x2[None, :]
    elif x1.ndim == x2.ndim:
        if x1.ndim == 1:
            x1_expanded = x1[:, None]
            x2_expanded = x2[None, :]
        else:
            x1_expanded = x1[..., :, None]
            x2_expanded = x2[..., None, :]
    else:
        if x1.ndim > x2.ndim:
            if x2.ndim == 1:
                x1_expanded = x1[..., :, None]
                x2_expanded = x2[None, :]
            else:
                x1_expanded = x1[..., :, None]
                x2_expanded = x2[None, :]
        else:
            if x1.ndim == 1:
                x1_expanded = x1[:, None]
                x2_expanded = x2[..., None, :]
            else:
                x1_expanded = x1[None, :]
                x2_expanded = x2[..., :, None]

    k_matrix = func(x1_expanded, x2_expanded)
    return k_matrix


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
            config = create_default_conf()

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

        self.length_scale = jnp.ones((self.input_dim, self.output_dim))
        self.s = jnp.ones((self.input_dim, self.output_dim))
        self.jitter = jnp.ones((self.input_dim, self.output_dim))

        if self.device_config["use_gpu"]:
            self._move_to_gpu()

        self.reset_gp_hyp()

    def _move_to_gpu(self) -> None:
        gpu_device = jax.devices("gpu")[0]
        self.z = jax.device_put(self.z, gpu_device)
        self.h = jax.device_put(self.h, gpu_device)
        self.length_scale = jax.device_put(self.length_scale, gpu_device)
        self.s = jax.device_put(self.s, gpu_device)
        self.jitter = jax.device_put(self.jitter, gpu_device)

    # Getters to ensure consistent transformation applied
    def get_jitter(self) -> jax.Array:
        return jnp.exp(self.jitter) + self.baseline_jitter

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
            jnp.ones((self.input_dim, self.output_dim)) * self.global_jitter
        )

    def get_params(self) -> Dict[str, jax.Array]:
        return {
            "z": self.z,
            "h": self.h,
            "l": self.length_scale,  # Alias
            "s": self.s,
            "jitter": self.jitter,
        }

    def set_params(self, params: Dict[str, jax.Array]) -> None:
        self.z = params["z"]
        self.h = params["h"]
        self.s = params["s"]
        self.jitter = params["jitter"]

        # Handle both "length_scale" and "l"
        if "length_scale" in params:
            self.length_scale = params["length_scale"]
        elif "l" in params:
            self.length_scale = params["l"]

    def _build_kernel_fcn(
        self, s: jax.Array, length_scale: jax.Array
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        def kernel_fcn(x1: jax.Array, x2: jax.Array) -> jax.Array:
            return s**2 * jnp.exp(-((x1 - x2) ** 2) / (2 * length_scale**2))

        return kernel_fcn

    def _cholesky_decomposition(
        self, K: jax.Array, jitter: jax.Array, P: int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if K.ndim == 2:
            K_noise = K + (jitter**2) * jnp.eye(P)
            L = jax.scipy.linalg.cholesky(K_noise)
            L_inv = jax.scipy.linalg.inv(L)
            L_inv_T = jnp.transpose(L_inv, (1, 0))
        else:
            K_noise = (
                K
                + (jitter**2) * jnp.eye(P)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]
            )
            L = jax.scipy.linalg.cholesky(K_noise)
            L_inv = jax.scipy.linalg.inv(L)
            L_inv_T = jnp.transpose(L_inv, (0, 1, 2, 4, 3))
        return L, L_inv, L_inv_T

    def _predict(
        self,
        x_query: jax.Array,
        z: jax.Array,
        h: jax.Array,
        kernel_fcn: Callable[[jax.Array, jax.Array], jax.Array],
        jitter: jax.Array,
        P: int,
    ) -> tuple[jax.Array, jax.Array]:

        K_hh = build_kernel_mat(z, z, kernel_fcn)
        k_xh = build_kernel_mat(x_query, z, kernel_fcn)

        _, _, L_inv_T = self._cholesky_decomposition(K_hh, jitter, P)

        A = k_xh @ L_inv_T
        mean = (A @ h.reshape(-1, 1)).flatten()

        k_xx_diag = jnp.array([kernel_fcn(x, x) for x in x_query])
        var = k_xx_diag - jnp.sum(A * A, axis=1)

        return mean, var

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
        cholesky_factor = jax.scipy.linalg.cholesky(kernel_matrix)
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
        # GP PREDICTION: μ* = K(x*,z) @ K(z,z)^(-1) @ h
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

        # Uncertainty reduction from inducing points 2πs⁴l² q_xh K_hh⁻¹ q_hx
        uncertainty_reduction_matrix = (
            query_inducing_weighted_for_variance @ query_inducing_weighted_transpose
        )
        uncertainty_reduction_reshaped = uncertainty_reduction_matrix.reshape(
            N, input_dim, output_dim, 1
        )

        predictive_variance_per_neuron = (
            signal_variance_component
            - length_scale_scaling_factor * uncertainty_reduction_reshaped
            + self.global_jitter
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

        # FUNCTION INNER PRODUCT: ∫ N(x|μ, σ²) k(x, z) dx = k(μ, z) * exp(σ²/(2ℓ²))
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
        out_var = jnp.maximum(out_var, 1e-6)  # Positive variance
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
            I, O, P = z.shape

            def gaussian_kernel(x1: jax.Array, x2: jax.Array) -> jax.Array:
                signal_variance = s.reshape(I, O, 1, 1)
                length_scale_sq = length_scale.reshape(I, O, 1, 1) ** 2
                return signal_variance**2 * normal_pdf(x1, x2, length_scale_sq)

            inducing_kernel_matrix = build_kernel_mat(z, z, gaussian_kernel)
            inducing_kernel_with_noise = (
                inducing_kernel_matrix
                + (jitter.reshape(I, O, 1, 1) ** 2)
                * jnp.eye(P)[jnp.newaxis, jnp.newaxis, :, :]
            )

            cholesky_factor, cholesky_inverse, cholesky_inverse_transpose = (
                self._scipy_cholesky(inducing_kernel_with_noise)
            )

            inducing_function_values = h.reshape(I, O, 1, P)
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

            return total_log_likelihood / (I * P)  # Expected log-likelihood per neuron

        # JIT once
        if not hasattr(self, "_loglikelihood_core_jit"):
            self._loglikelihood_core_jit = jax.jit(_loglikelihood_core)

        s = self.get_s()
        length_scale = self.get_length_scale()
        jitter = self.get_jitter()
        z = self.get_z()
        h = self.h

        result = self._loglikelihood_core_jit(s, length_scale, jitter, z, h)
        return jnp.asarray(result)

    def plot_neuron(
        self, axes: plt.Axes, I_idx: int, O_idx: int, num_pts: int = 100
    ) -> None:
        inducing_points = self.get_z()[I_idx, O_idx, :]
        inducing_function_values = self.h[I_idx, O_idx, :]
        signal_variance = self.get_s()[I_idx, O_idx]
        length_scale = self.get_length_scale()[I_idx, O_idx]
        jitter = self.get_jitter()[I_idx, O_idx]

        plot_min = float(jnp.min(inducing_points) - 2 * length_scale)
        plot_max = float(jnp.max(inducing_points) + 2 * length_scale)
        query_points = jnp.linspace(plot_min, plot_max, num_pts)

        kernel_function = self._build_kernel_fcn(signal_variance, length_scale)

        predictive_mean, predictive_variance = self._predict(
            query_points,
            inducing_points,
            inducing_function_values,
            kernel_function,
            jitter,
            int(self.P),
        )
        predictive_std = jnp.sqrt(predictive_variance)

        axes.plot(np.array(query_points), np.array(predictive_mean), color="black")
        axes.fill_between(
            np.array(query_points),
            np.array(predictive_mean) + 2 * np.array(predictive_std),
            np.array(predictive_mean) - 2 * np.array(predictive_std),
            color="gray",
            alpha=0.3,
        )
        axes.scatter(
            np.array(inducing_points), np.array(inducing_function_values), color="red"
        )
        axes.grid(True, alpha=0.3)

    def save_fig(self, path: str, max_neurons_shown: int = 5) -> None:
        plot_num = min(max_neurons_shown, self.num_neurons)

        num_cols = min(plot_num, 5)
        num_rows = (plot_num + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, squeeze=False)

        axes_idx = 0
        for i_idx in range(self.input_dim):
            for o_idx in range(self.output_dim):
                if axes_idx < plot_num:
                    row_idx = axes_idx // num_cols
                    col_idx = axes_idx % num_cols
                    self.plot_neuron(axes[row_idx, col_idx], i_idx, o_idx)
                    axes_idx += 1

        for i in range(plot_num, num_rows * num_cols):
            row_idx = i // num_cols
            col_idx = i % num_cols
            axes[row_idx, col_idx].set_visible(False)

        fig.set_figwidth(20)
        fig.set_figheight(5)
        fig.savefig(path)
        plt.close()

    def __repr__(self) -> str:
        return (
            f"DenseGPLayer(I={self.input_dim}, O={self.output_dim}, "
            f"P={self.P}, GPU={self.device_config['use_gpu']})"
        )
