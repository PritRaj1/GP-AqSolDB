from configparser import ConfigParser
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.gp_kan.normal_dist import NormalDist, get_device_config, setup_jax_device

SQRT_2PI: float = jnp.sqrt(2 * jnp.pi)


def load_gp_config(config: ConfigParser) -> dict:
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
    """Layer of univariate GP neurons."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[ConfigParser] = None,
        key: Optional[jax.random.PRNGKey] = None,
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
        _z_single_neuron = jnp.linspace(self.z_init_low, self.z_init_high, self.P)
        _z = (
            jnp.zeros((self.input_dim, self.output_dim, self.P))
            + _z_single_neuron[jnp.newaxis, jnp.newaxis, :]
        )
        self.z = _z

        # Fcn values @ inducing points
        self.h = jax.random.uniform(
            h_key,
            (self.input_dim, self.output_dim, self.P),
            minval=self.h_init_low,
            maxval=self.h_init_high,
        )

        self.length_scale = jnp.ones((self.input_dim, self.output_dim))
        self.s = jnp.ones((self.input_dim, self.output_dim))
        self.jitter = jnp.ones((self.input_dim, self.output_dim))

        if self.device_config["use_gpu"]:
            self._move_to_gpu()

        self.reset_gp_hyp()

    def _move_to_gpu(self):
        """Move all parameters to GPU."""
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

    def reset_gp_hyp(self):
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

    def get_params(self) -> dict:
        return {
            "z": self.z,
            "h": self.h,
            "length_scale": self.length_scale,
            "s": self.s,
            "jitter": self.jitter,
        }

    def set_params(self, params: dict):
        self.z = params["z"]
        self.h = params["h"]
        self.length_scale = params["length_scale"]
        self.s = params["s"]
        self.jitter = params["jitter"]

    def forward(self, x: NormalDist) -> NormalDist:
        assert x.mean.ndim == 2
        assert x.mean.shape[1] == self.input_dim

        N = x.mean.shape[0]
        input_dim = self.input_dim
        output_dim = self.output_dim
        P = self.P

        x_mean = x.mean.reshape(N, input_dim, 1)
        x_var = x.var.reshape(N, input_dim, 1, 1)

        s = self.get_s().reshape(1, input_dim, output_dim, 1)
        length_scale = self.get_length_scale().reshape(1, input_dim, output_dim, 1)
        jitter = self.get_jitter().reshape(1, input_dim, output_dim, 1, 1)
        z = self.get_z().reshape(1, input_dim, output_dim, P)

        def kernel_func1(x1, x2):
            N = x_var.shape[0]
            x_var_reshaped = x_var.reshape(N, input_dim, 1, 1, 1)
            var = jnp.broadcast_to(x_var_reshaped, x1.shape)
            length_scale_b = jnp.broadcast_to(
                length_scale.reshape(1, input_dim, output_dim, 1, 1), x1.shape
            )
            return normal_pdf(x1, x2, var + length_scale_b**2)

        def kernel_func2(x1, x2):
            length_scale_reshaped = length_scale.reshape(1, input_dim, output_dim, 1, 1)
            length_scale_b = jnp.broadcast_to(length_scale_reshaped, x1.shape)
            return normal_pdf(x1, x2, length_scale_b**2)

        Q_hh = build_kernel_mat(z, z, kernel_func2)
        q_xh = build_kernel_mat(
            jnp.repeat(x_mean, output_dim, axis=2).reshape(N, input_dim, output_dim, 1),
            z,
            kernel_func1,
        )

        Q_hh_noise = (
            Q_hh
            + (
                (jitter**2)
                / (
                    s.reshape(1, input_dim, output_dim, 1, 1) ** 2
                    * jnp.abs(length_scale.reshape(1, input_dim, output_dim, 1, 1))
                    * SQRT_2PI
                )
            )
            * jnp.eye(P)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]
        )

        L = jax.scipy.linalg.cholesky(Q_hh_noise)
        L_inv = jax.scipy.linalg.inv(L)
        L_inv_T = jnp.transpose(L_inv, (0, 1, 2, 4, 3))
        Q_hh_inv = L_inv_T @ L_inv

        t1 = q_xh @ Q_hh_inv
        h = self.h.reshape(1, input_dim, output_dim, P, 1)
        t2 = t1 @ h
        out_mean = jnp.sum(t2, axis=1).reshape(N, output_dim)

        A = q_xh @ L_inv_T
        t5 = A @ jnp.transpose(A, (0, 1, 2, 4, 3))
        t6 = t5.reshape(N, input_dim, output_dim, 1)

        t3 = (s**2) * (jnp.abs(length_scale) / jnp.sqrt(length_scale**2 + 2 * x_var))
        t4 = SQRT_2PI * (s**2) * jnp.abs(length_scale)
        t7 = t3 - t4 * t6 + self.global_jitter
        out_var = jnp.sum(t7, axis=1).reshape(N, output_dim)

        out_var = jnp.maximum(out_var, 1e-6)  # Positive variance

        return NormalDist(out_mean, out_var)

    def loglikelihood(self) -> jax.Array:

        # Top level JIT
        def _loglikelihood_core(s, length_scale, jitter, z, h):
            I, O, P = z.shape

            def covar_func(x1, x2):
                length_scale_b = jnp.broadcast_to(
                    length_scale.reshape(I, O, 1, 1), (I, O, P, P)
                )
                s_b = jnp.broadcast_to(s.reshape(I, O, 1, 1), (I, O, P, P))
                return s_b**2 * jnp.exp(-((x1 - x2) ** 2) / (2 * length_scale_b**2))

            K_hh = build_kernel_mat(z, z, covar_func)
            K_hh_noise = (
                K_hh
                + (jitter.reshape(I, O, 1, 1) ** 2)
                * jnp.eye(P)[jnp.newaxis, jnp.newaxis, :, :]
            )

            L = jax.scipy.linalg.cholesky(K_hh_noise)
            L_inv = jax.scipy.linalg.inv(L)
            L_inv_T = jnp.transpose(L_inv, (0, 1, 3, 2))

            h_ = h.reshape(I, O, 1, P)
            A = h_ @ L_inv_T
            A_T = jnp.transpose(A, (0, 1, 3, 2))

            t1 = jnp.log(jnp.linalg.det(L))
            t2 = A @ A_T

            loglik = -0.5 * t2 - t1 - P * jnp.log(SQRT_2PI)
            loglik_sum = jnp.sum(loglik)

            return loglik_sum

        # JIT once
        if not hasattr(self, "_loglikelihood_core_jit"):
            self._loglikelihood_core_jit = jax.jit(_loglikelihood_core)

        s = self.get_s()
        length_scale = self.get_length_scale()
        jitter = self.get_jitter()
        z = self.get_z()
        h = self.h

        return self._loglikelihood_core_jit(s, length_scale, jitter, z, h)

    def plot_neuron(self, axes: plt.Axes, I_idx: int, O_idx: int, num_pts: int = 100):
        """Plot a single neuron's function."""
        z = self.get_z()[I_idx, O_idx, :]
        jitter = self.get_jitter()[I_idx, O_idx]

        x_plot = jnp.linspace(
            jnp.min(z) - 2 * self.length_scale,
            jnp.max(z) + 2 * self.length_scale,
            num_pts,
        )

        def covar_func(x1, x2):
            return self.s**2 * jnp.exp(-((x1 - x2) ** 2) / (2 * self.length_scale**2))

        # Standard GP variance calculation
        K_hh = build_kernel_mat(z, z, covar_func)
        K_hh_noise = K_hh + (jitter**2) * jnp.eye(self.P)

        L = jax.scipy.linalg.cholesky(K_hh_noise)
        L_inv = jax.scipy.linalg.inv(L)
        L_inv_T = jnp.transpose(L_inv, (1, 0))

        k_xh = build_kernel_mat(x_plot, z, covar_func)
        A = k_xh @ L_inv_T

        # Mean
        h = self.h.reshape(1, self.P)
        mean = A @ h.T

        # Variance
        k_xx = build_kernel_mat(x_plot, x_plot, covar_func)
        var = jnp.diag(k_xx) - jnp.sum(A * A, axis=1)

        axes.plot(x_plot, mean.flatten(), "b-", label="Mean")
        axes.fill_between(
            x_plot,
            mean.flatten() - 2 * jnp.sqrt(var),
            mean.flatten() + 2 * jnp.sqrt(var),
            alpha=0.3,
            label=r"$\pm 2\sigma$",
        )
        axes.scatter(z, h, c="red", s=50, label="Inducing points")
        axes.legend()
        axes.grid(True, alpha=0.3)

    def save_fig(self, path: str, max_neurons_shown: int = 5):
        """Save a figure showing neuron functions."""

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
