import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Callable
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normal_dist import NormalDist

SQRT_2PI: float = jnp.sqrt(2 * jnp.pi)

@dataclass
class GPConfig:
    num_inducing_points: int = 10
    z_init_low: float = -2.0
    z_init_high: float = 2.0
    h_init_low: float = -1.0
    h_init_high: float = 1.0
    
    global_length_scale: float = 0.4
    min_length_scale: float = 0.2
    global_covariance_scale: float = 1.0
    min_covariance_scale: float = 0.1
    
    global_jitter: float = 1e-3
    baseline_jitter: float = 1e-2
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GPConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        return {
            'num_inducing_points': self.num_inducing_points,
            'z_init_low': self.z_init_low,
            'z_init_high': self.z_init_high,
            'h_init_low': self.h_init_low,
            'h_init_high': self.h_init_high,
            'global_length_scale': self.global_length_scale,
            'min_length_scale': self.min_length_scale,
            'global_covariance_scale': self.global_covariance_scale,
            'min_covariance_scale': self.min_covariance_scale,
            'global_jitter': self.global_jitter,
            'baseline_jitter': self.baseline_jitter,
        }
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

def normal_pdf(x1: jax.Array, x2: jax.Array, var: jax.Array) -> jax.Array:
    return jnp.exp(-0.5 * (x1 - x2) ** 2 / var) / jnp.sqrt(2 * jnp.pi * var)

def get_kmatrix(
    x1: jax.Array,
    x2: jax.Array,
    func: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    N1 = x1.shape[-1]
    N2 = x2.shape[-1]

    extra_dims = x1.shape[:-1]

    x1_repeat = jnp.repeat(x1, N2, axis=-1).reshape(*extra_dims, N2, N1).transpose(*range(len(extra_dims)), -1, -2).reshape(*extra_dims, N1 * N2)
    x2_repeat = jnp.repeat(x2, N1, axis=-1)

    k_matrix = func(x1_repeat, x2_repeat)
    return k_matrix.reshape(*extra_dims, N1, N2)


class DenseGPLayer:
    """Layer of univariate GP neurons."""

    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        config: Optional[GPConfig] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> None:
        self.I = input_size
        self.O = output_size
        
        self.config = config if config is not None else GPConfig()
        self.P = self.config.num_inducing_points
        self.num_neurons = input_size * output_size
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        z_key, h_key, l_key, s_key, jitter_key = jax.random.split(key, 5)
        
        # Inducing points 
        _z_single_neuron = jnp.linspace(self.config.z_init_low, self.config.z_init_high, self.P)
        _z = jnp.zeros((self.I, self.O, self.P)) + _z_single_neuron[jnp.newaxis, jnp.newaxis, :]
        self.z = _z # (I, O, P)

        # Fcn values @ inducing points
        self.h = jax.random.uniform(
            h_key, 
            (self.I, self.O, self.P), 
            minval=self.config.h_init_low, 
            maxval=self.config.h_init_high
        )  # (I, O, P)

        self.l = jnp.ones((self.I, self.O))  # (I, O)
        self.s = jnp.ones((self.I, self.O))  # (I, O)
        self.jitter = jnp.ones((self.I, self.O))  # (I, O)

        self.reset_gp_hyp()

    # Getters to ensure consistent transformation applied
    def get_jitter(self) -> jax.Array:
        return jnp.exp(self.jitter) + self.config.baseline_jitter

    def get_s(self) -> jax.Array:
        return jnp.exp(self.s) + self.config.min_covariance_scale

    def get_l(self) -> jax.Array:
        return jnp.exp(self.l) + self.config.min_length_scale

    def get_z(self) -> jax.Array:
        return jnp.tanh(self.z)

    def reset_gp_hyp(self):
        z_tanh = self.get_z()
        max_z = jnp.max(z_tanh, axis=-1)
        min_z = jnp.min(z_tanh, axis=-1)
        new_lengthscale = (max_z - min_z) / self.P
        
        self.l = jnp.log(new_lengthscale)  # (I, O)
        self.s = jnp.log(jnp.ones((self.I, self.O)) * self.config.global_covariance_scale)  # (I, O)
        self.jitter = jnp.log(jnp.ones((self.I, self.O)) * self.config.global_jitter)  # (I, O)

    def get_params(self) -> dict:
        return {
            'z': self.z,
            'h': self.h,
            'l': self.l,
            's': self.s,
            'jitter': self.jitter
        }

    def set_params(self, params: dict):
        self.z = params['z']
        self.h = params['h']
        self.l = params['l']
        self.s = params['s']
        self.jitter = params['jitter']

    def forward(self, x: NormalDist) -> NormalDist:
        assert x.mean.ndim == 2
        assert x.mean.shape[1] == self.I

        N = x.mean.shape[0]
        I = self.I
        O = self.O
        P = self.P

        x_mean = x.mean.reshape(N, I, 1)  # (N, I, 1)
        x_var = x.var.reshape(N, I, 1, 1)  # (N, I, 1, 1)

        s = self.get_s().reshape(1, I, O, 1)
        l = self.get_l().reshape(1, I, O, 1)
        jitter = self.get_jitter().reshape(1, I, O, 1, 1)
        z = self.get_z().reshape(1, I, O, P)  # (1, I, O, P)

        kernel_func1 = lambda x1, x2: normal_pdf(x1, x2, x_var + l**2)
        kernel_func2 = lambda x1, x2: normal_pdf(x1, x2, l**2)

        Q_hh = get_kmatrix(z, z, kernel_func2)  # (1, I, O, P, P)
        q_xh = get_kmatrix(
            jnp.repeat(x_mean, self.O, axis=1).reshape(N, I, O, 1), z, kernel_func1
        )  # (N, I, O, 1, P)
        
        Q_hh_noise = Q_hh + (
            (jitter**2)
            / (
                s.reshape(1, I, O, 1, 1) ** 2
                * jnp.abs(l.reshape(1, I, O, 1, 1))
                * SQRT_2PI
            )
        ) * jnp.eye(P)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]

        # L: (1, I, O, P, P)
        L = jax.scipy.linalg.cholesky(Q_hh_noise)
        L_inv = jax.scipy.linalg.inv(L)
        L_inv_T = jnp.transpose(L_inv, (0, 1, 2, 4, 3))
        Q_hh_inv = L_inv_T @ L_inv

        # mean
        # t1: (N, I, O, 1, P)
        t1 = q_xh @ Q_hh_inv
        h = self.h.reshape(1, I, O, P, 1)
        # t2: (N, I, O, 1, 1)
        t2 = t1 @ h
        # out_mean: (N, O)
        out_mean = jnp.sum(t2, axis=1).reshape(N, O)

        # variance
        # A: (N, I, O, 1, P)
        A = q_xh @ L_inv_T
        A_T = jnp.transpose(A, (0, 1, 2, 4, 3))  # (N, I, O, P, 1)
        t3 = (s**2) * (jnp.abs(l) / jnp.sqrt(l**2 + 2 * x_var))  # (N, I, O, 1)
        t4 = SQRT_2PI * (s**2) * jnp.abs(l)  # (1, I, O, 1)
        # t5: (N, I, O, 1, 1)
        t5 = A @ A_T
        t6 = t5.reshape(N, I, O, 1)  # (N, I, O, 1)
        t7 = t3 - t4 * t6 + self.config.global_jitter  # (N, I, O, 1)
        # out_var: (N, O)
        out_var = jnp.sum(t7, axis=1).reshape(N, O)

        return NormalDist(out_mean, out_var)

    def loglikelihood(self) -> jax.Array:
        s = self.get_s().reshape(self.I, self.O, 1)
        l = self.get_l().reshape(self.I, self.O, 1)
        jitter = self.get_jitter().reshape(self.I, self.O, 1, 1)
        z = self.get_z()  # (I, O, P)
        
        def covar_func(x1, x2):
            return s**2 * jnp.exp(-((x1 - x2) ** 2) / (2 * l**2))

        K_hh = get_kmatrix(z, z, covar_func)  # (I, O, P, P)
        K_hh_noise = K_hh + (jitter**2) * jnp.eye(self.P)[jnp.newaxis, jnp.newaxis, :, :]

        # L: (I, O, P, P)
        L = jax.scipy.linalg.cholesky(K_hh_noise)
        L_inv = jax.scipy.linalg.inv(L)
        L_inv_T = jnp.transpose(L_inv, (0, 1, 3, 2))
        # A: (I, O, 1, P)
        h = self.h.reshape(self.I, self.O, 1, self.P)
        A = h @ L_inv_T
        A_T = jnp.transpose(A, (0, 1, 3, 2))
        # t1: (I, O, 1)
        t1 = jnp.log(jnp.linalg.det(L))
        # t2: (I, O, 1, 1)
        t2 = A @ A_T
        loglik = -0.5 * t2 - t1 - self.P * jnp.log(SQRT_2PI)  # (I, O, 1, 1)
        loglik_sum = jnp.sum(loglik)  # (1)

        return loglik_sum / (self.I * self.P) # Avg ll per neuron

    def __gp_dist(self, x: jax.Array, I_idx: int, O_idx: int) -> NormalDist:
        assert x.ndim == 1 and x.shape[0] == 1
        
        l = self.get_l()[I_idx, O_idx]  # (1)
        s = self.get_s()[I_idx, O_idx]  # (1)
        jitter = self.get_jitter()[I_idx, O_idx]  # (1)
        z = self.get_z()[I_idx, O_idx].reshape(1, self.P)  # (1, P)
        h = self.h[I_idx, O_idx].reshape(1, self.P)  # (1, P)
        
        def covar_func(x1, x2):
            return s**2 * jnp.exp(-((x1 - x2) ** 2) / (2 * l**2))

        K_hh = get_kmatrix(z, z, covar_func)  # (1, P, P)
        K_hh_noise = K_hh + (jitter**2) * jnp.eye(self.P)[jnp.newaxis, :, :]
        k_xh = get_kmatrix(x.reshape(1, 1), z, covar_func)  # (1, 1, P)

        L = jax.scipy.linalg.cholesky(K_hh_noise)
        L_inv = jax.scipy.linalg.inv(L)
        L_inv_T = jnp.transpose(L_inv, (0, 2, 1))
        K_hh_inv = L_inv_T @ L_inv

        # mean
        # t1: (1, 1, P)
        t1 = k_xh @ K_hh_inv
        # t2: (1, 1, 1)
        h_reshaped = h.reshape(1, self.P, 1)
        t2 = t1 @ h_reshaped
        mean = t2.reshape(1)

        # variance
        # A: (1, 1, P)
        A = k_xh @ L_inv_T
        A_T = jnp.transpose(A, (0, 2, 1))
        Kxx = covar_func(x, x)  # (1)
        t3 = Kxx - A @ A_T
        var = t3.reshape(1)

        return NormalDist(mean, var)

    def plot_neuron(self, axes: plt.Axes, I_idx: int, O_idx: int):
        """Shows the (I_idx, O_idx)-th neuron's current GP"""
        z = self.get_z()[I_idx, O_idx]  # (P)
        h = self.h[I_idx, O_idx]  # (P)
        NUM_PLT_PTS = 100
        x_pts = jnp.linspace(jnp.min(z) - 1, jnp.max(z) + 1, NUM_PLT_PTS)  # (NUM_PLT_PTS)
        
        gp_dist_pts = [self.__gp_dist(x.reshape(1), I_idx, O_idx) for x in x_pts]

        mean = jnp.array([p.mean for p in gp_dist_pts]).reshape(-1)
        std_dev = jnp.array([jnp.sqrt(p.var) for p in gp_dist_pts]).reshape(-1)

        axes.plot(x_pts, mean, color="black")
        axes.fill_between(
            x_pts, mean + 2 * std_dev, mean - 2 * std_dev, color="gray"
        )

        axes.scatter(z, h, color="red")

    def save_fig(self, path: str):
        MAX_NEURONS_SHOWN = 5
        plot_num = min(self.num_neurons, MAX_NEURONS_SHOWN)

        fig, axes = plt.subplots(1, plot_num, squeeze=False)

        axes_idx = 0
        for i_idx in range(self.I):
            for o_idx in range(self.O):
                if axes_idx < plot_num:
                    self.plot_neuron(axes[0, axes_idx], i_idx, o_idx)
                    axes_idx += 1

        fig.set_figwidth(20)
        fig.set_figheight(5)
        fig.savefig(path)

        plt.close(fig)

    def __repr__(self) -> str:
        return f"DenseGPLayer(in={self.I} out={self.O} gp_pts_per_neuron={self.P})"
    
# Breakpoint testing - temporary
if __name__ == "__main__":
    layer = DenseGPLayer(input_size=2, output_size=3, num_gp_pts=5)
    print(layer)
    print(layer.get_params())
    input_mean = jnp.array([[0.0, 1.0]])
    input_var = jnp.array([[1.0, 1.0]])
    print(layer.forward(NormalDist(input_mean, input_var)))
    print(layer.loglikelihood())
    print(layer.save_fig("test.png"))