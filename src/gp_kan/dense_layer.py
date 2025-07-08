import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Callable
from configparser import ConfigParser

from src.gp_kan.normal_dist import NormalDist, get_device_config, setup_jax_device

SQRT_2PI: float = jnp.sqrt(2 * jnp.pi)

def load_gp_config(config: ConfigParser) -> dict:
    if 'GP' not in config:
        raise ValueError("GP section not found in config")
    
    gp_section = config['GP']
    
    return {
        'num_inducing_points': int(gp_section.get('num_inducing_points', '10')),
        'z_init_low': float(gp_section.get('z_init_low', '-2.0')),
        'z_init_high': float(gp_section.get('z_init_high', '2.0')),
        'h_init_low': float(gp_section.get('h_init_low', '-1.0')),
        'h_init_high': float(gp_section.get('h_init_high', '1.0')),
        'global_length_scale': float(gp_section.get('global_length_scale', '0.4')),
        'min_length_scale': float(gp_section.get('min_length_scale', '0.2')),
        'global_covariance_scale': float(gp_section.get('global_covariance_scale', '1.0')),
        'min_covariance_scale': float(gp_section.get('min_covariance_scale', '0.1')),
        'global_jitter': float(gp_section.get('global_jitter', '0.001')),
        'baseline_jitter': float(gp_section.get('baseline_jitter', '0.01')),
    }

def create_default_conf() -> ConfigParser:
    config = ConfigParser()
    config['GP'] = {
        'num_inducing_points': '10',
        'z_init_low': '-2.0',
        'z_init_high': '2.0',
        'h_init_low': '-1.0',
        'h_init_high': '1.0',
        'global_length_scale': '0.4',
        'min_length_scale': '0.2',
        'global_covariance_scale': '1.0',
        'min_covariance_scale': '0.1',
        'global_jitter': '0.001',
        'baseline_jitter': '0.01'
    }
    config['DEVICE'] = {
        'use_gpu': 'false',
        'device': 'cpu',
        'precision': 'float32'
    }
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
        x1_expanded = x1[:, None]  # (N1, 1)
        x2_expanded = x2[None, :]  # (1, N2)
    elif x1.ndim == x2.ndim:
        if x1.ndim == 1:
            x1_expanded = x1[:, None]  # (N1, 1)
            x2_expanded = x2[None, :]  # (1, N2)
        else:
            x1_expanded = x1[..., :, None]  # (..., N1, 1)
            x2_expanded = x2[..., None, :]  # (..., 1, N2)
    else:
        if x1.ndim > x2.ndim:
            if x2.ndim == 1:
                x1_expanded = x1[..., :, None]  # (..., N1, 1)
                x2_expanded = x2[None, :]  # (1, N2)
            else:
                x1_expanded = x1[..., :, None]  # (..., N1, 1)
                x2_expanded = x2[None, :]  # (1, N2)
        else:
            if x1.ndim == 1:
                x1_expanded = x1[:, None]  # (N1, 1)
                x2_expanded = x2[..., None, :]  # (..., 1, N2)
            else:
                x1_expanded = x1[None, :]  # (1, N1)
                x2_expanded = x2[..., :, None]  # (..., N2, 1)
    
    k_matrix = func(x1_expanded, x2_expanded)
    return k_matrix


class DenseGPLayer:
    """Layer of univariate GP neurons."""

    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        config: Optional[ConfigParser] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> None:
        self.I = input_size
        self.O = output_size
        
        if config is None:
            config = create_default_conf()
        
        self.config = config
        gp_params = load_gp_config(config)
        
        self.device_config = get_device_config(config)
        setup_jax_device(config)
        
        self.P = gp_params['num_inducing_points']
        self.num_neurons = input_size * output_size
        
        self.z_init_low = gp_params['z_init_low']
        self.z_init_high = gp_params['z_init_high']
        self.h_init_low = gp_params['h_init_low']
        self.h_init_high = gp_params['h_init_high']
        self.global_length_scale = gp_params['global_length_scale']
        self.min_length_scale = gp_params['min_length_scale']
        self.global_covariance_scale = gp_params['global_covariance_scale']
        self.min_covariance_scale = gp_params['min_covariance_scale']
        self.global_jitter = gp_params['global_jitter']
        self.baseline_jitter = gp_params['baseline_jitter']
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        h_key = jax.random.split(key, 1)[0]
        
        # Inducing points 
        _z_single_neuron = jnp.linspace(self.z_init_low, self.z_init_high, self.P)
        _z = jnp.zeros((self.I, self.O, self.P)) + _z_single_neuron[jnp.newaxis, jnp.newaxis, :]
        self.z = _z # (I, O, P)

        # Fcn values @ inducing points
        self.h = jax.random.uniform(
            h_key, 
            (self.I, self.O, self.P), 
            minval=self.h_init_low, 
            maxval=self.h_init_high
        )  # (I, O, P)

        self.l = jnp.ones((self.I, self.O))  # (I, O)
        self.s = jnp.ones((self.I, self.O))  # (I, O)
        self.jitter = jnp.ones((self.I, self.O))  # (I, O)

        if self.device_config['use_gpu']:
            self._move_to_gpu()

        self.reset_gp_hyp()

    def _move_to_gpu(self):
        """Move all parameters to GPU."""
        gpu_device = jax.devices('gpu')[0]
        self.z = jax.device_put(self.z, gpu_device)
        self.h = jax.device_put(self.h, gpu_device)
        self.l = jax.device_put(self.l, gpu_device)
        self.s = jax.device_put(self.s, gpu_device)
        self.jitter = jax.device_put(self.jitter, gpu_device)

    # Getters to ensure consistent transformation applied
    def get_jitter(self) -> jax.Array:
        return jnp.exp(self.jitter) + self.baseline_jitter

    def get_s(self) -> jax.Array:
        return jnp.exp(self.s) + self.min_covariance_scale

    def get_l(self) -> jax.Array:
        return jnp.exp(self.l) + self.min_length_scale

    def get_z(self) -> jax.Array:
        return jnp.tanh(self.z)

    def reset_gp_hyp(self):
        z_tanh = self.get_z()
        max_z = jnp.max(z_tanh, axis=-1)
        min_z = jnp.min(z_tanh, axis=-1)
        new_lengthscale = (max_z - min_z) / self.P
        
        self.l = jnp.log(new_lengthscale)  # (I, O)
        self.s = jnp.log(jnp.ones((self.I, self.O)) * self.global_covariance_scale)  # (I, O)
        self.jitter = jnp.log(jnp.ones((self.I, self.O)) * self.global_jitter)  # (I, O)

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

        def kernel_func1(x1, x2):
            N = x_var.shape[0]
            x_var_reshaped = x_var.reshape(N, I, 1, 1, 1)
            var = jnp.broadcast_to(x_var_reshaped, x1.shape)
            l_b = jnp.broadcast_to(l.reshape(1, I, O, 1, 1), x1.shape)
            return normal_pdf(x1, x2, var + l_b**2)
        
        def kernel_func2(x1, x2):
            l_reshaped = l.reshape(1, I, O, 1, 1)  # (1, I, O, 1, 1)
            l_b = jnp.broadcast_to(l_reshaped, x1.shape)
            return normal_pdf(x1, x2, l_b**2)

        Q_hh = build_kernel_mat(z, z, kernel_func2)  # (1, I, O, P, P)
        q_xh = build_kernel_mat(
            jnp.repeat(x_mean, O, axis=2).reshape(N, I, O, 1), z, kernel_func1
        )  # (N, I, O, 1, P)
        
        Q_hh_noise = Q_hh + (
            (jitter**2)
            / (
                s.reshape(1, I, O, 1, 1) ** 2
                * jnp.abs(l.reshape(1, I, O, 1, 1))
                * SQRT_2PI
            )
        ) * jnp.eye(P)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]

        L = jax.scipy.linalg.cholesky(Q_hh_noise)
        L_inv = jax.scipy.linalg.inv(L)
        L_inv_T = jnp.transpose(L_inv, (0, 1, 2, 4, 3)) # (1, I, O, P, P)
        Q_hh_inv = L_inv_T @ L_inv

        t1 = q_xh @ Q_hh_inv # (N, I, O, 1, P)
        h = self.h.reshape(1, I, O, P, 1)
        t2 = t1 @ h # (N, I, O, 1, 1)
        out_mean = jnp.sum(t2, axis=1).reshape(N, O) # (N, O)

        A = q_xh @ L_inv_T
        A_T = jnp.transpose(A, (0, 1, 2, 4, 3)) # (N, I, O, P, 1)
        t5 = A @ A_T
        t6 = t5.reshape(N, I, O, 1) # (N, I, O, 1)
        
        t3 = (s**2) * (jnp.abs(l) / jnp.sqrt(l**2 + 2 * x_var)) # (N, I, O, 1)
        t4 = SQRT_2PI * (s**2) * jnp.abs(l) # (1, I, O, 1)
        t7 = t3 - t4 * t6 + self.global_jitter
        out_var = jnp.sum(t7, axis=1).reshape(N, O)
        
        out_var = jnp.maximum(out_var, 1e-6) # Positive variance

        return NormalDist(out_mean, out_var)

    def loglikelihood(self) -> jax.Array:
        
        # Top level JIT 
        def _loglikelihood_core(s, l, jitter, z, h):
            I, O, P = z.shape
            
            def covar_func(x1, x2):
                # x1, x2: (I, O, P, 1) and (I, O, 1, P)
                l_b = jnp.broadcast_to(l.reshape(I, O, 1, 1), (I, O, P, P))
                s_b = jnp.broadcast_to(s.reshape(I, O, 1, 1), (I, O, P, P))
                return s_b**2 * jnp.exp(-((x1 - x2) ** 2) / (2 * l_b**2))

            K_hh = build_kernel_mat(z, z, covar_func)  # (I, O, P, P)
            K_hh_noise = K_hh + (jitter.reshape(I, O, 1, 1) ** 2) * jnp.eye(P)[jnp.newaxis, jnp.newaxis, :, :] # (I, O, P, P)

            L = jax.scipy.linalg.cholesky(K_hh_noise)
            L_inv = jax.scipy.linalg.inv(L)
            L_inv_T = jnp.transpose(L_inv, (0, 1, 3, 2)) # (I, O, P, P)

            h_ = h.reshape(I, O, 1, P)
            A = h_ @ L_inv_T
            A_T = jnp.transpose(A, (0, 1, 3, 2)) # (I, O, 1, P)

            t1 = jnp.log(jnp.linalg.det(L)) # (I, O, 1)
            t2 = A @ A_T # (I, O, 1, 1)

            loglik = -0.5 * t2 - t1 - P * jnp.log(SQRT_2PI)  # (I, O, 1, 1)
            loglik_sum = jnp.sum(loglik)  # (1)
            
            return loglik_sum
        
        # JIT once
        if not hasattr(self, '_loglikelihood_core_jit'):
            self._loglikelihood_core_jit = jax.jit(_loglikelihood_core)
        
        s = self.get_s()
        l = self.get_l()
        jitter = self.get_jitter()
        z = self.get_z()
        h = self.h
        
        return self._loglikelihood_core_jit(s, l, jitter, z, h)

    def plot_neuron(self, axes: plt.Axes, I_idx: int, O_idx: int, num_pts: int = 100):
        """Plot a single neuron's function."""
        z = self.get_z()[I_idx, O_idx, :]  # (P,)
        h = self.h[I_idx, O_idx, :]  # (P,)
        l = self.get_l()[I_idx, O_idx]  # scalar
        s = self.get_s()[I_idx, O_idx]  # scalar
        jitter = self.get_jitter()[I_idx, O_idx]  # scalar
        
        x_plot = jnp.linspace(jnp.min(z) - 2*l, jnp.max(z) + 2*l, num_pts)
        
        def covar_func(x1, x2):
            return s**2 * jnp.exp(-((x1 - x2) ** 2) / (2 * l**2))

        K_hh = build_kernel_mat(z, z, covar_func)  # (P, P)
        K_hh_noise = K_hh + (jitter**2) * jnp.eye(self.P) # (P, P)

        # L: (P, P)
        L = jax.scipy.linalg.cholesky(K_hh_noise)
        L_inv = jax.scipy.linalg.inv(L)
        L_inv_T = jnp.transpose(L_inv, (1, 0))

        # A: (num_pts, P)
        k_xh = build_kernel_mat(x_plot, z, covar_func)  # (num_pts, P)
        A = k_xh @ L_inv_T
        A_T = jnp.transpose(A, (1, 0))

        # Mean
        h = h.reshape(1, self.P)
        mean = A @ h.T  # (num_pts, 1)
        
        # Variance
        k_xx = build_kernel_mat(x_plot, x_plot, covar_func)  # (num_pts, num_pts)
        var = jnp.diag(k_xx) - jnp.sum(A * A, axis=1)  # (num_pts,)
        
        axes.plot(x_plot, mean.flatten(), 'b-', label='Mean')
        axes.fill_between(x_plot, mean.flatten() - 2*jnp.sqrt(var), 
                         mean.flatten() + 2*jnp.sqrt(var), alpha=0.3, label=r'$\pm 2\sigma$')
        axes.scatter(z, h, c='red', s=50, label='Inducing points')
        axes.legend()
        axes.grid(True, alpha=0.3)

    def save_fig(self, path: str, max_neurons_shown: int = 5):
        """Save a figure showing neuron functions."""
        import matplotlib.pyplot as plt
        
        num_neurons = min(max_neurons_shown, self.num_neurons)
        fig, axes = plt.subplots(1, num_neurons, figsize=(5*num_neurons, 4))
        
        if num_neurons == 1:
            axes = [axes]
        
        for neuron_idx in range(num_neurons):
            i_idx = neuron_idx // self.O
            o_idx = neuron_idx % self.O
            
            if i_idx < self.I:
                self.plot_neuron(axes[neuron_idx], i_idx, o_idx)
                axes[neuron_idx].set_title(f'Neuron ({i_idx},{o_idx})')
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def __repr__(self) -> str:
        return f"DenseGPLayer(I={self.I}, O={self.O}, P={self.P}, GPU={self.device_config['use_gpu']})"
    