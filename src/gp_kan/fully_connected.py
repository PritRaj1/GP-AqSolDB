import jax
from typing import List, Dict, Any
from configparser import ConfigParser
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit
import optax

from src.gp_kan.normal_dist import NormalDist, get_device_config, setup_jax_device
from src.gp_kan.dense_layer import DenseGPLayer
from src.gp_kan.invariant_acts import NormaliseGaussian

def load_kan_conf(config: ConfigParser) -> dict:
    if 'NETWORK' not in config:
        raise ValueError("NETWORK section not found in config")
    if 'GP' not in config:
        raise ValueError("GP section not found in config")
    if 'NORMALIZATION' not in config:
        raise ValueError("NORMALIZATION section not found in config")
    if 'TRAINING' not in config:
        raise ValueError("TRAINING section not found in config")
    
    network_section = config['NETWORK']
    normalization_section = config['NORMALIZATION']
    training_section = config['TRAINING']
    
    return {
        'input_size': int(network_section.get('input_size')),
        'output_size': int(network_section.get('output_size')),
        'num_inducing_points': int(config['GP'].get('num_inducing_points', '10')),
        'z_init_low': float(config['GP'].get('z_init_low', '-2.0')),
        'z_init_high': float(config['GP'].get('z_init_high', '2.0')),
        'h_init_low': float(config['GP'].get('h_init_low', '-1.0')),
        'h_init_high': float(config['GP'].get('h_init_high', '1.0')),
        'global_length_scale': float(config['GP'].get('global_length_scale', '0.4')),
        'min_length_scale': float(config['GP'].get('min_length_scale', '0.2')),
        'global_covariance_scale': float(config['GP'].get('global_covariance_scale', '1.0')),
        'min_covariance_scale': float(config['GP'].get('min_covariance_scale', '0.1')),
        'global_jitter': float(config['GP'].get('global_jitter', '0.001')),
        'baseline_jitter': float(config['GP'].get('baseline_jitter', '0.01')),
        'min_var': float(normalization_section.get('min_var', '0.2')),
        'seed': int(training_section.get('seed', '42'))
    }

def create_default_conf() -> ConfigParser:
    config = ConfigParser()
    config['NETWORK'] = {
        'input_size': '3',
        'output_size': '1'
    }
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
    config['NORMALIZATION'] = {
        'min_var': '0.2'
    }
    config['TRAINING'] = {
        'seed': '42'
    }
    config['DEVICE'] = {
        'use_gpu': 'false',
        'device': 'cpu',
        'precision': 'float32'
    }
    return config


class GP_KAN:
    """Gaussian Process Kolmogorov-Arnold Network."""
    
    def __init__(self, config: ConfigParser, hidden_sizes: List[int] = None):
        self.config = config
        self.layers = []
        self.normalizers = []
        
        self.device_config = get_device_config(config)
        setup_jax_device(config)
        
        config_params = load_kan_conf(config)
        self.input_size = config_params['input_size']
        self.output_size = config_params['output_size']
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else []
        self.seed = config_params['seed']
        
        self._build_network()
    
    def _build_network(self):
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            layer = DenseGPLayer(
                input_size=input_size,
                output_size=output_size,
                config=self.config,
                key=jax.random.PRNGKey(self.seed + i)
            )
            self.layers.append(layer)
            
            if i < len(layer_sizes) - 2:
                min_var = float(self.config['NORMALIZATION'].get('min_var', '0.2'))
                normalizer = NormaliseGaussian(min_var=min_var, config=self.config)
                self.normalizers.append(normalizer)
            else:
                self.normalizers.append(None)

        if self.device_config['use_gpu']:
            self._loglikelihood_jit = jax.jit(self.loglikelihood)
    
    def forward(self, x: NormalDist) -> NormalDist:
        current = x
        
        for i, (layer, normalizer) in enumerate(zip(self.layers, self.normalizers)):
            current = layer.forward(current)
            
            if normalizer is not None:
                current = normalizer(current)
        
        return current
    
    def _forward_jit(self, x_mean: jax.Array, x_var: jax.Array) -> tuple[jax.Array, jax.Array]:
        
        # JIT-able array-heavy part
        def _forward_core(x_mean, x_var):
            current_mean = x_mean
            current_var = x_var
            
            for i, (layer, normalizer) in enumerate(zip(self.layers, self.normalizers)):
                current_dist = NormalDist(current_mean, current_var)
                
                layer_output = layer.forward(current_dist)
                current_mean = layer_output.mean
                current_var = layer_output.var
                
                if normalizer is not None:
                    normalizer_output = normalizer(layer_output)
                    current_mean = normalizer_output.mean
                    current_var = normalizer_output.var
            
            return current_mean, current_var
        
        # JIT once
        if not hasattr(self, '_forward_core_jit'):
            self._forward_core_jit = jax.jit(_forward_core)
        
        return self._forward_core_jit(x_mean, x_var)
    
    def get_params(self) -> Dict[str, Any]:
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'layer_{i}'] = layer.get_params()
        return params
    
    def set_params(self, params: Dict[str, Any]):
        for i, layer in enumerate(self.layers):
            if f'layer_{i}' in params:
                layer.set_params(params[f'layer_{i}'])
    
    def loglikelihood(self) -> jax.Array:
        total_ll = 0.0
        for layer in self.layers:
            total_ll += layer.loglikelihood()
        return total_ll
    
    def train(self, X_train: jax.Array, y_train: jax.Array, 
              X_val: jax.Array = None, y_val: jax.Array = None,
              learning_rate: float = 0.001, num_epochs: int = 30, 
              batch_size: int = 32, patience: int = 10, pretrain_iters: int = 10):
        """
        Train the GP-KAN network using gradient descent.
        
        Parameters:
        -----------
        X_train : jax.Array
            Training features
        y_train : jax.Array
            Training targets
        X_val : jax.Array, optional
            Validation features
        y_val : jax.Array, optional
            Validation targets
        learning_rate : float
            Learning rate for optimization
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Early stopping patience
        pretrain_iters : int
            Number of pretraining iterations for GP hyperparameters
        """
        X_train = jnp.array(X_train, dtype=jnp.float32)
        y_train = jnp.array(y_train, dtype=jnp.float32).reshape(-1, 1)
        
        if X_val is not None:
            X_val = jnp.array(X_val, dtype=jnp.float32)
            y_val = jnp.array(y_val, dtype=jnp.float32).reshape(-1, 1)
        
        # Pretrain hps
        print("Pretraining GP hyperparameters...")
        self._pretrain_gp_hyperparameters(X_train, y_train, pretrain_iters)
        
        optimizer = optax.adam(learning_rate)        
        params = self.get_params()
        opt_state = optimizer.init(params)
        
        key = jax.random.PRNGKey(self.seed)
        
        def loss_fn(params, X_batch, y_batch):
            self.set_params(params)
            
            X_batch_mean = X_batch
            X_batch_var = jnp.zeros_like(X_batch)
            X_batch_dist = NormalDist(X_batch_mean, X_batch_var)
            
            output_dist = self.forward(X_batch_dist)
            
            loglik = self._log_likelihood(output_dist.mean, output_dist.var, y_batch)
            negloglik = -jnp.mean(loglik)
            
            return negloglik
        
        grad_fn = jit(grad(loss_fn))
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            key, subkey = jax.random.split(key)
            indices = jax.random.permutation(subkey, len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                grads = grad_fn(params, X_batch, y_batch)                
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                
                loss = loss_fn(params, X_batch, y_batch)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if X_val is not None:
                val_loss = loss_fn(params, X_val, y_val)
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_params = params.copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.set_params(best_params)
                    break
            else:
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
        
        self.set_params(params)
    
    def _pretrain_gp_hyperparameters(self, X_train: jax.Array, y_train: jax.Array, num_iters: int):
        """Pretrain GP hyperparameters by maximizing ll of inducing points."""
        
        def pretrain_loss_fn(params):
            self.set_params(params)
            return -self.loglikelihood()
        
        optimizer = optax.adam(0.001)
        params = self.get_params()
        opt_state = optimizer.init(params)
        
        for i in range(num_iters):
            grads = grad(pretrain_loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            if i % 2 == 0:
                loss = pretrain_loss_fn(params)
                print(f"  Pretrain {i}: internal loglik {-loss:.4f}")
        
        self.set_params(params)
    
    def _log_likelihood(self, mean: jax.Array, var: jax.Array, targets: jax.Array) -> jax.Array:
        return -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * (targets - mean) ** 2 / var
    
    def save_fig(self, path: str, max_neurons_per_layer: int = 3):
        
        total_layers = len(self.layers)
        fig, axes = plt.subplots(total_layers, max_neurons_per_layer, 
                                figsize=(5*max_neurons_per_layer, 4*total_layers))
        
        if total_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx in range(min(max_neurons_per_layer, layer.num_neurons)):
                i_idx = neuron_idx // layer.O
                o_idx = neuron_idx % layer.O
                
                if i_idx < layer.I:
                    layer.plot_neuron(axes[layer_idx, neuron_idx], i_idx, o_idx)
                    axes[layer_idx, neuron_idx].set_title(f'Layer {layer_idx}, Neuron ({i_idx},{o_idx})')
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
    def to_device(self, device: str):
        """Move the entire network to a specific device."""
        if device == 'gpu':
            for layer in self.layers:
                layer._move_to_gpu()
            
            for normalizer in self.normalizers:
                if normalizer is not None and hasattr(normalizer, 'device_config'):
                    normalizer.device_config['use_gpu'] = True
        else:
            cpu_device = jax.devices('cpu')[0]
            for layer in self.layers:
                layer.z = jax.device_put(layer.z, cpu_device)
                layer.h = jax.device_put(layer.h, cpu_device)
                layer.l = jax.device_put(layer.l, cpu_device)
                layer.s = jax.device_put(layer.s, cpu_device)
                layer.jitter = jax.device_put(layer.jitter, cpu_device)
    
    def __repr__(self) -> str:
        layer_info = [f"{layer.I}→{layer.O}" for layer in self.layers]
        device_info = f"GPU={self.device_config['use_gpu']}"
        return f"GP_KAN({' → '.join(layer_info)}, {device_info})"
