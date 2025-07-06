import jax
import jax.numpy as jnp
from typing import List, Dict, Any
from dataclasses import dataclass

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normal_dist import NormalDist
from dense_layer import DenseGPLayer, GPConfig
from invariant_acts import NormaliseGaussian, ReshapeGaussian

@dataclass
class GP_KANConfig:
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    
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
    
    min_var: float = 0.2
    
    seed: int = 42


class GP_KAN:
    
    def __init__(self, config: GP_KANConfig):
        self.config = config
        self.layers = []
        self.normalizers = []
        
        self._build_network()
    
    def _build_network(self):
        layer_sizes = [self.config.input_size] + self.config.hidden_sizes + [self.config.output_size]
        
        gp_config = GPConfig(
            num_inducing_points=self.config.num_inducing_points,
            z_init_low=self.config.z_init_low,
            z_init_high=self.config.z_init_high,
            h_init_low=self.config.h_init_low,
            h_init_high=self.config.h_init_high,
            global_length_scale=self.config.global_length_scale,
            min_length_scale=self.config.min_length_scale,
            global_covariance_scale=self.config.global_covariance_scale,
            min_covariance_scale=self.config.min_covariance_scale,
            global_jitter=self.config.global_jitter,
            baseline_jitter=self.config.baseline_jitter,
        )
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            layer = DenseGPLayer(
                input_size=input_size,
                output_size=output_size,
                config=gp_config,
                key=jax.random.PRNGKey(self.config.seed + i)
            )
            self.layers.append(layer)
            
            if i < len(layer_sizes) - 2:
                normalizer = NormaliseGaussian(min_var=self.config.min_var)
                self.normalizers.append(normalizer)
            else:
                self.normalizers.append(None)
    
    def forward(self, x: NormalDist) -> NormalDist:
        current = x
        
        for i, (layer, normalizer) in enumerate(zip(self.layers, self.normalizers)):
            current = layer.forward(current)
            
            if normalizer is not None:
                current = normalizer(current)
        
        return current
    
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
    
    def save_fig(self, path: str, max_neurons_per_layer: int = 3):
        import matplotlib.pyplot as plt
        
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
    
    def __repr__(self) -> str:
        layer_info = [f"{layer.I}→{layer.O}" for layer in self.layers]
        return f"GP_KAN({' → '.join(layer_info)})"


# Breakpoint testing - temporary
if __name__ == "__main__":
    config = GP_KANConfig(
        input_size=2,
        hidden_sizes=[4, 3],
        output_size=1,
        num_inducing_points=5,
        seed=42 # TODO: remove
    )
    
    network = GP_KAN(config)
    print(network)
    
    input_mean = jnp.array([[0.0, 1.0]])  # Shape: (1, 2)
    input_var = jnp.array([[0.1, 0.1]])   # Shape: (1, 2)
    input_dist = NormalDist(input_mean, input_var)
    
    output_dist = network.forward(input_dist)
    print(f"Input: {input_dist}")
    print(f"Output: {output_dist}")
    print(f"Log-likelihood: {network.loglikelihood()}")
    
    network.save_fig("gp_kan_visualization.png")
