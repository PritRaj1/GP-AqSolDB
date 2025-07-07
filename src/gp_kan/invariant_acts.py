import jax
import jax.numpy as jnp
from typing import List, Optional
from configparser import ConfigParser

from src.gp_kan.normal_dist import NormalDist, get_device_config

def load_normalization_config(config: ConfigParser) -> dict:
    if 'NORMALIZATION' not in config:
        raise ValueError("NORMALIZATION section not found in config")
    
    normalization_section = config['NORMALIZATION']
    
    return {
        'min_var': float(normalization_section.get('min_var', '0.2'))
    }

def create_default_conf() -> ConfigParser:
    config = ConfigParser()
    config['NORMALIZATION'] = {
        'min_var': '0.2'
    }
    config['DEVICE'] = {
        'use_gpu': 'false',
        'device': 'cpu',
        'precision': 'float32'
    }
    return config

class NormaliseGaussian:
    """Normalize - tanh for mean and sigmoid for variance."""
    
    def __init__(self, min_var: float = 0.2, config: Optional[ConfigParser] = None):
        if config is not None:
            norm_params = load_normalization_config(config)
            self.min_var = norm_params['min_var']
            self.device_config = get_device_config(config)
        else:
            self.min_var = min_var
            self.device_config = {'use_gpu': False, 'device': 'cpu'}
        
        # Precompute sigmoid offset for efficiency
        self.sigmoid_offset = self.inverse_sigmoid(self.min_var)

    @staticmethod
    def inverse_sigmoid(x: float):
        t1 = (1 / x) - 1
        t2 = -jnp.log(t1)
        return t2

    def __call__(self, x: NormalDist) -> NormalDist:
        out_mean = jnp.tanh(x.mean)
        out_var = jax.nn.sigmoid(x.var - x.mean**2 + self.sigmoid_offset)
        
        if self.device_config['use_gpu']:
            return NormalDist(out_mean, out_var).to_device('gpu')

        return NormalDist(out_mean, out_var)


class ReshapeGaussian:
    """Reshape"""
    
    def __init__(self, new_shape: List[int], config: Optional[ConfigParser] = None):
        self.new_shape = new_shape
        if config is not None:
            self.device_config = get_device_config(config)
        else:
            self.device_config = {'use_gpu': False, 'device': 'cpu'}

    def __call__(self, x: NormalDist) -> NormalDist:
        out_mean = jnp.reshape(x.mean, self.new_shape)
        out_var = jnp.reshape(x.var, self.new_shape)
        
        if self.device_config['use_gpu']:
            return NormalDist(out_mean, out_var).to_device('gpu')

        return NormalDist(out_mean, out_var)



