import jax
import jax.numpy as jnp
from typing import List

from normal_dist import NormalDist

class NormaliseGaussian:
    """Normalize - tanh for mean and sigmoid for variance."""
    
    def __init__(self, min_var=0.2):
        self.min_var = min_var

    @staticmethod
    def inverse_sigmoid(x: float):
        t1 = (1 / x) - 1
        t2 = -jnp.log(t1)
        return t2

    def __call__(self, x: NormalDist) -> NormalDist:
        out_mean = jnp.tanh(x.mean)
        sigmoid_offset = self.inverse_sigmoid(self.min_var)
        out_var = jax.nn.sigmoid(x.var - x.mean**2 + sigmoid_offset)
        return NormalDist(out_mean, out_var)


class ReshapeGaussian:
    
    def __init__(self, new_shape: List[int]):
        self.new_shape = new_shape

    def __call__(self, x: NormalDist) -> NormalDist:
        out_mean = jnp.reshape(x.mean, self.new_shape)
        out_var = jnp.reshape(x.var, self.new_shape)
        return NormalDist(out_mean, out_var)
