import jax
import jax.numpy as jnp
from typing import Optional
from dataclasses import dataclass

@dataclass
class NormalDist:
    mean: jax.Array
    var: jax.Array
    
    def __post_init__(self):
        if self.mean.shape != self.var.shape:
            raise ValueError(f"Mean shape {self.mean.shape} must match variance shape {self.var.shape}")
    
    @classmethod
    def from_array(cls, mean: jax.Array, var: Optional[jax.Array] = None):
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
    
    def __add__(self, other: 'NormalDist') -> 'NormalDist':
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
    
    def sample(self, key: jax.random.PRNGKey, shape: Optional[tuple] = None) -> jax.Array:
        """
        Sample from NormalDist.
        
        Parameters:
        -----------
        key : jax.random.PRNGKey
            Random key for sampling
        shape : tuple, optional
            Shape of samples to generate. If None, uses the shape of mean.
            
        Returns:
        --------
        jax.Array : Sampled values
        """
        if shape is None:
            shape = self.mean.shape
        
        std_samples = jax.random.normal(key, shape)
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
        return -0.5 * (jnp.log(2 * jnp.pi * self.var) + 
                       (x - self.mean)**2 / self.var)
    
    @property
    def std(self) -> jax.Array:
        return jnp.sqrt(self.var)
    
    def to_numpy(self):
        return NormalDist(jnp.array(self.mean), jnp.array(self.var))
    