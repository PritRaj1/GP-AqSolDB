from typing import Any

import jax
import jax.numpy as jnp

from ...kernels import compute_kernel


class BaseGP:
    """Shared base for DenseGP and FITCGP."""

    def __init__(self, config: Any, sigma: jnp.ndarray) -> None:
        self.config = config
        self.sigma = jnp.asarray(sigma)
        self.kernel_type = config.get("KERNEL", "type")
        self.alpha = config.getfloat("KERNEL", "alpha")
        self.noise_var = config.getfloat("KERNEL", "lmbda")

    def _kernel(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        return compute_kernel(self.kernel_type, X1, X2, self.sigma, self.alpha)

    def _recast_2D(self, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    def _cholesky_with_jitter(self, K: jnp.ndarray) -> jnp.ndarray:
        """Cholesky decomposition with escalating jitter for numerical stability."""
        n = K.shape[0]
        orig_dtype = K.dtype
        with jax.enable_x64(True):
            K_f64 = jnp.asarray(K, dtype=jnp.float64)
            for jitter in [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
                if jitter > 0:
                    eye = jnp.eye(n, dtype=jnp.float64)
                    K_j = K_f64 + jitter * eye

                else:
                    K_j = K_f64

                L = jax.scipy.linalg.cholesky(K_j, lower=True)
                if not jnp.any(jnp.isnan(L)):
                    return L.astype(orig_dtype)

        raise RuntimeError("Cholesky failed even with jitter=1e-2")

    def get_model_complexity(self) -> int:
        n_params = len(self.sigma) + 1  # sigmas + lambda
        if self.kernel_type == "RQ":
            n_params += 1

        return n_params
