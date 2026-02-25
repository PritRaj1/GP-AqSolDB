from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .base_gp import BaseGP

# ---------------------------------------------------------------------------
# Jitted predict helpers
# ---------------------------------------------------------------------------


@jax.jit
def _dense_predict_mean(
    K_star: jnp.ndarray, L: jnp.ndarray, alpha_vec: jnp.ndarray
) -> jnp.ndarray:
    return K_star @ jax.scipy.linalg.solve_triangular(L.T, alpha_vec, lower=False)


@jax.jit
def _dense_predict_std(
    K_star: jnp.ndarray,
    K_ss: jnp.ndarray,
    L: jnp.ndarray,
    noise_var: float,
) -> jnp.ndarray:
    v = jax.scipy.linalg.solve_triangular(L, K_star.T, lower=True)
    var_pred = jnp.diag(K_ss) - jnp.sum(v**2, axis=0)
    return jnp.sqrt(jnp.maximum(var_pred, 0)) + noise_var


# ---------------------------------------------------------------------------
# DenseGP
# ---------------------------------------------------------------------------


class DenseGP(BaseGP):
    def __init__(self, config: Any, sigma: jnp.ndarray) -> None:
        super().__init__(config, sigma)
        self.L: Optional[jnp.ndarray] = None
        self.alpha_vec: Optional[jnp.ndarray] = None
        self.X_train: Optional[jnp.ndarray] = None
        self.y_train: Optional[jnp.ndarray] = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> "DenseGP":
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
            )

        self.X_train = self._recast_2D(X)
        self.y_train = jnp.asarray(y)

        K = self._kernel(self.X_train, self.X_train)
        K = K + self.noise_var * jnp.eye(len(self.X_train))

        self.L = self._cholesky_with_jitter(K)

        self.alpha_vec = jax.scipy.linalg.solve_triangular(
            self.L, self.y_train, lower=True
        )
        return self

    def predict(
        self, X_test: jnp.ndarray, return_std: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        if self.L is None or self.alpha_vec is None or self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")

        X_test = self._recast_2D(X_test)
        K_star = self._kernel(X_test, self.X_train)
        mean_pred = _dense_predict_mean(K_star, self.L, self.alpha_vec)

        if return_std:
            K_star_star = self._kernel(X_test, X_test)
            std_pred = _dense_predict_std(K_star, K_star_star, self.L, self.noise_var)
            return mean_pred, std_pred

        return mean_pred
