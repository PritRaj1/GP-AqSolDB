from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from ....utils import get_inducing_selector
from .base_gp import BaseGP

# ---------------------------------------------------------------------------
# Jitted predict helpers
# ---------------------------------------------------------------------------


@jax.jit
def _fitc_predict_mean(
    Ksm: jnp.ndarray, LA: jnp.ndarray, v: jnp.ndarray
) -> jnp.ndarray:
    return Ksm @ jax.scipy.linalg.solve_triangular(LA.T, v, lower=False)


@jax.jit
def _fitc_predict_std(
    Kss_diag: jnp.ndarray,
    Ksm: jnp.ndarray,
    Kmm_inv: jnp.ndarray,
    Kms: jnp.ndarray,
    LA: jnp.ndarray,
    noise_var: float,
) -> jnp.ndarray:
    tmp = jax.scipy.linalg.solve_triangular(LA, Kms, lower=True)
    Qss_diag = jnp.einsum("ij,jk,ki->i", Ksm, Kmm_inv, Kms)
    var_pred = Kss_diag - Qss_diag + jnp.sum(tmp**2, axis=0) + noise_var
    return jnp.sqrt(jnp.maximum(var_pred, 0))


# ---------------------------------------------------------------------------
# FITCGP
# ---------------------------------------------------------------------------


class FITCGP(BaseGP):
    """
    Sparse GP using FITC (Fully Independent Training Conditional) approximation.

    Reduces computational complexity from O(N^3) to O(NM^2) where M << N.
    """

    def __init__(self, config: Any, sigma: jnp.ndarray) -> None:
        super().__init__(config, sigma)
        self.num_inducing = config.getint(
            "SPARSE",
            "num_inducing",
            fallback=config.getint("KERNEL", "num_inducing", fallback=20),
        )

        self.X_train: Optional[jnp.ndarray] = None
        self.y_train: Optional[jnp.ndarray] = None
        self.Z: Optional[jnp.ndarray] = None
        self.LA: Optional[jnp.ndarray] = None
        self.v: Optional[jnp.ndarray] = None
        self.Lambda: Optional[jnp.ndarray] = None
        self.Kmm: Optional[jnp.ndarray] = None
        self.Kmm_inv: Optional[jnp.ndarray] = None
        self.Knm: Optional[jnp.ndarray] = None
        self.is_fitted = False

    def _select_inducing_points(
        self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None, method: str = "kmeans"
    ) -> jnp.ndarray:
        N = X.shape[0]
        if self.num_inducing >= N:
            return X
        selector = get_inducing_selector(method, self.num_inducing, random_state=42)
        # sklearn selectors expect/return numpy arrays
        y_np = np.asarray(y) if y is not None else None
        result = selector.select(np.asarray(X), y_np)
        return jnp.asarray(result)

    def fit(
        self, X: jnp.ndarray, y: jnp.ndarray, inducing_method: str = "kmeans"
    ) -> "FITCGP":
        self.X_train = self._recast_2D(X)
        self.y_train = jnp.asarray(y)
        self._inducing_method = inducing_method

        if inducing_method in ["stratified", "adaptive"]:
            self.Z = self._select_inducing_points(
                self.X_train, y, method=inducing_method
            )

        else:
            self.Z = self._select_inducing_points(self.X_train, method=inducing_method)
        M = self.Z.shape[0]

        self.Kmm = self._kernel(self.Z, self.Z) + 1e-6 * jnp.eye(M)
        self.Knm = self._kernel(self.X_train, self.Z)
        Kmn = self.Knm.T

        diag_Knn = jnp.diag(self._kernel(self.X_train, self.X_train))
        self.Kmm_inv = jnp.linalg.inv(self.Kmm)

        Qnn_diag = jnp.einsum("ij,jk,ki->i", self.Knm, self.Kmm_inv, Kmn)
        # Clamp to noise_var floor — float32 can make diag_Knn - Qnn_diag negative
        self.Lambda = jnp.maximum(diag_Knn - Qnn_diag + self.noise_var, self.noise_var)

        A = self.Kmm + Kmn @ (self.Knm / self.Lambda[:, None])

        self.LA = self._cholesky_with_jitter(A)

        b = Kmn @ (self.y_train / self.Lambda)
        self.v = jax.scipy.linalg.solve_triangular(self.LA, b, lower=True)

        self.is_fitted = True
        return self

    def predict(
        self, X_test: jnp.ndarray, return_std: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_test = self._recast_2D(X_test)

        if self.Z is None or self.LA is None or self.v is None or self.Kmm_inv is None:
            raise ValueError("Model not properly fitted")

        Kms = self._kernel(self.Z, X_test)
        Ksm = Kms.T
        Kss_diag = jnp.diag(self._kernel(X_test, X_test))

        mean_pred = _fitc_predict_mean(Ksm, self.LA, self.v)

        if return_std:
            std_pred = _fitc_predict_std(
                Kss_diag, Ksm, self.Kmm_inv, Kms, self.LA, self.noise_var
            )
            return mean_pred, std_pred

        return mean_pred

    def get_sparse_info(self) -> Optional[Dict[str, Any]]:
        if self.is_fitted and self.Z is not None and self.X_train is not None:
            return {
                "num_inducing": self.Z.shape[0],
                "num_training": self.X_train.shape[0],
                "compression_ratio": self.Z.shape[0] / self.X_train.shape[0],
                "inducing_points": self.Z,
                "inducing_method": getattr(self, "_inducing_method", "unknown"),
            }
        return None

    def get_model_complexity(self) -> int:
        n_params: int = super().get_model_complexity()
        n_params += self.num_inducing * len(self.sigma)  # inducing points
        return int(n_params)
