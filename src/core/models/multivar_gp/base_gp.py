from typing import Any

import numpy as np
from scipy import linalg

from ...kernels import compute_kernel


class BaseGP:
    """Shared base for DenseGP and FITCGP."""

    def __init__(self, config: Any, sigma: np.ndarray) -> None:
        self.config = config
        self.sigma = np.asarray(sigma)
        self.kernel_type = config.get("KERNEL", "type")
        self.alpha = config.getfloat("KERNEL", "alpha")
        self.noise_var = config.getfloat("KERNEL", "lmbda")

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return compute_kernel(self.kernel_type, X1, X2, self.sigma, self.alpha)

    def _recast_2D(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _cholesky_with_jitter(self, K: np.ndarray) -> np.ndarray:
        """Cholesky decomposition with escalating jitter for numerical stability."""
        n = K.shape[0]
        for jitter in [0, 1e-8, 1e-6, 1e-4]:
            try:
                if jitter > 0:
                    K = K + jitter * np.eye(n)
                return linalg.cholesky(K, lower=True)
            except linalg.LinAlgError:
                continue
        raise linalg.LinAlgError("Cholesky failed even with jitter=1e-4")

    def get_model_complexity(self) -> int:
        n_params = len(self.sigma) + 1  # sigmas + lambda
        if self.kernel_type == "RQ":
            n_params += 1
        return n_params
