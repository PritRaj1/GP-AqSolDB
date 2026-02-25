from typing import Any, Optional, Tuple, Union

import numpy as np
from scipy import linalg, stats

from ...kernels import compute_kernel


class DenseGP:
    def __init__(self, config: Any, sigma: np.ndarray) -> None:
        self.config = config
        self.sigma = np.asarray(sigma)
        self.kernel_type = config.get("KERNEL", "type")
        self.alpha = config.getfloat("KERNEL", "alpha")
        self.L: Optional[np.ndarray] = None
        self.alpha_vec: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.noise_var = config.getfloat("KERNEL", "lmbda")

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return compute_kernel(self.kernel_type, X1, X2, self.sigma, self.alpha)

    def _recast_2D(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DenseGP":
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
            )

        self.X_train = self._recast_2D(X)
        self.y_train = y

        K = self._kernel(self.X_train, self.X_train)
        K += self.noise_var * np.eye(len(self.X_train))

        # Escalating jitter for numerical stability
        for jitter in [0, 1e-8, 1e-6, 1e-4]:
            try:
                if jitter > 0:
                    K += jitter * np.eye(len(self.X_train))
                self.L = linalg.cholesky(K, lower=True)
                break

            except linalg.LinAlgError:
                continue
        else:
            raise linalg.LinAlgError("Cholesky failed even with jitter=1e-4")

        self.alpha_vec = linalg.solve_triangular(self.L, self.y_train, lower=True)
        return self

    def predict(
        self, X_test: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.L is None or self.alpha_vec is None or self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")

        X_test = self._recast_2D(X_test)
        K_star = self._kernel(X_test, self.X_train)
        mean_pred = K_star @ linalg.solve_triangular(
            self.L.T, self.alpha_vec, lower=False
        )

        if return_std:
            K_star_star = self._kernel(X_test, X_test)
            v = linalg.solve_triangular(self.L, K_star.T, lower=True)
            var_pred = np.diag(K_star_star) - np.sum(v**2, axis=0)
            std_pred = np.sqrt(np.maximum(var_pred, 0)) + self.noise_var
            return np.asarray(mean_pred), np.asarray(std_pred)

        return np.asarray(mean_pred)

    def eval_fit(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float, float]:
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        return r_value, p_value, std_err

    def get_model_complexity(self) -> int:
        n_params = len(self.sigma) + 1  # sigmas + lambda
        if self.kernel_type == "RQ":
            n_params += 1
        return n_params
