from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import linalg, stats

from ..kernels import clear_kernel_cache, get_cache_stats, get_kernel


class DenseGP:
    def __init__(self, config: Any, sigma: np.ndarray) -> None:
        self.config = config
        self.sigma = np.asarray(sigma)
        self.use_cache = config.getboolean("KERNEL", "use_cache", fallback=True)
        cache_size = config.getint("KERNEL", "cache_size", fallback=100)

        self.kernel = get_kernel(
            config, sigma, use_cache=self.use_cache, cache_size=cache_size
        )
        self.L: Optional[np.ndarray] = None  # Cholesky factor
        self.alpha: Optional[np.ndarray] = None  # Solution vector
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.noise_var = config.getfloat("KERNEL", "lmbda")

    def _recast_2D(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is 2D array for vectorized kernels"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DenseGP":
        """
        Fit to data.

        Cholesky decomposition is used instead of np.linalg.inv
        since kernel matrix is symmetric positive definite.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Training targets

        Returns
        -------
        self : object
            Returns self.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}"
            )

        self.X_train = self._recast_2D(X)
        self.y_train = y

        K = self.kernel(self.X_train, self.X_train)
        K += self.noise_var * np.eye(len(self.X_train))

        # Sometimes jitter is needed for stability
        try:
            self.L = linalg.cholesky(K, lower=True)
        except linalg.LinAlgError:
            jitter = 1e-8
            K += jitter * np.eye(len(self.X_train))
            self.L = linalg.cholesky(K, lower=True)

        # Forward substitution to solve L @ alpha = y
        self.alpha = linalg.solve_triangular(self.L, self.y_train, lower=True)
        return self

    def predict(
        self, X_test: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using fitted model

        Parameters
        ----------
        X_test : array-like, shape (n_test_samples, n_features)
            Test points
        return_std : bool, optional
            If True, return standard deviation along with mean

        Returns
        -------
        mean_pred : array-like, shape (n_test_samples,)
            Predicted mean
        std_pred : array-like, shape (n_test_samples,), optional
            Predicted standard deviation (if return_std=True)
        """
        if self.L is None or self.alpha is None:
            raise ValueError("Model must be fitted before making predictions")

        X_test = self._recast_2D(X_test)
        K_star = self.kernel(X_test, self.X_train)

        # Predicted mean: K_star @ K^(-1) @ y = K_star @ solve(L.T, alpha)
        mean_pred = K_star @ linalg.solve_triangular(self.L.T, self.alpha, lower=False)

        # Predictive variance: k(x*,x*) - k(x*,X) @ K^(-1) @ k(X,x*)
        if return_std:
            K_star_star = self.kernel(X_test, X_test)
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

    def get_cache_stats(self) -> Optional[Dict[str, Union[int, float]]]:
        if self.use_cache:
            return get_cache_stats()
        else:
            return None

    def get_model_complexity(self) -> int:
        """Get the number of parameters in the model"""
        n_params = len(self.sigma) + 1  # sigmas + lambda
        if hasattr(self, "kernel") and hasattr(self.kernel, "alpha"):
            n_params += 1  # alpha parameter for RQ kernel
        return n_params

    def clear_cache(self) -> None:
        if self.use_cache:
            clear_kernel_cache()
