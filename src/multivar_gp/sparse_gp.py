from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from src.multivar_gp.kernels import clear_kernel_cache, get_cache_stats, get_kernel


class SparseGP:
    """
    Sparse GP using FITC (Fully Independent Training Conditional) approximation.

    This scales better to large datasets by using a subset of inducing points
    instead of the full training dataset, reducing computational complexity from O(N³)
    to O(NM²) where M << N.
    """

    def __init__(self, config: Any, sigma: np.ndarray) -> None:
        self.config = config
        self.use_cache = config.getboolean("KERNEL", "use_cache", fallback=True)
        cache_size = config.getint("KERNEL", "cache_size", fallback=100)
        self.num_inducing = config.getint("KERNEL", "num_inducing", fallback=20)

        self.kernel = get_kernel(
            config, sigma, use_cache=self.use_cache, cache_size=cache_size
        )
        self.noise_var = config.getfloat("KERNEL", "lmbda")

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

        # Sparse GP variables
        self.Z: Optional[np.ndarray] = None  # Inducing points
        self.LA: Optional[np.ndarray] = None  # Cholesky factor for sparse GP
        self.v: Optional[np.ndarray] = None  # Solution vector for sparse GP
        self.Lambda: Optional[np.ndarray] = None  # Diagonal correction term
        self.Kmm: Optional[np.ndarray] = None  # Kernel matrix between inducing points
        self.Kmm_inv: Optional[np.ndarray] = None  # Inverse of Kmm
        self.Knm: Optional[np.ndarray] = (
            None  # Kernel matrix between training and inducing points
        )
        self.Kmn: Optional[np.ndarray] = None  # Transpose of Knm

        self.is_fitted = False

    def _recast_2D(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is 2D array for vectorized kernels"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _select_inducing_points(
        self, X: np.ndarray, method: str = "random"
    ) -> np.ndarray:
        """
        Select inducing points from training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        method : str, optional
            Selection method: 'random' or 'uniform'

        Returns
        -------
        Z : array-like, shape (num_inducing, n_features)
            Selected inducing points
        """
        N = X.shape[0]

        # If num_inducing >= N, use all points
        if self.num_inducing >= N:
            return X

        # Selection method
        if method == "random":
            idx = np.random.choice(N, self.num_inducing, replace=False)
            return X[idx]
        elif method == "uniform":
            step = N // self.num_inducing
            idx = np.arange(0, N, step)[: self.num_inducing]
            return X[idx]
        else:
            raise ValueError(f"Unknown inducing point selection method: {method}")

    def fit(
        self, X: np.ndarray, y: np.ndarray, inducing_method: str = "random"
    ) -> "SparseGP":
        """
        Fit using FITC approximation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Training targets
        inducing_method : str, optional
            Method for selecting inducing points: 'random' or 'uniform'

        Returns
        -------
        self : object
            Returns self.
        """
        self.X_train = self._recast_2D(X)
        self.y_train = y
        # N = self.X_train.shape[0]  # Unused variable

        self.Z = self._select_inducing_points(self.X_train, method=inducing_method)
        M = self.Z.shape[0]

        self.Kmm = self.kernel(self.Z, self.Z) + 1e-6 * np.eye(M)
        self.Knm = self.kernel(self.X_train, self.Z)
        self.Kmn = self.Knm.T

        diag_Knn = np.diag(self.kernel(self.X_train, self.X_train))

        # Qnn_diag = diag(Knm @ Kmm^{-1} @ Kmn)
        self.Kmm_inv = np.linalg.inv(self.Kmm)
        Qnn_diag = np.einsum("ij,jk,ki->i", self.Knm, self.Kmm_inv, self.Kmn)
        self.Lambda = diag_Knn - Qnn_diag + self.noise_var

        # Compute A = Kmm + Kmn @ diag(1/Lambda) @ Knm
        A = self.Kmm + self.Kmn @ (self.Knm / self.Lambda[:, None])

        try:
            self.LA = linalg.cholesky(A, lower=True)
        except linalg.LinAlgError:
            A += 1e-8 * np.eye(M)
            self.LA = linalg.cholesky(A, lower=True)

        # b = Kmn @ (y / Lambda)
        b = self.Kmn @ (self.y_train / self.Lambda)

        # Solve LA @ v = b
        self.v = linalg.solve_triangular(self.LA, b, lower=True)
        self.is_fitted = True
        return self

    def predict(
        self, X_test: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the fitted sparse GP model.

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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_test = self._recast_2D(X_test)

        Kms = self.kernel(self.Z, X_test)
        Ksm = Kms.T
        Kss_diag = np.diag(self.kernel(X_test, X_test))

        tmp = linalg.solve_triangular(self.LA, Kms, lower=True)
        mean_pred = Ksm @ linalg.solve_triangular(self.LA.T, self.v, lower=False)

        # Predictive variance: Kss - Qss + sum(tmp^2, axis=0) + noise_var
        if return_std:
            Qss_diag = np.einsum("ij,jk,ki->i", Ksm, self.Kmm_inv, Kms)
            var_pred = Kss_diag - Qss_diag + np.sum(tmp**2, axis=0) + self.noise_var
            std_pred = np.sqrt(np.maximum(var_pred, 0))

            return mean_pred, std_pred

        return mean_pred

    def get_sparse_info(self) -> Optional[Dict[str, Any]]:
        if self.is_fitted:
            return {
                "num_inducing": self.Z.shape[0],
                "num_training": self.X_train.shape[0],
                "compression_ratio": self.Z.shape[0] / self.X_train.shape[0],
                "inducing_points": self.Z,
            }
        else:
            return None

    def get_cache_stats(self) -> Optional[Dict[str, Union[int, float]]]:
        if self.use_cache:
            return get_cache_stats()
        else:
            return None

    def clear_cache(self) -> None:
        if self.use_cache:
            clear_kernel_cache()
