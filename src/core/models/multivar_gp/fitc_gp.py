from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from ....utils import get_inducing_selector
from ...kernels import compute_kernel


class FITCGP:
    """
    Sparse GP using FITC (Fully Independent Training Conditional) approximation.

    Reduces computational complexity from O(N^3) to O(NM^2) where M << N.
    """

    def __init__(self, config: Any, sigma: np.ndarray) -> None:
        self.config = config
        self.sigma = np.asarray(sigma)
        self.kernel_type = config.get("KERNEL", "type")
        self.alpha = config.getfloat("KERNEL", "alpha")
        self.num_inducing = config.getint("KERNEL", "num_inducing", fallback=20)
        self.noise_var = config.getfloat("KERNEL", "lmbda")

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.LA: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.Lambda: Optional[np.ndarray] = None
        self.Kmm: Optional[np.ndarray] = None
        self.Kmm_inv: Optional[np.ndarray] = None
        self.Knm: Optional[np.ndarray] = None
        self.is_fitted = False

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return compute_kernel(self.kernel_type, X1, X2, self.sigma, self.alpha)

    def _recast_2D(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _select_inducing_points(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, method: str = "kmeans"
    ) -> np.ndarray:
        N = X.shape[0]
        if self.num_inducing >= N:
            return X
        selector = get_inducing_selector(method, self.num_inducing, random_state=42)
        return np.asarray(selector.select(X, y))

    def fit(
        self, X: np.ndarray, y: np.ndarray, inducing_method: str = "kmeans"
    ) -> "FITCGP":
        self.X_train = self._recast_2D(X)
        self.y_train = y
        self._inducing_method = inducing_method

        if inducing_method in ["stratified", "adaptive"]:
            self.Z = self._select_inducing_points(
                self.X_train, y, method=inducing_method
            )
        else:
            self.Z = self._select_inducing_points(self.X_train, method=inducing_method)
        M = self.Z.shape[0]

        self.Kmm = self._kernel(self.Z, self.Z) + 1e-6 * np.eye(M)
        self.Knm = self._kernel(self.X_train, self.Z)
        Kmn = self.Knm.T

        diag_Knn = np.diag(self._kernel(self.X_train, self.X_train))
        self.Kmm_inv = np.linalg.inv(self.Kmm)

        Qnn_diag = np.einsum("ij,jk,ki->i", self.Knm, self.Kmm_inv, Kmn)
        self.Lambda = diag_Knn - Qnn_diag + self.noise_var

        A = self.Kmm + Kmn @ (self.Knm / self.Lambda[:, None])

        for jitter in [0, 1e-8, 1e-6, 1e-4]:
            try:
                if jitter > 0:
                    A += jitter * np.eye(M)
                self.LA = linalg.cholesky(A, lower=True)
                break
            except linalg.LinAlgError:
                continue
        else:
            raise linalg.LinAlgError("Cholesky failed even with jitter=1e-4")

        b = Kmn @ (self.y_train / self.Lambda)
        self.v = linalg.solve_triangular(self.LA, b, lower=True)

        self.is_fitted = True
        return self

    def predict(
        self, X_test: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_test = self._recast_2D(X_test)

        if self.Z is None or self.LA is None or self.v is None or self.Kmm_inv is None:
            raise ValueError("Model not properly fitted")

        Kms = self._kernel(self.Z, X_test)
        Ksm = Kms.T
        Kss_diag = np.diag(self._kernel(X_test, X_test))

        tmp = linalg.solve_triangular(self.LA, Kms, lower=True)
        mean_pred = Ksm @ linalg.solve_triangular(self.LA.T, self.v, lower=False)

        if return_std:
            Qss_diag = np.einsum("ij,jk,ki->i", Ksm, self.Kmm_inv, Kms)
            var_pred = Kss_diag - Qss_diag + np.sum(tmp**2, axis=0) + self.noise_var
            std_pred = np.sqrt(np.maximum(var_pred, 0))
            return np.asarray(mean_pred), np.asarray(std_pred)

        return np.asarray(mean_pred)

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
        n_params: int = len(self.sigma) + 1  # sigmas + lambda
        if self.kernel_type == "RQ":
            n_params += 1
        n_params += self.num_inducing * len(self.sigma)  # inducing points
        return int(n_params)
