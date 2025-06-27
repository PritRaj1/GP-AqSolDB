import numpy as np
from src.kernels import get_kernel, get_cache_stats, clear_kernel_cache
from scipy import stats
from scipy import linalg

class GP:
    def __init__(self, config, sigma):
        self.config = config
        self.use_cache = config.getboolean("KERNEL", "use_cache", fallback=True)
        cache_size = config.getint("KERNEL", "cache_size", fallback=100)
        
        self.kernel = get_kernel(config, sigma, use_cache=self.use_cache, cache_size=cache_size)
        self.L = None
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.noise_var = config.getfloat("KERNEL", "lmbda")

    def _recast_2D(self, X):
        """Ensure X is 2D array for vectorized kernels"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X, y):
        """
        Fit the GP model to the training data

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
    
    def predict(self, X_test, return_std=False):
        X_test = self._recast_2D(X_test)
        K_star = self.kernel(X_test, self.X_train)
        
        # Predicted mean: K_star @ K^(-1) @ y = K_star @ K^(-1) @ y
        mean_pred = K_star @ linalg.solve_triangular(self.L.T, self.alpha, lower=False)
        
        # Predictive variance: k(x*,x*) - k(x*,X) @ K^(-1) @ k(X,x*)
        if return_std:
            K_star_star = self.kernel(X_test, X_test)
            v = linalg.solve_triangular(self.L, K_star.T, lower=True)
            var_pred = np.diag(K_star_star) - np.sum(v**2, axis=0)
            std_pred = np.sqrt(np.maximum(var_pred, 0)) + self.noise_var
            return mean_pred, std_pred
        
        return mean_pred
    
    def eval_fit(self, y_pred, y_true):
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        return r_value, p_value, std_err
    
    def get_cache_stats(self):
        if self.use_cache:
            return get_cache_stats()
        else:
            return None
    
    def clear_cache(self):
        if self.use_cache:
            clear_kernel_cache()