import numpy as np
from src.kernels import get_kernel
from scipy import stats

class GP:
    def __init__(self, config, sigma):
        self.config = config
        self.kernel = get_kernel(config, sigma)
        self.C = None
        self.K_inv = None
        self.X_train = None
        self.y_train = None
        self.noise_var = config.getfloat("KERNEL", "lmbda")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X)
        K += self.noise_var * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        self.C = self.K_inv @ y
    
    def predict(self, X_test, return_std=False):
        K_star = self.kernel(X_test, self.X_train)        
        mean_pred = K_star @ self.C
        
        # Predictive variance: k(x*,x*) - k(x*,X) @ K^(-1) @ k(X,x*)
        if return_std:
            K_star_star = self.kernel(X_test, X_test)
            var_pred = np.diag(K_star_star) - np.sum((K_star @ self.K_inv) * K_star, axis=1)
            std_pred = np.sqrt(np.maximum(var_pred, 0))
            return mean_pred, std_pred
        
        return mean_pred
    
    def eval_fit(self, y_pred, y_true):
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        return r_value, p_value, std_err