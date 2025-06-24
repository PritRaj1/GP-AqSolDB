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

    def construct_kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.construct_kernel_matrix(X)
        K += self.noise_var * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        self.C = self.K_inv @ y
    
    def predict(self, X_test, return_std=False):
        n_test = len(X_test)
        n_train = len(self.X_train)
        K_star = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            for j in range(n_train):
                K_star[i, j] = self.kernel(X_test[i], self.X_train[j])
        
        # Predicted mean
        mean_pred = K_star @ self.C
        
        # Epistemic uncertainty quantification on test points
        if return_std:
            K_star_star = np.zeros((n_test, n_test))
            for i in range(n_test):
                for j in range(n_test):
                    K_star_star[i, j] = self.kernel(X_test[i], X_test[j])
            
            var_pred = np.diag(K_star_star) - np.sum((K_star @ self.K_inv) * K_star, axis=1)
            std_pred = np.sqrt(np.maximum(var_pred, 0))  
            return mean_pred, std_pred
        
        return mean_pred
    
    def eval_fit(self, y_pred, y_true):
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        return r_value, p_value, std_err