import numpy as np
from src.kernels import get_kernel
from scipy import stats

class GP:
    def __init__(self, config, sigma):
        self.config = config
        self.kernel = get_kernel(config, sigma)
        self.C = None

    def construct_kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        K = self.construct_kernel_matrix(X)
        self.C = np.linalg.lstsq(K, y, rcond=None)[0]
    
    def predict(self, X_test):
        K_star = self.construct_kernel_matrix(X_test)
        return K_star @ self.C
    
    def eval_fit(self, y_pred, y_true):
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)