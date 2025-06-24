import numpy as np
from src.kernels import get_kernel

class GP:
    def __init__(self, config, sigma):
        self.config = config
        self.kernel = get_kernel(config, sigma)

    def construct_kernel_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])
        return K
    
    def predict(self, X_test):
        K_star = self.construct_kernel_matrix(X_test)
        K_star_star = self.construct_kernel_matrix(X_test)
        K_star_star_inv = np.linalg.inv(K_star_star)
        K_star_inv_K = np.linalg.inv(K_star)
        K_star_inv_K_star = np.linalg.inv(K_star_star)
        return K_star_inv_K_star