import numpy as np

def RBF(x, y, sigma):
    """Radial Basis Function kernel with support for multivariate sigma"""
    if np.isscalar(sigma):
        return np.exp(-(x-y)**2 / (2*sigma**2))
    else:
        diff = x - y
        return np.exp(-0.5 * np.sum((diff / sigma)**2))

def RQ(x, y, sigma, alpha):
    """Rational Quadratic kernel with support for multivariate sigma"""
    if np.isscalar(sigma):
        return (1 + (x-y)**2 / (2*alpha*sigma**2))**(-alpha)
    else:
        diff = x - y
        return (1 + 0.5 * np.sum((diff / sigma)**2) / alpha)**(-alpha)

def get_kernel(config, sigma):
    kernel_type = config.get("KERNEL", "type")
    alpha = config.getfloat("KERNEL", "alpha")

    kernel_functions = {
        "RBF": lambda x, y: RBF(x, y, sigma),
        "RQ": lambda x, y: RQ(x, y, sigma, alpha)
    }
    
    if kernel_type not in kernel_functions:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernel_functions[kernel_type]
