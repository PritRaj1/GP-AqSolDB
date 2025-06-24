import numpy as np

def RBF(x, y, sigma):
    """Radial Basis Function kernel"""
    return np.exp(-0.5 * np.sum(((x - y) / sigma)**2, axis=-1))

def RQ(x, y, sigma, alpha):
    """Rational Quadratic kernel"""
    return (1 + 0.5 * np.sum(((x - y) / sigma)**2, axis=-1) / alpha)**(-alpha)

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
