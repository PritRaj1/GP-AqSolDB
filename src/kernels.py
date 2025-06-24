import numpy as np

RBF = lambda x, y, sigma: np.exp(-(x-y)**2 / (2*sigma**2))
RQ = lambda x, y, sigma, alpha: (1 + (x-y)**2 / (2*alpha*sigma**2))**(-alpha)

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
