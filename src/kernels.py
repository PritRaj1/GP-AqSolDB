import numpy as np

RBF = lambda x, y, l, _alpha: np.exp(-(x-y)**2 / (2*l**2))
RQ = lambda x, y, l, alpha: (1 + (x-y)**2 / (2*alpha*l**2))**(-alpha)

def get_kernel(config):
    kernel_type = config.get("KERNEL", "type")
    lmbda = config.getfloat("KERNEL", "lmbda")
    alpha = config.getfloat("KERNEL", "alpha")

    kernel_functions = {
        "RBF": RBF,
        "RQ": RQ
    }
    
    if kernel_type not in kernel_functions:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return lambda x, y: kernel_functions[kernel_type](x, y, lmbda, alpha)
