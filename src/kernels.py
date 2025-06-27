import numpy as np

def RBF(X1, X2, sigma):
    """
    Radial Basis Function kernel
    
    Parameters:
    -----------
    X1 : np.ndarray, shape (n1, d)
        First set of points
    X2 : np.ndarray, shape (n2, d) 
        Second set of points
    sigma : np.ndarray, shape (d,)
        Length scales for each dimension
        
    Returns:
    --------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix
    """
    X1_norm = X1 / sigma
    X2_norm = X2 / sigma
    
    # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T
    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    
    return np.exp(-0.5 * sq_dist)

def RQ(X1, X2, sigma, alpha):
    """
    Rational Quadratic kernel
    
    Parameters:
    -----------
    X1 : np.ndarray, shape (n1, d)
        First set of points
    X2 : np.ndarray, shape (n2, d)
        Second set of points
    sigma : np.ndarray, shape (d,)
        Length scales for each dimension
    alpha : float
        Shape parameter
        
    Returns:
    --------
    K : np.ndarray, shape (n1, n2)
        Kernel matrix
    """
    X1_norm = X1 / sigma
    X2_norm = X2 / sigma
    
    # ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    X1_sq = np.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T
    sq_dist = X1_sq + X2_sq - 2 * inner_prod
    
    return (1 + 0.5 * sq_dist / alpha)**(-alpha)

def get_kernel(config, sigma):
    """
    Get kernel function based on config
    
    Parameters:
    -----------
    config : ConfigParser
        Configuration object
    sigma : np.ndarray
        Length scales for each dimension
        
    Returns:
    --------
    kernel_func : function
        Vectorized kernel function that takes (X1, X2) and returns kernel matrix
    """
    kernel_type = config.get("KERNEL", "type")
    alpha = config.getfloat("KERNEL", "alpha")

    kernel_functions = {    
        "RBF": lambda X1, X2: RBF(X1, X2, sigma),
        "RQ": lambda X1, X2: RQ(X1, X2, sigma, alpha)
    }
    
    if kernel_type not in kernel_functions:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernel_functions[kernel_type]
