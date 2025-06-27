import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
import configparser
import sys
import os

sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("paper", font_scale=1.2)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
})

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kernels import RBF, RQ, get_kernel

figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Created test figures directory: {figures_dir}")

@pytest.fixture
def config():
    """Fixture to provide config file"""
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    return config

@pytest.fixture
def test_points():
    """Fixture for test points"""
    return np.array([[1.0, 2.0, 3.0]]), np.array([[0.5, 1.5, 2.5]])

@pytest.mark.test_kernel
@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_vectorized_kernel_shape(config, kernel_type, test_points):
    """Test vectorized kernel output shape for different kernel types"""
    config.set("KERNEL", "type", kernel_type)
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))
    
    X1, X2 = test_points
    
    # Test with single points (population of one)
    result = kernel(X1, X2)
    assert result.shape == (1, 1)
    assert isinstance(result[0, 0], (int, float, np.number))
    assert result[0, 0] >= 0 and result[0, 0] <= 1
    
    # Test with multiple points
    X1_multi = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    X2_multi = np.array([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]])
    
    results = kernel(X1_multi, X2_multi)
    assert results.shape == (2, 2)
    assert np.all(results >= 0) and np.all(results <= 1)

@pytest.mark.test_kernel
@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_vectorized_kernel_symmetry(config, kernel_type):
    """Test that vectorized kernels are symmetric"""
    config.set("KERNEL", "type", kernel_type)
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))
    
    # Test symmetry: k(x,y) = k(y,x)
    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])
    
    K_xy = kernel(X1, X2)
    K_yx = kernel(X2, X1)
    
    assert np.allclose(K_xy, K_yx.T, rtol=1e-10)

@pytest.mark.test_kernel
@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_vectorized_kernel_identity(config, kernel_type):
    """Test that vectorized kernel at same point equals 1"""
    config.set("KERNEL", "type", kernel_type)
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))
    
    # Test identity: k(x,x) = 1
    X = np.array([[1.0, 2.0, 3.0]])
    K_xx = kernel(X, X)
    
    assert np.allclose(np.diag(K_xx), 1.0, rtol=1e-10)

@pytest.mark.test_kernel
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_vectorized_kernel_sigma_scaling(config, sigma):
    """Test that vectorized kernel values scale properly with sigma"""
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config, np.array([sigma, sigma, sigma]))
    
    # Test that larger sigma gives broader kernel
    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])
    
    k_value = kernel(X1, X2)[0, 0]
    
    # For RBF, larger sigma should give higher values for same distance
    if sigma == 0.5:
        assert k_value < 0.5  # Should be small for small sigma
    elif sigma == 2.0:
        assert k_value > 0.5  # Should be larger for large sigma

@pytest.mark.test_kernel
def test_vectorized_kernel_matrix_shape(config):
    """Test vectorized kernel matrix computation and shape"""
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))
    
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], 
                  [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])
    n = len(X)
    
    # Kernel matrix using vectorized computation
    K = kernel(X, X)
    
    assert K.shape == (n, n)
    assert K.shape == (5, 5)
    
    # Test symmetry
    assert np.allclose(K, K.T)
    
    # Test diagonal elements are 1
    assert np.allclose(np.diag(K), 1.0)

@pytest.mark.test_kernel
@pytest.mark.parametrize("sigma_type", ["scalar", "array"])
def test_vectorized_multivariate_kernel(sigma_type):
    """Test vectorized kernels with multivariate sigma"""
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    
    config.set("KERNEL", "type", "RBF")
    
    if sigma_type == "scalar":
        sigma = np.array([1.0, 1.0, 1.0])
        X1 = np.array([[1.0, 2.0, 3.0]])
        X2 = np.array([[0.5, 1.5, 2.5]])
    else:  # array
        sigma = np.array([1.0, 0.5, 2.0])
        X1 = np.array([[1.0, 2.0, 3.0]])
        X2 = np.array([[0.5, 1.5, 2.5]])
    
    kernel = get_kernel(config, sigma)
    
    result = kernel(X1, X2)
    assert result.shape == (1, 1)
    assert isinstance(result[0, 0], (int, float, np.number))
    assert result[0, 0] >= 0 and result[0, 0] <= 1
    
    # Test identity for multivariate
    K_xx = kernel(X1, X1)
    assert np.allclose(K_xx[0, 0], 1.0, rtol=1e-10)

@pytest.mark.test_kernel
@pytest.mark.error_handling
def test_invalid_kernel_type(config):
    """Test that invalid kernel type raises error"""
    config.set("KERNEL", "type", "INVALID_KERNEL")
    
    with pytest.raises(ValueError, match="Unknown kernel type"):
        get_kernel(config, np.array([1.0, 1.0, 1.0]))

@pytest.mark.visualization
def test_visual():
    """Visualize vectorized kernels with seaborn styling"""
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    
    x = np.linspace(-5, 5, 100).reshape(-1, 1)  # Reshape for vectorized kernels
    x0 = np.array([[0]])  # Reference point (population of one)
    
    kernel_types = ["RBF", "RQ"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    colors = sns.color_palette("husl", 2)
    
    for i, kernel_type in enumerate(kernel_types):
        config.set("KERNEL", "type", kernel_type)
        
        kernel = get_kernel(config, np.array([1.0]))
        
        k_values = np.array([kernel(x_val.reshape(1, 1), x0)[0, 0] for x_val in x])
        
        if kernel_type == "RBF":
            label = r'$k_{\text{RBF}}(x, x_0) = \exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$'
        else:  
            label = r'$k_{\text{RQ}}(x, x_0) = \left(1 + \frac{(x-x_0)^2}{2\alpha\sigma^2}\right)^{-\alpha}$'
        
        sns.lineplot(x=x.flatten(), y=k_values, color=colors[i], linewidth=2.5, 
                    label=label, ax=axes[i])
        axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                       linewidth=1.5, label=r'$x_0 = 0$')
        axes[i].set_xlabel(r'$x - x_0$', fontweight='bold')
        axes[i].set_ylabel(r'$k(x, x_0)$', fontweight='bold')
        axes[i].set_title(f'{kernel_type} Kernel (Vectorized)', fontweight='bold', pad=15)
        axes[i].legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1.1)
        
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  
    
    plt.savefig(os.path.join(figures_dir, 'kernels.png'), dpi=300, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
if __name__ == "__main__":
    print("\nRunning vectorized kernel tests...")
    
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    
    test_points = np.array([[1.0, 2.0, 3.0]]), np.array([[0.5, 1.5, 2.5]])
    
    test_vectorized_kernel_shape(config, "RBF", test_points)
    test_vectorized_kernel_shape(config, "RQ", test_points)
    
    test_vectorized_kernel_symmetry(config, "RBF")
    test_vectorized_kernel_symmetry(config, "RQ")
    test_vectorized_kernel_identity(config, "RBF")
    test_vectorized_kernel_identity(config, "RQ")
    
    test_vectorized_kernel_sigma_scaling(config, 1.0)
    test_vectorized_kernel_matrix_shape(config)
    test_vectorized_multivariate_kernel("scalar")
    test_vectorized_multivariate_kernel("array")
    test_invalid_kernel_type(config)
    
    print("All vectorized kernel tests passed!")
    
    print(f"Creating kernel visualizations in {figures_dir}...")
    test_visual()
    print("Kernel visualizations saved. These can be verified against https://www.cs.toronto.edu/~duvenaud/cookbook/")