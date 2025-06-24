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
from src.kernels import get_kernel

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

@pytest.mark.test_kernel
def test_rbf_kernel_shape(config):
    """Test RBF kernel output shape"""
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config, 1.0)
    
    # Test single
    result = kernel(1.0, 2.0)
    assert isinstance(result, (int, float, np.number))
    assert result >= 0 and result <= 1
    
    # Test array
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.5, 1.5, 2.5])
    results = np.array([kernel(xi, yi) for xi, yi in zip(x, y)])
    
    assert results.shape == (3,)
    assert np.all(results >= 0) and np.all(results <= 1)

@pytest.mark.test_kernel
def test_rq_kernel_shape(config):
    """Test RQ kernel output shape"""
    config.set("KERNEL", "type", "RQ")
    kernel = get_kernel(config, 1.0)
    
    # Test single
    result = kernel(1.0, 2.0)
    assert isinstance(result, (int, float, np.number))
    assert result >= 0 and result <= 1
    
    # Test array
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.5, 1.5, 2.5])
    results = np.array([kernel(xi, yi) for xi, yi in zip(x, y)])
    
    assert results.shape == (3,)
    assert np.all(results >= 0) and np.all(results <= 1)

@pytest.mark.test_kernel
def test_kernel_symmetry(config):
    """Test that kernels are symmetric"""
    kernel_types = ["RBF", "RQ"]
    
    for kernel_type in kernel_types:
        config.set("KERNEL", "type", kernel_type)
        kernel = get_kernel(config, 1.0)
        
        # Test symmetry: k(x,y) = k(y,x)
        x, y = 1.0, 2.0
        k_xy = kernel(x, y)
        k_yx = kernel(y, x)
        
        assert np.isclose(k_xy, k_yx, rtol=1e-10)

@pytest.mark.test_kernel
def test_kernel_identity(config):
    """Test that kernel at same point equals 1"""
    kernel_types = ["RBF", "RQ"]
    
    for kernel_type in kernel_types:
        config.set("KERNEL", "type", kernel_type)
        kernel = get_kernel(config, 1.0)
        
        # Test identity: k(x,x) = 1
        x = 1.0
        k_xx = kernel(x, x)
        
        assert np.isclose(k_xx, 1.0, rtol=1e-10)

@pytest.mark.test_kernel
def test_kernel_matrix_shape(config):
    """Test kernel matrix computation and shape"""
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config, 1.0)
    
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = len(x)
    
    # Kernel matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(x[i], x[j])
    
    assert K.shape == (n, n)
    assert K.shape == (5, 5)
    
    # Test symmetry
    assert np.allclose(K, K.T)
    
    # Test diagonal elements are 1
    assert np.allclose(np.diag(K), 1.0)

@pytest.mark.test_kernel
def test_multivariate_kernel():
    """Test kernels with multivariate sigma"""
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    
    config.set("KERNEL", "type", "RBF")
    sigma_multivariate = np.array([1.0, 0.5])
    kernel = get_kernel(config, sigma_multivariate)
    
    x = np.array([1.0, 2.0])
    y = np.array([0.5, 1.5])
    
    result = kernel(x, y)
    assert isinstance(result, (int, float, np.number))
    assert result >= 0 and result <= 1
    
    k_xx = kernel(x, x)
    assert np.isclose(k_xx, 1.0, rtol=1e-10)

@pytest.mark.test_kernel
def test_invalid_kernel_type(config):
    """Test that invalid kernel type raises error"""
    config.set("KERNEL", "type", "INVALID_KERNEL")
    
    with pytest.raises(ValueError, match="Unknown kernel type"):
        get_kernel(config, 1.0)

@pytest.mark.visualization
def test_visual():
    """Visualize kernels with seaborn styling"""
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    
    x = np.linspace(-5, 5, 100)
    x0 = 0  # Reference point
    
    kernel_types = ["RBF", "RQ"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    colors = sns.color_palette("husl", 2)
    
    for i, kernel_type in enumerate(kernel_types):
        config.set("KERNEL", "type", kernel_type)
        
        kernel = get_kernel(config, 1.0)
        
        k_values = [kernel(x_val, x0) for x_val in x]
        
        if kernel_type == "RBF":
            label = r'$k_{\text{RBF}}(x, x_0) = \exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$'
        else:  
            label = r'$k_{\text{RQ}}(x, x_0) = \left(1 + \frac{(x-x_0)^2}{2\alpha\sigma^2}\right)^{-\alpha}$'
        
        sns.lineplot(x=x, y=k_values, color=colors[i], linewidth=2.5, 
                    label=label, ax=axes[i])
        axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                       linewidth=1.5, label=r'$x_0 = 0$')
        axes[i].set_xlabel(r'$x - x_0$', fontweight='bold')
        axes[i].set_ylabel(r'$k(x, x_0)$', fontweight='bold')
        axes[i].set_title(f'{kernel_type} Kernel', fontweight='bold', pad=15)
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
    print("\nRunning kernel tests...")
    
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    
    test_rbf_kernel_shape(config)
    test_rq_kernel_shape(config)
    test_kernel_symmetry(config)
    test_kernel_identity(config)
    test_kernel_matrix_shape(config)
    test_multivariate_kernel()
    test_invalid_kernel_type(config)
    
    print("All kernel tests passed!")
    
    print(f"Creating kernel visualizations in {figures_dir}...")
    test_visual()
    print("Kernel visualizations saved. These can be verified against https://www.cs.toronto.edu/~duvenaud/cookbook/")