import numpy as np
import matplotlib.pyplot as plt
import pytest
import configparser
import sys
import os

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

def test_rbf_kernel_shape(config):
    """Test RBF kernel output shape"""
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config)
    
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

def test_rq_kernel_shape(config):
    """Test RQ kernel output shape"""
    config.set("KERNEL", "type", "RQ")
    kernel = get_kernel(config)
    
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

def test_kernel_symmetry(config):
    """Test that kernels are symmetric"""
    kernel_types = ["RBF", "RQ"]
    
    for kernel_type in kernel_types:
        config.set("KERNEL", "type", kernel_type)
        kernel = get_kernel(config)
        
        # Test symmetry: k(x,y) = k(y,x)
        x, y = 1.0, 2.0
        k_xy = kernel(x, y)
        k_yx = kernel(y, x)
        
        assert np.isclose(k_xy, k_yx, rtol=1e-10)

def test_kernel_identity(config):
    """Test that kernel at same point equals 1"""
    kernel_types = ["RBF", "RQ"]
    
    for kernel_type in kernel_types:
        config.set("KERNEL", "type", kernel_type)
        kernel = get_kernel(config)
        
        # Test identity: k(x,x) = 1
        x = 1.0
        k_xx = kernel(x, x)
        
        assert np.isclose(k_xx, 1.0, rtol=1e-10)

def test_kernel_matrix_shape(config):
    """Test kernel matrix computation and shape"""
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config)
    
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

def test_visual():
    """Visualize kernels"""
    config = configparser.ConfigParser()
    config.read('config/GP.ini')
    
    x = np.linspace(-5, 5, 100)
    x0 = 0  # Reference point
    
    kernel_types = ["RBF", "RQ"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, kernel_type in enumerate(kernel_types):
        config.set("KERNEL", "type", kernel_type)
        
        kernel = get_kernel(config)
        
        k_values = [kernel(x_val, x0) for x_val in x]
        
        axes[i].plot(x, k_values, 'b-', linewidth=2, label=f'{kernel_type} Kernel')
        axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Reference point')
        axes[i].set_xlabel('Distance from reference point')
        axes[i].set_ylabel('Kernel value')
        axes[i].set_title(f'{kernel_type} Kernel Shape')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'kernels.png'))
    plt.close()
    
if __name__ == "__main__":
    
    print("\nRunning pytest tests...")
    pytest.main([__file__, "-v"])

    print(f"Visualizing kernels in {figures_dir}. These can be verified against https://www.cs.toronto.edu/~duvenaud/cookbook/")
    test_visual()