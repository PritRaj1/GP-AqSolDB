import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
from configparser import ConfigParser
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
from src.dense_gp import DenseGP
from tests.fcn import get_data

def create_config(kernel_type="RBF", lmbda=0.1, alpha=1.0):
    """Create a variable config object for testing"""
    config = ConfigParser()
    config['KERNEL'] = {
        'type': kernel_type,
        'lmbda': str(lmbda),
        'alpha': str(alpha)
    }
    return config

@pytest.fixture
def sample_data_1d():
    """Fixture for 1D test data"""
    return get_data(num_points=20, noise=True, noise_std=0.1, x_range=(0, 10))

@pytest.fixture
def sample_data_2d():
    """Fixture for 2D test data"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (30, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 30)
    return X_train, y_train

@pytest.mark.parametrize("kernel_type,sigma", [
    ("RBF", 1.0),
    ("RQ", 1.0),
])
def test_gp_basic_functionality(kernel_type, sigma, sample_data_1d):
    """Test basic GP functionality with different kernels"""
    config = create_config(kernel_type=kernel_type, lmbda=0.1)
    X_train, y_train = sample_data_1d
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    y_pred = gp.predict(X_test, return_std=False)
    
    assert len(y_pred) == len(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert not np.any(np.isnan(y_pred))

@pytest.mark.parametrize("sigma_type", ["univariate", "multivariate"])
def test_gp_sigma_types(sigma_type, sample_data_1d, sample_data_2d):
    """Test GP with different sigma types"""
    config = create_config(kernel_type="RBF", lmbda=0.1)
    
    if sigma_type == "univariate":
        sigma = 1.0
        X_train, y_train = sample_data_1d
        X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    else:  # multivariate
        sigma = np.array([1.0, 0.5])
        X_train, y_train = sample_data_2d
        x1 = np.linspace(0, 5, 20)
        x2 = np.linspace(0, 5, 20)
        X1, X2 = np.meshgrid(x1, x2)
        X_test = np.column_stack([X1.ravel(), X2.ravel()])
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    assert len(y_pred) == len(X_test)
    assert len(y_std) == len(X_test)
    assert np.all(y_std >= 0)
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isnan(y_std))

def test_uncertainty_behavior():
    """Test that uncertainty behaves as expected"""
    config = create_config(kernel_type="RBF", lmbda=0.01)
    sigma = 1.0
    
    # Sparse dataset
    X_train = np.array([1.0, 3.0, 7.0, 9.0]).reshape(-1, 1)
    y_train = np.sin(X_train.flatten()) + np.random.normal(0, 0.01, 4)
    
    # Test points including and beyond training points
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # Uncertainty should be lower at training points
    training_indices = []
    for i, x in enumerate(X_test):
        if np.any(np.abs(x - X_train) < 1e-6):
            training_indices.append(i)
    
    if training_indices:
        training_uncertainty = y_std[training_indices]
        other_uncertainty = y_std[~np.isin(np.arange(len(X_test)), training_indices)]
        assert np.mean(training_uncertainty) < np.mean(other_uncertainty)

def test_uncertainty_distance_relationship(sample_data_1d):
    """Test that uncertainty increases with distance from training points"""
    config = create_config(kernel_type="RBF", lmbda=0.1)
    sigma = 1.0
    
    X_train, y_train = sample_data_1d
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # Test that uncertainty is higher when far away from training points
    distances = np.min(np.abs(X_test - X_train), axis=1)
    far_points = distances > 2.0
    near_points = distances < 0.5
    
    if np.any(far_points) and np.any(near_points):
        assert np.mean(y_std[far_points]) > np.mean(y_std[near_points])

def test_gp_fit_attributes(sample_data_1d):
    """Test that GP fit method sets all required attributes"""
    config = create_config(kernel_type="RBF", lmbda=0.1)
    sigma = 1.0
    
    X_train, y_train = sample_data_1d
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    assert gp.X_train is not None
    assert gp.y_train is not None
    assert gp.X_train.shape[0] == len(y_train)
    assert gp.y_train.shape[0] == len(y_train)

def test_gp_invalid_inputs():
    """Test GP error handling for invalid inputs"""
    config = create_config(kernel_type="RBF", lmbda=0.1)
    sigma = 1.0
    
    gp = DenseGP(config, sigma)
    
    # Test with empty training data
    with pytest.raises(ValueError):
        gp.fit(np.array([]), np.array([]))
    
    # Test with mismatched X and y lengths
    with pytest.raises(ValueError):
        gp.fit(np.array([[1], [2]]), np.array([1]))

def test_gp_univariate_sigma(return_data=False):
    """Test GP with univariate sigma (single length scale)"""
    config = create_config(kernel_type="RBF", lmbda=0.1)
    sigma = 1.0 
    
    X_train, y_train = get_data(num_points=20, noise=True, noise_std=0.1, x_range=(0, 10))
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    assert len(y_pred) == len(X_test)
    assert len(y_std) == len(X_test)
    assert np.all(y_std >= 0)  # Standard deviations should be non-negative
    
    # Test that uncertainty is higher, when far away from training points
    distances = np.min(np.abs(X_test - X_train), axis=1)
    far_points = distances > 2.0
    near_points = distances < 0.5
    
    if np.any(far_points) and np.any(near_points):
        assert np.mean(y_std[far_points]) > np.mean(y_std[near_points])

    if return_data:
        return X_train, y_train, X_test, y_pred, y_std

def test_gp_multivariate_sigma(return_data=False):
    """Test GP with multivariate sigma (different length scales per feature)"""
    config = create_config(kernel_type="RBF", lmbda=0.1)
    sigma = np.array([1.0, 0.5])  
    
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (30, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 30)
    
    x1 = np.linspace(0, 5, 20)
    x2 = np.linspace(0, 5, 20)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    y_pred_grid = y_pred.reshape(X1.shape)
    y_std_grid = y_std.reshape(X1.shape)
    
    assert len(y_pred) == len(X_test)
    assert len(y_std) == len(X_test)
    assert np.all(y_std >= 0)

    if return_data:
        return X_train, y_train, X_test, y_pred_grid, y_std_grid, X1, X2

def test_gp_rational_quadratic_kernel(return_data=False):
    """Test GP with Rational Quadratic kernel"""
    config = create_config(kernel_type="RQ", lmbda=0.1, alpha=2.0)
    sigma = 1.0
    
    X_train, y_train = get_data(num_points=25, noise=True, noise_std=0.1, x_range=(0, 10))
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    assert len(y_pred) == len(X_test)
    assert len(y_std) == len(X_test)
    assert np.all(y_std >= 0)

    if return_data:
        return X_train, y_train, X_test, y_pred, y_std

def visualize_gp_results():
    """Create visualizations for GP testing"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    X_train, y_train, X_test, y_pred, y_std = test_gp_univariate_sigma(return_data=True)
    
    sns.scatterplot(x=X_train.flatten(), y=y_train, color='red', s=30, 
                   label='Training Data', ax=axes[0, 0], zorder=5)
    axes[0, 0].plot(X_test, y_pred, color='steelblue', linewidth=2.5, 
                   label=r'$f(x)$ Prediction', alpha=0.8)
    axes[0, 0].fill_between(X_test.flatten(), 
                           y_pred - 2*y_std, 
                           y_pred + 2*y_std, 
                           alpha=0.3, color='steelblue', label=r'$\pm 2\sigma$ Uncertainty')
    axes[0, 0].set_title(r'Univariate $\sigma$ (RBF Kernel)', fontweight='bold', pad=15)
    axes[0, 0].set_xlabel(r'$x$', fontweight='bold')
    axes[0, 0].set_ylabel(r'$y$', fontweight='bold')
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 0].grid(True, alpha=0.3)
    
    X_train_2d, y_train_2d, X_test_2d, y_pred_grid, y_std_grid, X1, X2 = test_gp_multivariate_sigma(return_data=True)
    
    im1 = axes[0, 1].contourf(X1, X2, y_pred_grid, levels=25, cmap='viridis', alpha=0.8)
    sns.scatterplot(x=X_train_2d[:, 0], y=X_train_2d[:, 1], color='red', s=60, 
                   marker='x', label='Training Data', ax=axes[0, 1])
    axes[0, 1].set_title(r'Mean Prediction $\mathbb{E}[f(\mathbf{x})]$', fontweight='bold', pad=15)
    axes[0, 1].set_xlabel(r'$x_1$', fontweight='bold')
    axes[0, 1].set_ylabel(r'$x_2$', fontweight='bold')
    axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    cbar1.set_label(r'$\mathbb{E}[f(\mathbf{x})]$', fontweight='bold')
    
    im2 = axes[1, 0].contourf(X1, X2, y_std_grid, levels=25, cmap='plasma', alpha=0.8)
    sns.scatterplot(x=X_train_2d[:, 0], y=X_train_2d[:, 1], color='lime', s=60, 
                   marker='x', label='Training Data', ax=axes[1, 0])
    axes[1, 0].set_title(r'Uncertainty $\sqrt{\text{Var}[f(\mathbf{x})]}$', fontweight='bold', pad=15)
    axes[1, 0].set_xlabel(r'$x_1$', fontweight='bold')
    axes[1, 0].set_ylabel(r'$x_2$', fontweight='bold')
    axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
    cbar2 = plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)
    cbar2.set_label(r'$\sigma(\mathbf{x})$', fontweight='bold')
    
    X_train_rq, y_train_rq, X_test_rq, y_pred_rq, y_std_rq = test_gp_rational_quadratic_kernel(return_data=True)
    
    sns.scatterplot(x=X_train_rq.flatten(), y=y_train_rq, color='red', s=30, 
                   label='Training Data', ax=axes[1, 1], zorder=5)
    axes[1, 1].plot(X_test_rq, y_pred_rq, color='forestgreen', linewidth=2.5, 
                   label=r'$f(x)$ Prediction (RQ)', alpha=0.8)
    axes[1, 1].fill_between(X_test_rq.flatten(), 
                           y_pred_rq - 2*y_std_rq, 
                           y_pred_rq + 2*y_std_rq, 
                           alpha=0.3, color='forestgreen', label=r'$\pm 2\sigma$ Uncertainty')
    axes[1, 1].set_title(r'Rational Quadratic Kernel', fontweight='bold', pad=15)
    axes[1, 1].set_xlabel(r'$x$', fontweight='bold')
    axes[1, 1].set_ylabel(r'$y$', fontweight='bold')
    axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  
    
    plt.savefig('tests/figures/gp_test_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

if __name__ == "__main__":
    print("\nRunning GP tests...")
    
    X_train_1d, y_train_1d = get_data(num_points=20, noise=True, noise_std=0.1, x_range=(0, 10))
    sample_data_1d = (X_train_1d, y_train_1d)
    
    np.random.seed(42)
    X_train_2d = np.random.uniform(0, 5, (30, 2))
    y_train_2d = np.sin(X_train_2d[:, 0]) * np.exp(X_train_2d[:, 1]/5) + np.random.normal(0, 0.1, 30)
    sample_data_2d = (X_train_2d, y_train_2d)
    
    test_gp_basic_functionality("RBF", 1.0, sample_data_1d)
    test_gp_basic_functionality("RQ", 1.0, sample_data_1d)
    test_gp_sigma_types("univariate", sample_data_1d, sample_data_2d)
    test_gp_sigma_types("multivariate", sample_data_1d, sample_data_2d)
    test_uncertainty_behavior()
    test_uncertainty_distance_relationship(sample_data_1d)
    test_gp_fit_attributes(sample_data_1d)
    test_gp_invalid_inputs()
    test_gp_univariate_sigma()
    test_gp_multivariate_sigma()
    test_gp_rational_quadratic_kernel()
    
    print("All tests passed!")
    
    print("Creating visualizations...")
    visualize_gp_results()
    print("Visualizations saved to tests/figures/gp_test_results.png")
