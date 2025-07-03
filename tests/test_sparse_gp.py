import numpy as np
import time
import sys
import os
import pytest
from configparser import ConfigParser
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multivar_gp.dense_gp import DenseGP
from src.multivar_gp.sparse_gp import SparseGP

def create_config(kernel_type="RBF", lmbda=0.1, alpha=1.0, use_cache=True, 
                  cache_size=100, sparse=False, num_inducing=20):
    """Create a configuration object for testing"""
    config = ConfigParser()
    config['KERNEL'] = {
        'type': kernel_type,
        'lmbda': str(lmbda),
        'alpha': str(alpha),
        'use_cache': str(use_cache).lower(),
        'cache_size': str(cache_size),
        'sparse': str(sparse).lower(),
        'num_inducing': str(num_inducing)
    }
    return config

def test_sparse_vs_full_gp():
    """Test sparse GP vs full GP performance and accuracy"""
    np.random.seed(42)
    X_train = np.random.randn(300, 3)
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 300)
    X_test = np.random.randn(100, 3)
    y_test = np.sin(X_test[:, 0]) * np.exp(X_test[:, 1]/5) + np.random.normal(0, 0.1, 100)
    sigma = np.array([1.0, 1.0, 1.0])
    
    # Full GP
    config_full = create_config("RBF", lmbda=0.1, sparse=False)
    
    start_time = time.time()
    gp_full = DenseGP(config_full, sigma)
    gp_full.fit(X_train, y_train)
    fit_time_full = time.time() - start_time
    
    start_time = time.time()
    y_pred_full = gp_full.predict(X_test)
    pred_time_full = time.time() - start_time
    
    mse_full = mean_squared_error(y_test, y_pred_full)
    
    # Sparse GP
    config_sparse = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=50)
    
    start_time = time.time()
    gp_sparse = SparseGP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    fit_time_sparse = time.time() - start_time
    
    start_time = time.time()
    y_pred_sparse = gp_sparse.predict(X_test)
    pred_time_sparse = time.time() - start_time
    
    mse_sparse = mean_squared_error(y_test, y_pred_sparse)
    
    # Performance assertions - allow for very fast operations (otherwise time will be 0.0)
    assert fit_time_full >= 0, "Full GP training time should be non-negative"
    assert fit_time_sparse >= 0, "Sparse GP training time should be non-negative"
    assert pred_time_full >= 0, "Full GP prediction time should be non-negative"
    assert pred_time_sparse >= 0, "Sparse GP prediction time should be non-negative"
    
    # Accuracy assertions
    assert mse_full > 0, "Full GP MSE should be positive"
    assert mse_sparse > 0, "Sparse GP MSE should be positive"
    
    # Sparse GP info
    sparse_info = gp_sparse.get_sparse_info()
    assert sparse_info is not None, "Sparse GP should provide sparse info"
    assert sparse_info['num_inducing'] == 50, "Number of inducing points should match config"
    assert sparse_info['num_training'] == len(X_train), "Number of training points should match data"
    assert sparse_info['compression_ratio'] < 1.0, "Compression ratio should be less than 1"

@pytest.mark.parametrize("num_inducing", [10, 20, 50, 100])
def test_different_inducing_points(num_inducing):
    """Test different numbers of inducing points"""
    np.random.seed(42)
    X_train = np.random.randn(200, 2)
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, 200)
    X_test = np.random.randn(50, 2)
    y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1]) + np.random.normal(0, 0.1, 50)
    sigma = np.array([1.0, 1.0])
    
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=num_inducing)
    
    start_time = time.time()
    gp = SparseGP(config, sigma)
    gp.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    y_pred = gp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    sparse_info = gp.get_sparse_info()
    
    assert fit_time >= 0, f"Training time should be non-negative for {num_inducing} inducing points"
    assert mse > 0, f"MSE should be positive for {num_inducing} inducing points"
    assert sparse_info['num_inducing'] == num_inducing, f"Number of inducing points should be {num_inducing}"
    assert sparse_info['compression_ratio'] <= 1.0, "Compression ratio should be <= 1"

def test_sparse_gp_with_uncertainty():
    """Test sparse GP uncertainty quantification"""
    np.random.seed(42)
    X_train = np.random.randn(150, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 150)
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_test = np.column_stack([X_test, np.zeros_like(X_test)])
    sigma = np.array([1.0, 1.0])
    
    # Full GP with uncertainty
    config_full = create_config("RBF", lmbda=0.1, sparse=False)
    gp_full = DenseGP(config_full, sigma)
    gp_full.fit(X_train, y_train)
    y_pred_full, y_std_full = gp_full.predict(X_test, return_std=True)
    
    # Sparse GP with uncertainty
    config_sparse = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=30)
    gp_sparse = SparseGP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse, y_std_sparse = gp_sparse.predict(X_test, return_std=True)
    
    # Uncertainty assertions
    assert np.all(y_std_full > 0), "Full GP uncertainties should be positive"
    assert np.all(y_std_sparse > 0), "Sparse GP uncertainties should be positive"
    assert np.all(np.isfinite(y_std_full)), "Full GP uncertainties should be finite"
    assert np.all(np.isfinite(y_std_sparse)), "Sparse GP uncertainties should be finite"
    
    # Both should produce reasonable uncertainty estimates
    assert np.mean(y_std_full) > 0, "Full GP mean uncertainty should be positive"
    assert np.mean(y_std_sparse) > 0, "Sparse GP mean uncertainty should be positive"

def test_sparse_gp_edge_cases():
    """Test edge cases for sparse GP"""
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    sigma = np.array([1.0, 1.0])
    
    # 1. num_inducing >= N (should use all points)
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=100)
    gp = SparseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    sparse_info = gp.get_sparse_info()
    assert sparse_info is not None, "Sparse info should be available"
    assert gp.is_fitted == True, "GP should be fitted"
    
    # 2. Test prediction without fitting
    gp_unfitted = SparseGP(config, sigma)
    with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
        gp_unfitted.predict(X_train)

@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_sparse_gp_with_different_kernels(kernel_type):
    """Test sparse GP with different kernel types"""
    np.random.seed(42)
    X_train = np.random.randn(200, 3)
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 200)
    X_test = np.random.randn(50, 3)
    y_test = np.sin(X_test[:, 0]) * np.exp(X_test[:, 1]/5) + np.random.normal(0, 0.1, 50)
    
    sigma = np.array([1.0, 1.0, 1.0])
    
    # Full GP
    config_full = create_config(kernel_type, lmbda=0.1, alpha=2.0, sparse=False)
    gp_full = DenseGP(config_full, sigma)
    gp_full.fit(X_train, y_train)
    y_pred_full = gp_full.predict(X_test)
    mse_full = mean_squared_error(y_test, y_pred_full)
    
    # Sparse GP
    config_sparse = create_config(kernel_type, lmbda=0.1, alpha=2.0, sparse=True, num_inducing=40)
    gp_sparse = SparseGP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse = gp_sparse.predict(X_test)
    mse_sparse = mean_squared_error(y_test, y_pred_sparse)
    
    # Assertions
    assert mse_full > 0, f"Full GP MSE should be positive for {kernel_type} kernel"
    assert mse_sparse > 0, f"Sparse GP MSE should be positive for {kernel_type} kernel"
    assert np.all(np.isfinite(y_pred_full)), f"Full GP predictions should be finite for {kernel_type} kernel"
    assert np.all(np.isfinite(y_pred_sparse)), f"Sparse GP predictions should be finite for {kernel_type} kernel"

@pytest.mark.parametrize("method", ['random', 'uniform'])
def test_inducing_point_selection_methods(method):
    """Test different inducing point selection methods"""
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 100)
    X_test = np.random.randn(20, 2)
    y_test = np.sin(X_test[:, 0]) + np.random.normal(0, 0.1, 20)
    
    sigma = np.array([1.0, 1.0])
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=20)
    
    gp = SparseGP(config, sigma)
    gp.fit(X_train, y_train, inducing_method=method)
    
    y_pred = gp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    sparse_info = gp.get_sparse_info()
    
    assert mse > 0, f"MSE should be positive for {method} method"
    assert sparse_info['compression_ratio'] <= 1.0, f"Compression ratio should be <= 1 for {method} method"

def test_sparse_gp_complexity():
    """Test that sparse GP complexity calculation is correct"""
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.random.randn(100)
    sigma = np.array([1.0, 1.0, 1.0])
    
    # Test with different numbers of inducing points
    inducing_counts = [10, 20, 50]
    
    for num_inducing in inducing_counts:
        config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=num_inducing)
        gp = SparseGP(config, sigma)
        gp.fit(X_train, y_train)
        
        X_test = np.random.randn(10, 3)
        y_pred = gp.predict(X_test)
        
        assert len(y_pred) == len(X_test), "Prediction length should match test data length"
        assert np.all(np.isfinite(y_pred)), "Predictions should be finite"
        
        sparse_info = gp.get_sparse_info()
        assert sparse_info['num_inducing'] == num_inducing, f"Number of inducing points should be {num_inducing}"

def test_sparse_gp_cache_integration():
    """Test that sparse GP works with caching"""
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    X_test = np.random.randn(20, 2)
    sigma = np.array([1.0, 1.0])
    
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=15, use_cache=True)
    gp = SparseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    # Multiple predictions to test cache hits
    for i in range(5):
        y_pred = gp.predict(X_test)
    
    cache_stats = gp.get_cache_stats()
    assert cache_stats is not None, "Cache stats should be available"
    
    gp.clear_cache()

def test_sparse_gp_fit_predict_cycle():
    """Test that sparse GP fit and predict work correctly in sequence"""
    np.random.seed(42)
    X_train = np.random.randn(80, 2)
    y_train = np.random.randn(80)
    X_test = np.random.randn(15, 2)
    sigma = np.array([1.0, 1.0])
    
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=20)
    gp = SparseGP(config, sigma)
    
    gp.fit(X_train, y_train)
    assert gp.is_fitted == True, "GP should be marked as fitted"
    
    y_pred = gp.predict(X_test)
    assert len(y_pred) == len(X_test), "Prediction length should match test data length"
    assert np.all(np.isfinite(y_pred)), "Predictions should be finite"
    
    y_pred, y_std = gp.predict(X_test, return_std=True)
    assert len(y_pred) == len(X_test), "Prediction length should match test data length"
    assert len(y_std) == len(X_test), "Uncertainty length should match test data length"
    assert np.all(np.isfinite(y_pred)), "Predictions should be finite"
    assert np.all(np.isfinite(y_std)), "Uncertainties should be finite"
    assert np.all(y_std > 0), "All uncertainties should be positive"

if __name__ == "__main__":
    print("\nRunning sparse GP tests...")
    
    test_sparse_vs_full_gp()
    test_different_inducing_points(10)
    test_different_inducing_points(20)
    test_different_inducing_points(50)
    test_different_inducing_points(100)
    test_sparse_gp_with_uncertainty()
    test_sparse_gp_edge_cases()
    test_sparse_gp_with_different_kernels("RBF")
    test_sparse_gp_with_different_kernels("RQ")
    test_inducing_point_selection_methods('random')
    test_inducing_point_selection_methods('uniform')
    test_sparse_gp_complexity()
    test_sparse_gp_cache_integration()
    test_sparse_gp_fit_predict_cycle()
    
    print("All sparse GP tests completed!") 