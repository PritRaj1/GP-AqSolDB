import numpy as np
import time
import sys
import os
from configparser import ConfigParser
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dense_gp import GP
from src.sparse_gp import SparseGP

def create_config(kernel_type="RBF", lmbda=0.1, alpha=1.0, use_cache=True, 
                  cache_size=100, sparse=False, num_inducing=20):
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
    print("Testing Sparse GP vs Full GP...")
    
    np.random.seed(42)
    X_train = np.random.randn(300, 3) 
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 300)
    X_test = np.random.randn(100, 3)
    y_test = np.sin(X_test[:, 0]) * np.exp(X_test[:, 1]/5) + np.random.normal(0, 0.1, 100)
    
    sigma = np.array([1.0, 1.0, 1.0])
    
    # Full GP
    print("\n1. Full GP:")
    config_full = create_config("RBF", lmbda=0.1, sparse=False)
    
    start_time = time.time()
    gp_full = GP(config_full, sigma)
    gp_full.fit(X_train, y_train)
    fit_time_full = time.time() - start_time
    
    start_time = time.time()
    y_pred_full = gp_full.predict(X_test)
    pred_time_full = time.time() - start_time
    
    mse_full = mean_squared_error(y_test, y_pred_full)
    
    print(f"   Training time: {fit_time_full:.4f} seconds")
    print(f"   Prediction time: {pred_time_full:.4f} seconds")
    print(f"   MSE: {mse_full:.6f}")
    
    # Sparse GP
    print("\n2. Sparse GP (FITC):")
    config_sparse = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=50)
    
    start_time = time.time()
    gp_sparse = SparseGP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    fit_time_sparse = time.time() - start_time
    
    start_time = time.time()
    y_pred_sparse = gp_sparse.predict(X_test)
    pred_time_sparse = time.time() - start_time
    
    mse_sparse = mean_squared_error(y_test, y_pred_sparse)
    
    print(f"   Training time: {fit_time_sparse:.4f} seconds")
    print(f"   Prediction time: {pred_time_sparse:.4f} seconds")
    print(f"   MSE: {mse_sparse:.6f}")
    
    print(f"\n3. Performance Comparison:")
    speedup_fit = fit_time_full / fit_time_sparse if fit_time_sparse > 0 else float('inf')
    speedup_pred = pred_time_full / pred_time_sparse if pred_time_sparse > 0 else float('inf')
    
    print(f"   Training speedup: {speedup_fit:.2f}x")
    print(f"   Prediction speedup: {speedup_pred:.2f}x")
    print(f"   Accuracy ratio (sparse/full): {mse_sparse/mse_full:.3f}")
    
    sparse_info = gp_sparse.get_sparse_info()
    if sparse_info:
        print(f"   Inducing points: {sparse_info['num_inducing']}")
        print(f"   Training points: {sparse_info['num_training']}")
        print(f"   Compression ratio: {sparse_info['compression_ratio']:.3f}")

def test_different_inducing_points():
    print("\n" + "="*60)
    print("Testing different numbers of inducing points...")
    
    np.random.seed(42)
    X_train = np.random.randn(200, 2)
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, 200)
    X_test = np.random.randn(50, 2)
    y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1]) + np.random.normal(0, 0.1, 50)
    
    sigma = np.array([1.0, 1.0])
    
    inducing_counts = [10, 20, 50, 100]
    
    for num_inducing in inducing_counts:
        print(f"\nTesting with {num_inducing} inducing points:")
        
        config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=num_inducing)
        
        start_time = time.time()
        gp = SparseGP(config, sigma)
        gp.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        y_pred = gp.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        sparse_info = gp.get_sparse_info()
        
        print(f"   Training time: {fit_time:.4f}s")
        print(f"   MSE: {mse:.6f}")
        print(f"   Compression ratio: {sparse_info['compression_ratio']:.3f}")

def test_sparse_gp_with_uncertainty():
    print("\n" + "="*60)
    print("Testing sparse GP uncertainty quantification...")
    
    np.random.seed(42)
    X_train = np.random.randn(150, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 150)
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_test = np.column_stack([X_test, np.zeros_like(X_test)])
    
    sigma = np.array([1.0, 1.0])
    
    # Full GP
    config_full = create_config("RBF", lmbda=0.1, sparse=False)
    gp_full = GP(config_full, sigma)
    gp_full.fit(X_train, y_train)
    y_pred_full, y_std_full = gp_full.predict(X_test, return_std=True)
    
    # Sparse GP
    config_sparse = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=30)
    gp_sparse = SparseGP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse, y_std_sparse = gp_sparse.predict(X_test, return_std=True)
    
    print(f"\nUncertainty comparison:")
    print(f"   Full GP mean std: {np.mean(y_std_full):.4f}")
    print(f"   Sparse GP mean std: {np.mean(y_std_sparse):.4f}")
    print(f"   Uncertainty ratio (sparse/full): {np.mean(y_std_sparse)/np.mean(y_std_full):.3f}")
    
    assert np.all(y_std_full > 0), "Full GP uncertainties should be positive"
    assert np.all(y_std_sparse > 0), "Sparse GP uncertainties should be positive"
    print("   ✓ All uncertainties are positive")

def test_sparse_gp_edge_cases():
    print("\n" + "="*60)
    print("Testing sparse GP edge cases...")
    
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    sigma = np.array([1.0, 1.0])
    
    # 1. num_inducing >= N (should use all points)
    print("\n1. num_inducing >= N (should use all points):")
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=100)
    gp = SparseGP(config, sigma)
    gp.fit(X_train, y_train)
    
    sparse_info = gp.get_sparse_info()
    print(f"   Sparse info: {sparse_info}")
    print(f"   Is fitted: {gp.is_fitted}")
    
    # 2. Test prediction without fitting
    print("\n2. Test prediction without fitting:")
    try:
        gp_unfitted = SparseGP(config, sigma)
        gp_unfitted.predict(X_train)
        print("   ✗ Should have raised an error")
    except ValueError as e:
        print(f"   ✓ Correctly raised error: {e}")

def test_sparse_gp_with_different_kernels():
    print("\n" + "="*60)
    print("Testing sparse GP with different kernels...")
    
    np.random.seed(42)
    X_train = np.random.randn(200, 3)
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 200)
    X_test = np.random.randn(50, 3)
    y_test = np.sin(X_test[:, 0]) * np.exp(X_test[:, 1]/5) + np.random.normal(0, 0.1, 50)
    
    sigma = np.array([1.0, 1.0, 1.0])
    
    kernels = ["RBF", "RQ"]
    
    for kernel_type in kernels:
        print(f"\nTesting {kernel_type} kernel:")
        
        # Full GP
        config_full = create_config(kernel_type, lmbda=0.1, alpha=2.0, sparse=False)
        gp_full = GP(config_full, sigma)
        gp_full.fit(X_train, y_train)
        y_pred_full = gp_full.predict(X_test)
        mse_full = mean_squared_error(y_test, y_pred_full)
        
        # Sparse GP
        config_sparse = create_config(kernel_type, lmbda=0.1, alpha=2.0, sparse=True, num_inducing=40)
        gp_sparse = SparseGP(config_sparse, sigma)
        gp_sparse.fit(X_train, y_train)
        y_pred_sparse = gp_sparse.predict(X_test)
        mse_sparse = mean_squared_error(y_test, y_pred_sparse)
        
        print(f"   Full GP MSE: {mse_full:.6f}")
        print(f"   Sparse GP MSE: {mse_sparse:.6f}")
        print(f"   Accuracy ratio: {mse_sparse/mse_full:.3f}")

def test_inducing_point_selection_methods():
    """Test different inducing point selection methods"""
    print("\n" + "="*60)
    print("Testing different inducing point selection methods...")
    
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 100)
    X_test = np.random.randn(20, 2)
    y_test = np.sin(X_test[:, 0]) + np.random.normal(0, 0.1, 20)
    
    sigma = np.array([1.0, 1.0])
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=20)
    
    methods = ['random', 'uniform']
    
    for method in methods:
        print(f"\nTesting {method} selection method:")
        
        gp = SparseGP(config, sigma)
        gp.fit(X_train, y_train, inducing_method=method)
        
        y_pred = gp.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        sparse_info = gp.get_sparse_info()
        
        print(f"   MSE: {mse:.6f}")
        print(f"   Compression ratio: {sparse_info['compression_ratio']:.3f}")

if __name__ == "__main__":
    test_sparse_vs_full_gp()
    test_different_inducing_points()
    test_sparse_gp_with_uncertainty()
    test_sparse_gp_edge_cases()
    test_sparse_gp_with_different_kernels()
    test_inducing_point_selection_methods() 