import numpy as np
import sys
import os
from configparser import ConfigParser
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gp import GP

def create_config(use_sparse=False, num_inducing=20, inducing_method='random'):
    """Create a configuration object for testing"""
    config = ConfigParser()
    config['KERNEL'] = {
        'type': 'RBF',
        'lmbda': '0.1',
        'alpha': '1.0',
        'use_cache': 'true',
        'cache_size': '100'
    }
    config['SPARSE'] = {
        'use_sparse': str(use_sparse).lower(),
        'num_inducing': str(num_inducing),
        'inducing_method': inducing_method
    }
    return config

def test_unified_gp_switching():
    """Test that the unified GP correctly switches between dense and sparse"""
    print("Testing Unified GP Switching")
    print("=" * 50)
    
    np.random.seed(42)
    X_train = np.random.randn(200, 3)
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, 200)
    X_test = np.random.randn(50, 3)
    y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1]) + np.random.normal(0, 0.1, 50)
    
    sigma = np.array([1.0, 1.0, 1.0])
    
    print("\n1. Testing Dense GP:")
    config_dense = create_config(use_sparse=False)
    
    gp_dense = GP(config_dense, sigma)
    print(f"   Model type: {gp_dense.model_type}")
    print(f"   Use sparse: {gp_dense.use_sparse}")
    
    gp_dense.fit(X_train, y_train)
    y_pred_dense = gp_dense.predict(X_test)
    mse_dense = mean_squared_error(y_test, y_pred_dense)
    
    model_info = gp_dense.get_model_info()
    print(f"   Model info: {model_info}")
    print(f"   MSE: {mse_dense:.6f}")
    print(f"   Complexity: {gp_dense.get_model_complexity()} parameters")
    
    print("\n2. Testing Sparse GP:")
    config_sparse = create_config(use_sparse=True, num_inducing=50, inducing_method='random')
    
    gp_sparse = GP(config_sparse, sigma)
    print(f"   Model type: {gp_sparse.model_type}")
    print(f"   Use sparse: {gp_sparse.use_sparse}")
    
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse = gp_sparse.predict(X_test)
    mse_sparse = mean_squared_error(y_test, y_pred_sparse)
    
    model_info = gp_sparse.get_model_info()
    print(f"   Model info: {model_info}")
    print(f"   MSE: {mse_sparse:.6f}")
    print(f"   Complexity: {gp_sparse.get_model_complexity()} parameters")
    
    print(f"\n3. Comparison:")
    print(f"   MSE ratio (sparse/dense): {mse_sparse/mse_dense:.3f}")
    print(f"   Complexity ratio (sparse/dense): {gp_sparse.get_model_complexity()/gp_dense.get_model_complexity():.1f}")

def test_different_inducing_methods():
    """Test different inducing point selection methods"""
    print("\n" + "=" * 50)
    print("Testing Different Inducing Methods")
    
    np.random.seed(42)
    X_train = np.random.randn(150, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 150)
    X_test = np.random.randn(30, 2)
    y_test = np.sin(X_test[:, 0]) + np.random.normal(0, 0.1, 30)
    
    sigma = np.array([1.0, 1.0])
    
    methods = ['random', 'uniform']
    
    for method in methods:
        print(f"\nTesting {method} inducing method:")
        
        config = create_config(use_sparse=True, num_inducing=30, inducing_method=method)
        gp = GP(config, sigma)
        
        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        model_info = gp.get_model_info()
        
        print(f"   MSE: {mse:.6f}")
        print(f"   Inducing method: {model_info['inducing_method']}")
        print(f"   Number of inducing points: {model_info['num_inducing']}")

def test_uncertainty_quantification():
    """Test uncertainty quantification for both dense and sparse GPs"""
    print("\n" + "=" * 50)
    print("Testing Uncertainty Quantification")
    
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 100)
    X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    X_test = np.column_stack([X_test, np.zeros_like(X_test)])
    
    sigma = np.array([1.0, 1.0])
    
    print("\n1. Dense GP uncertainty:")
    config_dense = create_config(use_sparse=False)
    gp_dense = GP(config_dense, sigma)
    gp_dense.fit(X_train, y_train)
    y_pred_dense, y_std_dense = gp_dense.predict(X_test, return_std=True)
    
    print(f"   Mean std: {np.mean(y_std_dense):.4f}")
    print(f"   Std range: [{np.min(y_std_dense):.4f}, {np.max(y_std_dense):.4f}]")
    
    print("\n2. Sparse GP uncertainty:")
    config_sparse = create_config(use_sparse=True, num_inducing=25, inducing_method='random')
    gp_sparse = GP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse, y_std_sparse = gp_sparse.predict(X_test, return_std=True)
    
    print(f"   Mean std: {np.mean(y_std_sparse):.4f}")
    print(f"   Std range: [{np.min(y_std_sparse):.4f}, {np.max(y_std_sparse):.4f}]")
    
    print(f"\n3. Uncertainty comparison:")
    print(f"   Mean std ratio (sparse/dense): {np.mean(y_std_sparse)/np.mean(y_std_dense):.3f}")

def test_config_loading():
    """Test loading configurations from files"""
    print("\n" + "=" * 50)
    print("Testing Configuration Loading")
    
    config = ConfigParser()
    config['KERNEL'] = {
        'type': 'RBF',
        'lmbda': '0.1',
        'alpha': '1.0'
    }
    config['SPARSE'] = {
        'use_sparse': 'true',
        'num_inducing': '30',
        'inducing_method': 'uniform'
    }
    
    sigma = np.array([1.0, 1.0])
    
    gp = GP(config, sigma)
    print(f"Model type: {gp.model_type}")
    print(f"Use sparse: {gp.use_sparse}")
    
    model_info = gp.get_model_info()
    print(f"Model info: {model_info}")

if __name__ == "__main__":
    test_unified_gp_switching()
    test_different_inducing_methods()
    test_uncertainty_quantification()
    test_config_loading() 