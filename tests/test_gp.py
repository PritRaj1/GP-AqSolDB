import numpy as np
import pytest
from configparser import ConfigParser
from sklearn.metrics import mean_squared_error

from src.multivar_gp.gp import GP

def create_config(use_sparse=False, num_inducing=20, inducing_method='random'):
    config = ConfigParser()
    config['KERNEL'] = {
        'type': 'RBF',
        'lmbda': '0.1',
        'alpha': '1.0',
        'use_cache': 'true',
        'cache_size': '100',
        'num_inducing': str(num_inducing)
    }
    config['SPARSE'] = {
        'use_sparse': str(use_sparse).lower(),
        'inducing_method': inducing_method
    }
    return config

def test_unified_gp_switching():
    np.random.seed(42)
    X_train = np.random.randn(200, 3)
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, 200)
    X_test = np.random.randn(50, 3)
    y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1]) + np.random.normal(0, 0.1, 50)
    sigma = np.array([1.0, 1.0, 1.0])
    
    # Test Dense GP
    config_dense = create_config(use_sparse=False)
    gp_dense = GP(config_dense, sigma)
    
    assert gp_dense.model_type == "dense", "Model type should be 'dense'"
    assert gp_dense.use_sparse == False, "use_sparse should be False for dense GP"
    
    gp_dense.fit(X_train, y_train)
    y_pred_dense = gp_dense.predict(X_test)
    mse_dense = mean_squared_error(y_test, y_pred_dense)
    
    model_info = gp_dense.get_model_info()
    assert model_info['model_type'] == 'dense', "Model info should indicate dense GP"
    assert mse_dense > 0, "MSE should be positive"
    assert gp_dense.get_model_complexity() > 0, "Model complexity should be positive"
    
    # Test Sparse GP
    config_sparse = create_config(use_sparse=True, num_inducing=50, inducing_method='random')
    gp_sparse = GP(config_sparse, sigma)
    
    assert gp_sparse.model_type == "sparse", "Model type should be 'sparse'"
    assert gp_sparse.use_sparse == True, "use_sparse should be True for sparse GP"
    
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse = gp_sparse.predict(X_test)
    mse_sparse = mean_squared_error(y_test, y_pred_sparse)
    
    model_info = gp_sparse.get_model_info()
    assert model_info['model_type'] == 'sparse', "Model info should indicate sparse GP"
    assert mse_sparse > 0, "MSE should be positive"
    assert gp_sparse.get_model_complexity() > gp_dense.get_model_complexity(), "Sparse GP should have higher complexity"
    
    # Both models should produce reasonably similar predictions
    assert abs(mse_sparse - mse_dense) < 1.0, "MSE values should be reasonably close"

@pytest.mark.parametrize("method", ['random', 'uniform'])
def test_different_inducing_methods(method):
    np.random.seed(42)
    X_train = np.random.randn(150, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 150)
    X_test = np.random.randn(30, 2)
    y_test = np.sin(X_test[:, 0]) + np.random.normal(0, 0.1, 30)
    sigma = np.array([1.0, 1.0])
    
    config = create_config(use_sparse=True, num_inducing=30, inducing_method=method)
    gp = GP(config, sigma)
    
    gp.fit(X_train, y_train)
    y_pred = gp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    model_info = gp.get_model_info()
    
    assert mse > 0, f"MSE should be positive for {method} method"
    assert model_info['inducing_method'] == method, f"Inducing method should be {method}"
    assert model_info['num_inducing'] == 30, "Number of inducing points should match config"

def test_uncertainty_quantification():
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 100)
    X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    X_test = np.column_stack([X_test, np.zeros_like(X_test)])
    sigma = np.array([1.0, 1.0])
    
    # Dense GP with uncertainty
    config_dense = create_config(use_sparse=False)
    gp_dense = GP(config_dense, sigma)
    gp_dense.fit(X_train, y_train)
    y_pred_dense, y_std_dense = gp_dense.predict(X_test, return_std=True)
    
    assert np.all(y_std_dense > 0), "All uncertainties should be positive for dense GP"
    assert np.mean(y_std_dense) > 0, "Mean uncertainty should be positive for dense GP"
    
    # Sparse GP with uncertainty
    config_sparse = create_config(use_sparse=True, num_inducing=25, inducing_method='random')
    gp_sparse = GP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse, y_std_sparse = gp_sparse.predict(X_test, return_std=True)
    
    assert np.all(y_std_sparse > 0), "All uncertainties should be positive for sparse GP"
    assert np.mean(y_std_sparse) > 0, "Mean uncertainty should be positive for sparse GP"
    
    # Both should produce reasonable uncertainty estimates
    assert np.all(np.isfinite(y_std_dense)), "Dense GP uncertainties should be finite"
    assert np.all(np.isfinite(y_std_sparse)), "Sparse GP uncertainties should be finite"

def test_config_loading():
    config = ConfigParser()
    config['KERNEL'] = {
        'type': 'RBF',
        'lmbda': '0.1',
        'alpha': '1.0',
        'num_inducing': '30'
    }
    config['SPARSE'] = {
        'use_sparse': 'true',
        'inducing_method': 'uniform'
    }
    
    sigma = np.array([1.0, 1.0])
    
    gp = GP(config, sigma)
    assert gp.model_type == "sparse", "Model type should be 'sparse' for use_sparse=true"
    assert gp.use_sparse == True, "use_sparse should be True"
    
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    gp.fit(X_train, y_train)
    
    model_info = gp.get_model_info()
    assert model_info['model_type'] == 'sparse', "Model info should indicate sparse GP"
    assert model_info['inducing_method'] == 'uniform', "Inducing method should be uniform"

def test_model_complexity():
    sigma = np.array([1.0, 1.0, 1.0])
    
    # Dense GP complexity
    config_dense = create_config(use_sparse=False)
    gp_dense = GP(config_dense, sigma)
    dense_complexity = gp_dense.get_model_complexity()
    
    # Sparse GP complexity
    config_sparse = create_config(use_sparse=True, num_inducing=20, inducing_method='random')
    gp_sparse = GP(config_sparse, sigma)
    sparse_complexity = gp_sparse.get_model_complexity()
    
    assert dense_complexity > 0, "Dense GP complexity should be positive"
    assert sparse_complexity > 0, "Sparse GP complexity should be positive"
    assert sparse_complexity > dense_complexity, "Sparse GP should have higher complexity than dense GP"

def test_prediction_without_fitting():
    config = create_config(use_sparse=False)
    sigma = np.array([1.0, 1.0])
    gp = GP(config, sigma)
    
    X_test = np.random.randn(10, 2)
    
    with pytest.raises(Exception):
        gp.predict(X_test)

def test_fit_predict_cycle():
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    X_test = np.random.randn(10, 2)
    sigma = np.array([1.0, 1.0])
    
    # Test dense GP
    config_dense = create_config(use_sparse=False)
    gp_dense = GP(config_dense, sigma)
    gp_dense.fit(X_train, y_train)
    y_pred_dense = gp_dense.predict(X_test)
    
    assert len(y_pred_dense) == len(X_test), "Prediction length should match test data length"
    assert np.all(np.isfinite(y_pred_dense)), "Predictions should be finite"
    
    # Test sparse GP
    config_sparse = create_config(use_sparse=True, num_inducing=10, inducing_method='random')
    gp_sparse = GP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse = gp_sparse.predict(X_test)
    
    assert len(y_pred_sparse) == len(X_test), "Prediction length should match test data length"
    assert np.all(np.isfinite(y_pred_sparse)), "Predictions should be finite"

def test_cache_integration():
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    X_test = np.random.randn(20, 2)
    sigma = np.array([1.0, 1.0])
    
    # Test dense GP with cache
    config = create_config(use_sparse=False)
    gp = GP(config, sigma)
    gp.fit(X_train, y_train)
    
    # Multiple predictions to test cache
    for i in range(5):
        y_pred = gp.predict(X_test)
    
    cache_stats = gp.get_cache_stats()
    assert cache_stats is not None, "Cache stats should be available"
    
    gp.clear_cache()

if __name__ == "__main__":
    print("\nRunning GP tests...")
    
    test_unified_gp_switching()
    test_different_inducing_methods('random')
    test_different_inducing_methods('uniform')
    test_uncertainty_quantification()
    test_config_loading()
    test_model_complexity()
    test_prediction_without_fitting()
    test_fit_predict_cycle()
    test_cache_integration()
    
    print("All GP tests completed!") 