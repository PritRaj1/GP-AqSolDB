from configparser import ConfigParser

import numpy as np
from sklearn.metrics import mean_squared_error

from src.core.models import GP


def create_config(use_sparse=False, num_inducing=20, inducing_method="random"):
    config = ConfigParser()
    config["KERNEL"] = {
        "type": "RBF",
        "lmbda": "0.1",
        "alpha": "1.0",
        "num_inducing": str(num_inducing),
    }
    config["SPARSE"] = {
        "use_sparse": str(use_sparse).lower(),
        "inducing_method": inducing_method,
    }
    return config


def test_unified_gp_switching():
    np.random.seed(42)
    X_train = np.random.randn(200, 3)
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + np.random.normal(
        0, 0.1, 200
    )
    X_test = np.random.randn(50, 3)
    y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1]) + np.random.normal(0, 0.1, 50)
    sigma = np.array([1.0, 1.0, 1.0])

    # Dense GP
    config_dense = create_config(use_sparse=False)
    gp_dense = GP(config_dense, sigma)
    gp_dense.fit(X_train, y_train)
    y_pred_dense = gp_dense.predict(X_test)
    mse_dense = mean_squared_error(y_test, y_pred_dense)

    # Sparse GP
    config_sparse = create_config(
        use_sparse=True, num_inducing=50, inducing_method="random"
    )
    gp_sparse = GP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse = gp_sparse.predict(X_test)
    mse_sparse = mean_squared_error(y_test, y_pred_sparse)

    # Both models should produce reasonably similar predictions
    assert abs(mse_sparse - mse_dense) < 1.0


def test_different_inducing_methods():
    np.random.seed(42)
    X_train = np.random.randn(150, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 150)
    X_test = np.random.randn(30, 2)
    y_test = np.sin(X_test[:, 0]) + np.random.normal(0, 0.1, 30)
    sigma = np.array([1.0, 1.0])

    results = {}
    for method in ["random", "uniform"]:
        config = create_config(use_sparse=True, num_inducing=30, inducing_method=method)
        gp = GP(config, sigma)
        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test)
        results[method] = mean_squared_error(y_test, y_pred)

    # Both methods should produce reasonable predictions
    for method, mse in results.items():
        assert mse > 0 and mse < 5.0, f"{method} MSE={mse} out of range"


def test_uncertainty_quantification():
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 100)
    X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    X_test = np.column_stack([X_test, np.zeros_like(X_test)])
    sigma = np.array([1.0, 1.0])

    # Dense GP
    config_dense = create_config(use_sparse=False)
    gp_dense = GP(config_dense, sigma)
    gp_dense.fit(X_train, y_train)
    _, y_std_dense = gp_dense.predict(X_test, return_std=True)

    assert np.all(y_std_dense > 0)
    assert np.all(np.isfinite(y_std_dense))

    # Sparse GP
    config_sparse = create_config(
        use_sparse=True, num_inducing=25, inducing_method="random"
    )
    gp_sparse = GP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    _, y_std_sparse = gp_sparse.predict(X_test, return_std=True)

    assert np.all(y_std_sparse > 0)
    assert np.all(np.isfinite(y_std_sparse))
