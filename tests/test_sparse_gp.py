from configparser import ConfigParser

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error

from src.core.models import FITCGP, DenseGP


def create_config(
    kernel_type="RBF",
    lmbda=0.1,
    alpha=1.0,
    sparse=False,
    num_inducing=20,
):
    config = ConfigParser()
    config["KERNEL"] = {
        "type": kernel_type,
        "lmbda": str(lmbda),
        "alpha": str(alpha),
        "sparse": str(sparse).lower(),
        "num_inducing": str(num_inducing),
    }
    return config


def test_sparse_vs_full_gp():
    np.random.seed(42)
    X_train = np.random.randn(300, 3)
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 300
    )
    X_test = np.random.randn(100, 3)
    y_test = np.sin(X_test[:, 0]) * np.exp(X_test[:, 1] / 5) + np.random.normal(
        0, 0.1, 100
    )
    sigma = np.array([1.0, 1.0, 1.0])

    # Full GP
    config_full = create_config("RBF", lmbda=0.1, sparse=False)
    gp_full = DenseGP(config_full, sigma)
    gp_full.fit(X_train, y_train)
    y_pred_full = np.asarray(gp_full.predict(X_test))
    mse_full = mean_squared_error(y_test, y_pred_full)

    # Sparse GP
    config_sparse = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=50)
    gp_sparse = FITCGP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    y_pred_sparse = np.asarray(gp_sparse.predict(X_test))
    mse_sparse = mean_squared_error(y_test, y_pred_sparse)

    assert mse_full > 0
    assert mse_sparse > 0

    sparse_info = gp_sparse.get_sparse_info()
    assert sparse_info["compression_ratio"] < 1.0


@pytest.mark.parametrize("num_inducing", [10, 20, 50, 100])
def test_different_inducing_points(num_inducing):
    np.random.seed(42)
    X_train = np.random.randn(200, 2)
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + np.random.normal(
        0, 0.1, 200
    )
    X_test = np.random.randn(50, 2)
    y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1]) + np.random.normal(0, 0.1, 50)
    sigma = np.array([1.0, 1.0])

    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=num_inducing)
    gp = FITCGP(config, sigma)
    gp.fit(X_train, y_train)

    y_pred = np.asarray(gp.predict(X_test))
    mse = mean_squared_error(y_test, y_pred)

    assert mse > 0
    assert gp.get_sparse_info()["compression_ratio"] <= 1.0


def test_sparse_gp_with_uncertainty():
    np.random.seed(42)
    X_train = np.random.randn(150, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 150)
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_test = np.column_stack([X_test, np.zeros_like(X_test)])
    sigma = np.array([1.0, 1.0])

    # Full GP
    config_full = create_config("RBF", lmbda=0.1, sparse=False)
    gp_full = DenseGP(config_full, sigma)
    gp_full.fit(X_train, y_train)
    _, y_std_full = gp_full.predict(X_test, return_std=True)
    y_std_full = np.asarray(y_std_full)

    # Sparse GP
    config_sparse = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=30)
    gp_sparse = FITCGP(config_sparse, sigma)
    gp_sparse.fit(X_train, y_train)
    _, y_std_sparse = gp_sparse.predict(X_test, return_std=True)
    y_std_sparse = np.asarray(y_std_sparse)

    assert np.all(y_std_full > 0)
    assert np.all(y_std_sparse > 0)
    assert np.all(np.isfinite(y_std_full))
    assert np.all(np.isfinite(y_std_sparse))


@pytest.mark.parametrize(
    "method", ["random", "uniform", "kmeans", "kmeans_plus_plus", "furthest_point"]
)
def test_inducing_point_selection_methods(method):
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 100)
    X_test = np.random.randn(20, 2)
    y_test = np.sin(X_test[:, 0]) + np.random.normal(0, 0.1, 20)

    sigma = np.array([1.0, 1.0])
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=20)

    gp = FITCGP(config, sigma)
    gp.fit(X_train, y_train, inducing_method=method)

    y_pred = np.asarray(gp.predict(X_test))
    mse = mean_squared_error(y_test, y_pred)

    assert mse > 0
    assert gp.get_sparse_info()["compression_ratio"] <= 1.0


def test_adaptive_inducing_point():
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, 100)
    X_test = np.random.randn(20, 2)
    y_test = np.sin(X_test[:, 0]) + np.random.normal(0, 0.1, 20)

    sigma = np.array([1.0, 1.0])
    config = create_config("RBF", lmbda=0.1, sparse=True, num_inducing=20)

    # Stratified
    gp_stratified = FITCGP(config, sigma)
    gp_stratified.fit(X_train, y_train, inducing_method="stratified")
    y_pred = np.asarray(gp_stratified.predict(X_test))
    mse_stratified = mean_squared_error(y_test, y_pred)

    # Adaptive
    gp_adaptive = FITCGP(config, sigma)
    gp_adaptive.fit(X_train, y_train, inducing_method="adaptive")
    mse_adaptive = mean_squared_error(y_test, np.asarray(gp_adaptive.predict(X_test)))

    assert mse_stratified > 0
    assert mse_adaptive > 0
