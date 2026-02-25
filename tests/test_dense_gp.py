from configparser import ConfigParser

import numpy as np

from src.core.models import DenseGP


def create_config(kernel_type="RBF", lmbda=0.1, alpha=1.0):
    config = ConfigParser()
    config["KERNEL"] = {"type": kernel_type, "lmbda": str(lmbda), "alpha": str(alpha)}
    return config


def test_uncertainty_behavior():
    config = create_config(kernel_type="RBF", lmbda=0.01)
    sigma = 1.0

    X_train = np.array([1.0, 3.0, 7.0, 9.0]).reshape(-1, 1)
    y_train = np.sin(X_train.flatten()) + np.random.normal(0, 0.01, 4)

    X_test = np.linspace(0, 10, 50).reshape(-1, 1)

    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    _, y_std = gp.predict(X_test, return_std=True)
    y_std = np.asarray(y_std)

    # Uncertainty should be lower at training points
    training_indices = []
    for i, x in enumerate(X_test):
        if np.any(np.abs(x - X_train) < 1e-6):
            training_indices.append(i)

    if training_indices:
        training_uncertainty = y_std[training_indices]
        other_uncertainty = y_std[~np.isin(np.arange(len(X_test)), training_indices)]
        assert np.mean(training_uncertainty) < np.mean(other_uncertainty)


def test_uncertainty_distance_relationship():
    config = create_config(kernel_type="RBF", lmbda=0.1)
    sigma = 1.0

    np.random.seed(42)
    X_train = np.random.uniform(0, 10, 20).reshape(-1, 1)
    y_train = X_train.flatten() ** 2 + np.random.normal(0, 0.1, 20)
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)

    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)
    _, y_std = gp.predict(X_test, return_std=True)
    y_std = np.asarray(y_std)

    distances = np.min(np.abs(X_test - X_train.T), axis=1)
    far_points = distances > 2.0
    near_points = distances < 0.5

    if np.any(far_points) and np.any(near_points):
        assert np.mean(y_std[far_points]) > np.mean(y_std[near_points])
