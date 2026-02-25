import os
import pickle
import tempfile
from configparser import ConfigParser

import numpy as np
import pytest

from src.core.models import DenseGP
from src.optimization import GPAutoTuner


@pytest.mark.slow
def test_optimization_small_scale():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            "[KERNEL]\ntype=RBF\nlmbda=0.1\nalpha=1.0\n\n"
            "[TUNING]\nn_trials=50\ntimeout=300\n"
        )
        temp_config_path = temp_config.name

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name

    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        best_params = tuner.optimize(n_trials=5)

        assert isinstance(best_params, dict)
        assert "kernel_type" in best_params
        assert "lmbda" in best_params
        assert "sigma_0" in best_params
        assert os.path.exists(temp_config_path)
        assert os.path.exists(temp_sigma_path)

    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


def test_save_best_parameters():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            "[KERNEL]\ntype=RBF\nlmbda=0.1\nalpha=1.0\n\n"
            "[TUNING]\nn_trials=50\ntimeout=300\n"
        )
        temp_config_path = temp_config.name

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name

    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        best_params = {
            "kernel_type": "RQ",
            "lmbda": 0.05,
            "alpha": 2.5,
            "sigma_0": 1.2,
            "sigma_1": 0.8,
            "use_sparse": False,
        }
        tuner._save_best_parameters(best_params)

        config = ConfigParser()
        config.read(temp_config_path)
        assert config["KERNEL"]["type"] == "RQ"
        assert config["KERNEL"]["lmbda"] == "0.05"
        assert config["KERNEL"]["alpha"] == "2.5"

        with open(temp_sigma_path, "rb") as f:
            sigmas = pickle.load(f)
        assert len(sigmas) == 2
        assert sigmas[0] == 1.2
        assert sigmas[1] == 0.8

    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


def test_gp_with_optimized_parameters():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            "[KERNEL]\ntype=RBF\nlmbda=0.1\nalpha=1.0\n\n"
            "[TUNING]\nn_trials=50\ntimeout=300\n"
        )
        temp_config_path = temp_config.name

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name

    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        best_params = {
            "kernel_type": "RBF",
            "lmbda": 0.1,
            "alpha": 1.0,
            "sigma_0": 1.2,
            "sigma_1": 0.8,
            "use_sparse": False,
        }
        tuner._save_best_parameters(best_params)
        config, sigmas = tuner.load_optimized_parameters()

        gp = DenseGP(config, sigmas)
        gp.fit(X_train, y_train)

        X_test = np.random.uniform(0, 5, (10, 2))
        y_pred, y_std = gp.predict(X_test, return_std=True)
        y_pred = np.asarray(y_pred)
        y_std = np.asarray(y_std)

        assert np.all(y_std >= 0)
        assert not np.any(np.isnan(y_pred))

    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)

        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)
