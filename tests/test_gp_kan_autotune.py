import os
import shutil
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from src.core.models.gp_kan import GP_KAN, NormalDist
from src.optimization import GPKANAutoTuner
from src.optimization.gp_kan_autotuner import create_optimized_network


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def temp_config_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.slow
def test_opt_creation(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        max_hidden_layers=1,
        max_hidden_size=3,
        num_epochs=5,
    )

    result = tuner.optimize(n_trials=2)

    if result and "best_params" in result:
        network = create_optimized_network(config_path, params_path)

        assert isinstance(network, GP_KAN)
        assert len(network.layers) > 0

        input_mean = jnp.array(X[:3])
        input_var = jnp.ones_like(input_mean) * 0.01
        input_dist = NormalDist(input_mean, input_var)

        output_dist = network.forward(input_dist)

        assert output_dist.mean.shape == (3, 1)
        assert output_dist.var.shape == (3, 1)


@pytest.mark.slow
def test_activation_functions_in_auto_tune(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        max_hidden_layers=1,
        max_hidden_size=3,
        num_epochs=3,
        available_acts=["NormaliseGaussian", "ReduceSumGaussian", "None"],
    )

    result = tuner.optimize(n_trials=2)

    assert "activation_types" in result
    assert "activation_params" in result

    if len(result["hidden_sizes"]) > 0:
        network = GP_KAN(
            tuner.config,
            hidden_sizes=result["hidden_sizes"],
            activation_types=result["activation_types"],
            activation_params=result["activation_params"],
        )

        input_mean = jnp.array(X[:5])
        input_var = jnp.ones_like(input_mean) * 0.01
        input_dist = NormalDist(input_mean, input_var)

        output_dist = network.forward(input_dist)

        assert output_dist.mean.shape == (5, 1)
        assert output_dist.var.shape == (5, 1)
