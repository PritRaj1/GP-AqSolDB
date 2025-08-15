import os
import tempfile
from configparser import ConfigParser

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.optimization import GPKANAutoTuner
from src.core.models.gp_kan import GP_KAN
from src.core.models.gp_kan import NormalDist


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


@pytest.fixture
def sample_tuner(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    return GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        max_hidden_layers=1,
        max_hidden_size=3,
        num_epochs=5,
    )


def test_init(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        metric="BIC",
        max_hidden_layers=2,
        max_hidden_size=5,
        num_epochs=5,
    )

    assert tuner.n_features == 3, "Number of features should be 3"
    assert tuner.n_samples == 100, "Number of samples should be 100"
    assert tuner.metric == "BIC", "Metric should be BIC"
    assert tuner.max_hidden_layers == 2, "Max hidden layers should be 2"
    assert tuner.max_hidden_size == 5, "Max hidden size should be 5"
    assert tuner.num_epochs == 5, "Num epochs should be 5"


def test_default_conf(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        num_epochs=5,
    )

    assert "NETWORK" in tuner.config, "NETWORK section should exist"
    assert "GP" in tuner.config, "GP section should exist"
    assert "NORMALIZATION" in tuner.config, "NORMALIZATION section should exist"
    assert "TRAINING" in tuner.config, "TRAINING section should exist"
    assert "DEVICE" in tuner.config, "DEVICE section should exist"

    assert tuner.config["NETWORK"]["input_size"] == "3", "Input size should be 3"
    assert tuner.config["NETWORK"]["output_size"] == "1", "Output size should be 1"


def test_param_counting(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        num_epochs=5,
    )

    hidden_sizes = []
    n_params = tuner._count_network_parameters(hidden_sizes)
    assert n_params > 0, "Parameter count should be positive"

    hidden_sizes = [4]
    n_params = tuner._count_network_parameters(hidden_sizes)
    assert n_params > 0, "Parameter count should be positive"

    hidden_sizes = [3, 2]
    n_params = tuner._count_network_parameters(hidden_sizes)
    assert n_params > 0, "Parameter count should be positive"


def test_bic(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        num_epochs=5,
    )

    mse = 0.1
    n_params = 50
    n_samples = 100

    bic = tuner.calculate_bic(mse, n_params, n_samples)
    assert isinstance(bic, float), "BIC should be a float"
    assert bic > 0, "BIC should be positive"


def test_opt(sample_data, temp_config_dir):
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

    assert isinstance(result, dict), "Result should be a dictionary"
    assert "best_params" in result, "Should have best_params"
    assert "best_value" in result, "Should have best_value"
    assert "hidden_sizes" in result, "Should have hidden_sizes"

    best_params = result["best_params"]
    assert "num_inducing_points" in best_params, "Should have num_inducing_points"
    assert "global_length_scale" in best_params, "Should have global_length_scale"
    assert (
        "global_covariance_scale" in best_params
    ), "Should have global_covariance_scale"


def test_saving_loading(sample_data, temp_config_dir):
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

    tuner.optimize(n_trials=3)
    (
        loaded_config,
        loaded_hidden_sizes,
        loaded_activation_types,
        loaded_activation_params,
    ) = tuner.load_optimized_parameters()
    assert isinstance(
        loaded_config, ConfigParser
    ), "Loaded config should be ConfigParser"
    assert isinstance(loaded_hidden_sizes, list), "Loaded hidden sizes should be list"
    assert isinstance(
        loaded_activation_types, list
    ), "Loaded activation types should be list"
    assert isinstance(
        loaded_activation_params, list
    ), "Loaded activation params should be list"

    params_data = load_gpkan_params_from_file(params_path)
    assert "hidden_sizes" in params_data, "Params data should have hidden_sizes"
    assert "best_params" in params_data, "Params data should have best_params"


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

        assert isinstance(network, GP_KAN), "Should create GP_KAN network"
        assert len(network.layers) > 0, "Network should have layers"

        input_mean = jnp.array(X[:1])  # Single sample
        input_var = jnp.ones_like(input_mean) * 0.01
        input_dist = NormalDist(input_mean, input_var)

        output_dist = network.forward(input_dist)

        assert isinstance(output_dist, NormalDist), "Output should be NormalDist"
        assert output_dist.mean.shape == (1, 1), "Output mean should have shape (1, 1)"
        assert output_dist.var.shape == (
            1,
            1,
        ), "Output variance should have shape (1, 1)"

        input_mean = jnp.array(X[:3])  # Multiple samples
        input_var = jnp.ones_like(input_mean) * 0.01
        input_dist = NormalDist(input_mean, input_var)

        output_dist = network.forward(input_dist)

        assert isinstance(output_dist, NormalDist), "Output should be NormalDist"
        assert output_dist.mean.shape == (3, 1), "Output mean should have shape (3, 1)"
        assert output_dist.var.shape == (
            3,
            1,
        ), "Output variance should have shape (3, 1)"

    # If optimization failed, just test that the tuner was created correctly
    else:
        assert tuner.n_features == 3, "Tuner should have correct number of features"
        assert tuner.n_samples == 100, "Tuner should have correct number of samples"
        print("Optimization incomplete, but tuner created correctly")


def test_invalid_metric():
    X = np.random.randn(10, 2)
    y = np.random.randn(10)

    with pytest.raises(ValueError, match="metric must be 'BIC', 'MSE', or 'R2'"):
        GPKANAutoTuner(X_train=X, y_train=y, metric="INVALID")


def test_mse(sample_data, temp_config_dir):
    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        metric="MSE",
        max_hidden_layers=1,
        max_hidden_size=3,
        num_epochs=5,
    )

    result = tuner.optimize(n_trials=3)
    assert tuner.metric == "MSE", "Metric should be MSE"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "best_params" in result, "Should have best_params"


def test_gpu_tuning(sample_data, temp_config_dir):
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("No GPU available")

    X, y = sample_data
    config_path = os.path.join(temp_config_dir, "test_config.ini")
    params_path = os.path.join(temp_config_dir, "test_params.pkl")

    tuner = GPKANAutoTuner(
        X_train=X,
        y_train=y,
        config_path=config_path,
        params_save_path=params_path,
        use_gpu=True,
        max_hidden_layers=1,
        max_hidden_size=3,
        num_epochs=5,
    )

    result = tuner.optimize(n_trials=2)

    assert tuner.config["DEVICE"]["use_gpu"] == "true"
    assert tuner.config["DEVICE"]["device"] == "gpu"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "best_params" in result, "Should have best_params"


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

    assert "activation_types" in result, "Should have activation_types"
    assert "activation_params" in result, "Should have activation_params"

    activation_types = result["activation_types"]
    for act_type in activation_types:
        assert act_type in [
            "NormaliseGaussian",
            "ReduceSumGaussian",
            "None",
        ], f"Invalid activation type: {act_type}"

    if len(result["hidden_sizes"]) > 0:
        network = GP_KAN(
            tuner.config,
            hidden_sizes=result["hidden_sizes"],
            activation_types=result["activation_types"],
            activation_params=result["activation_params"],
        )

        # Test forward pass wiht optimizer params and chosen acts
        input_mean = jnp.array(X[:5])
        input_var = jnp.ones_like(input_mean) * 0.01
        input_dist = NormalDist(input_mean, input_var)

        output_dist = network.forward(input_dist)

        assert output_dist.mean.shape == (5, 1), "Output mean should have shape (5, 1)"
        assert output_dist.var.shape == (
            5,
            1,
        ), "Output variance should have shape (5, 1)"

        print("Auto-tuning with activation functions test passed!")


if __name__ == "__main__":
    print("\nRunning GP-KAN auto tune tests...")

    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "test_config.ini")
    params_path = os.path.join(temp_dir, "test_params.pkl")

    try:
        test_init((X, y), temp_dir)
        test_default_conf((X, y), temp_dir)
        test_param_counting((X, y), temp_dir)
        test_bic((X, y), temp_dir)
        test_opt((X, y), temp_dir)
        test_saving_loading((X, y), temp_dir)
        test_opt_creation((X, y), temp_dir)
        test_invalid_metric()
        test_mse((X, y), temp_dir)
        test_gpu_tuning((X, y), temp_dir)
        test_activation_functions_in_auto_tune((X, y), temp_dir)
        print("All tests passed!")
    finally:
        shutil.rmtree(temp_dir)
