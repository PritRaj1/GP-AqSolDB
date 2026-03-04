import time

import jax
import jax.numpy as jnp
import pytest

from src.core.models.gp_kan import DenseGPLayer, NormalDist
from src.utils.config_utils import create_default_config


@pytest.fixture
def sample_gp_config():
    config = create_default_config()
    config["GP"]["num_inducing_points"] = "5"
    config["GP"]["z_init_low"] = "-2.0"
    config["GP"]["z_init_high"] = "2.0"
    config["GP"]["h_init_low"] = "-1.0"
    config["GP"]["h_init_high"] = "1.0"
    config["GP"]["global_length_scale"] = "0.4"
    config["GP"]["min_length_scale"] = "0.2"
    config["GP"]["global_covariance_scale"] = "1.0"
    config["GP"]["min_covariance_scale"] = "0.1"
    config["GP"]["global_jitter"] = "0.001"
    config["GP"]["baseline_jitter"] = "0.01"
    return config


@pytest.fixture
def sample_layer(sample_gp_config):
    return DenseGPLayer(
        input_size=3, output_size=2, config=sample_gp_config, key=jax.random.PRNGKey(42)
    )


@pytest.fixture
def sample_input_dist():
    return NormalDist(jnp.array([[0.0, 1.0, -0.5]]), jnp.array([[0.1, 0.2, 0.15]]))


def test_jit_forward(sample_layer, sample_input_dist):
    jitted_forward = jax.jit(sample_layer.forward)

    # First call (compilation)
    start_time = time.time()
    output1 = jitted_forward(sample_input_dist)
    compilation_time = time.time() - start_time

    # Second call (execution)
    start_time = time.time()
    output2 = jitted_forward(sample_input_dist)
    execution_time = time.time() - start_time

    assert jnp.allclose(output1.mean, output2.mean)
    assert jnp.allclose(output1.var, output2.var)
    assert execution_time < compilation_time


def test_jit_loglikelihood(sample_layer):
    jitted_loglikelihood = jax.jit(sample_layer.loglikelihood)

    start_time = time.time()
    ll1 = jitted_loglikelihood()
    compilation_time = time.time() - start_time

    start_time = time.time()
    ll2 = jitted_loglikelihood()
    execution_time = time.time() - start_time

    assert jnp.allclose(ll1, ll2)
    assert execution_time < compilation_time


def test_gpu_vs_cpu_forward():
    cpu_config = create_default_config()
    cpu_config["DEVICE"]["use_gpu"] = "false"
    cpu_config["DEVICE"]["device"] = "cpu"

    cpu_layer = DenseGPLayer(
        input_size=3, output_size=2, config=cpu_config, key=jax.random.PRNGKey(42)
    )

    input_dist = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]), jnp.array([[0.1, 0.2, 0.15]])
    )
    cpu_output = cpu_layer.forward(input_dist)

    gpu_config = create_default_config()
    gpu_config["DEVICE"]["use_gpu"] = "true"
    gpu_config["DEVICE"]["device"] = "gpu"

    try:
        gpu_layer = DenseGPLayer(
            input_size=3, output_size=2, config=gpu_config, key=jax.random.PRNGKey(42)
        )
        gpu_output = gpu_layer.forward(input_dist)

        assert jnp.allclose(cpu_output.mean, gpu_output.mean, rtol=1e-5, atol=1e-8)
        assert jnp.allclose(cpu_output.var, gpu_output.var, rtol=1e-5, atol=1e-8)

    except Exception as e:
        if "gpu" in str(e).lower() or "device" in str(e).lower():
            pytest.skip("GPU not available")
        raise


def test_gpu_vs_cpu_loglikelihood():
    cpu_config = create_default_config()
    cpu_config["DEVICE"]["use_gpu"] = "false"
    cpu_config["DEVICE"]["device"] = "cpu"

    cpu_layer = DenseGPLayer(
        input_size=3, output_size=2, config=cpu_config, key=jax.random.PRNGKey(42)
    )
    cpu_ll = cpu_layer.loglikelihood()

    gpu_config = create_default_config()
    gpu_config["DEVICE"]["use_gpu"] = "true"
    gpu_config["DEVICE"]["device"] = "gpu"

    try:
        gpu_layer = DenseGPLayer(
            input_size=3, output_size=2, config=gpu_config, key=jax.random.PRNGKey(42)
        )
        gpu_ll = gpu_layer.loglikelihood()

        assert jnp.allclose(cpu_ll, gpu_ll, rtol=1e-5, atol=1e-8)

    except Exception as e:
        if "gpu" in str(e).lower() or "device" in str(e).lower():
            pytest.skip("GPU not available")
        raise


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_batch_perf(batch_size):
    input_mean = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3))
    input_var = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3))) + 0.1
    input_dist = NormalDist(input_mean, input_var)

    cpu_config = create_default_config()
    cpu_config["DEVICE"]["use_gpu"] = "false"
    cpu_layer = DenseGPLayer(
        input_size=3, output_size=2, config=cpu_config, key=jax.random.PRNGKey(42)
    )

    jitted_cpu_forward = jax.jit(cpu_layer.forward)
    _ = jitted_cpu_forward(input_dist)  # Warmup
    cpu_output = jitted_cpu_forward(input_dist)

    gpu_config = create_default_config()
    gpu_config["DEVICE"]["use_gpu"] = "true"

    try:
        gpu_layer = DenseGPLayer(
            input_size=3, output_size=2, config=gpu_config, key=jax.random.PRNGKey(42)
        )

        jitted_gpu_forward = jax.jit(gpu_layer.forward)
        _ = jitted_gpu_forward(input_dist)  # Warmup
        gpu_output = jitted_gpu_forward(input_dist)

        assert jnp.allclose(cpu_output.mean, gpu_output.mean, rtol=1e-5, atol=1e-8)
        assert jnp.allclose(cpu_output.var, gpu_output.var, rtol=1e-5, atol=1e-8)

    except Exception as e:
        if "gpu" not in str(e).lower() and "device" not in str(e).lower():
            raise


def test_xla_opt():
    config = create_default_config()
    layer = DenseGPLayer(
        input_size=3, output_size=2, config=config, key=jax.random.PRNGKey(42)
    )

    input_dist = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]), jnp.array([[0.1, 0.2, 0.15]])
    )

    # Without JIT
    start_time = time.time()
    for _ in range(10):
        output1 = layer.forward(input_dist)
    no_jit_time = time.time() - start_time

    # With JIT
    jitted_forward = jax.jit(layer.forward)
    _ = jitted_forward(input_dist)  # Warmup

    start_time = time.time()
    for _ in range(10):
        output2 = jitted_forward(input_dist)
    jit_time = time.time() - start_time

    assert jnp.allclose(output1.mean, output2.mean, rtol=1e-2, atol=1e-2)
    assert jnp.allclose(output1.var, output2.var, rtol=1e-2, atol=1e-2)
    assert jit_time < no_jit_time
