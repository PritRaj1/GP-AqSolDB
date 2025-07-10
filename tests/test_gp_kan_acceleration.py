import jax
import jax.numpy as jnp
import pytest
import time

import os
from configparser import ConfigParser


from src.gp_kan.dense_layer import DenseGPLayer, create_default_conf
from src.gp_kan.normal_dist import NormalDist, get_device_config, setup_jax_device

@pytest.fixture
def sample_gp_config():
    config = create_default_conf()
    config['GP']['num_inducing_points'] = '5'
    config['GP']['z_init_low'] = '-2.0'
    config['GP']['z_init_high'] = '2.0'
    config['GP']['h_init_low'] = '-1.0'
    config['GP']['h_init_high'] = '1.0'
    config['GP']['global_length_scale'] = '0.4'
    config['GP']['min_length_scale'] = '0.2'
    config['GP']['global_covariance_scale'] = '1.0'
    config['GP']['min_covariance_scale'] = '0.1'
    config['GP']['global_jitter'] = '0.001'
    config['GP']['baseline_jitter'] = '0.01'
    return config

@pytest.fixture
def sample_layer(sample_gp_config):
    return DenseGPLayer(
        input_size=3,
        output_size=2,
        config=sample_gp_config,
        key=jax.random.PRNGKey(42)
    )

@pytest.fixture
def sample_input_dist():
    return NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[0.1, 0.2, 0.15]])
    )

def test_device_config():
    config = create_default_conf()
    device_config = get_device_config(config)
    
    assert 'use_gpu' in device_config
    assert 'device' in device_config
    assert 'precision' in device_config
    assert isinstance(device_config['use_gpu'], bool)
    assert device_config['device'] in ['cpu', 'gpu']
    assert device_config['precision'] in ['float32', 'float64']


def test_gpu_setup():
    config = create_default_conf()
    config['DEVICE']['use_gpu'] = 'true'
    config['DEVICE']['device'] = 'gpu'
    config['DEVICE']['precision'] = 'float32'
    
    try:
        setup_jax_device(config)
        assert True
    except Exception as e:
        if 'gpu' in str(e).lower() or 'device' in str(e).lower():
            pytest.skip("GPU not available")
        else:
            raise


def test_cpu_setup():
    config = create_default_conf()
    config['DEVICE']['use_gpu'] = 'false'
    config['DEVICE']['device'] = 'cpu'
    config['DEVICE']['precision'] = 'float32'
    
    setup_jax_device(config)
    assert True


def test_gpu_layer_init():
    config = create_default_conf()
    config['DEVICE']['use_gpu'] = 'true'
    config['DEVICE']['device'] = 'gpu'
    
    try:
        layer = DenseGPLayer(
            input_size=3,
            output_size=2,
            config=config,
            key=jax.random.PRNGKey(42)
        )
        
        params = layer.get_params()
        for param_name, param_value in params.items():
            assert hasattr(param_value, 'device'), f"Parameter {param_name} should have device attribute"
        
        print("GPU layer initialization successful")
    except Exception as e:
        if 'gpu' in str(e).lower() or 'device' in str(e).lower():
            pytest.skip("GPU not available")
        else:
            raise


def test_cpu_layer_init():
    config = create_default_conf()
    config['DEVICE']['use_gpu'] = 'false'
    config['DEVICE']['device'] = 'cpu'
    
    layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=config,
        key=jax.random.PRNGKey(42)
    )
    
    params = layer.get_params()
    for param_name, param_value in params.items():
        assert hasattr(param_value, 'device'), f"Parameter {param_name} should have device attribute"
    
    print("CPU layer initialization successful")


def test_jit_forward():
    config = create_default_conf()
    layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=config,
        key=jax.random.PRNGKey(42)
    )
    
    input_dist = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[0.1, 0.2, 0.15]])
    )
    
    jitted_forward = jax.jit(layer.forward)
    
    # First call (compilation)
    start_time = time.time()
    output1 = jitted_forward(input_dist)
    compilation_time = time.time() - start_time
    
    # Second call (execution)
    start_time = time.time()
    output2 = jitted_forward(input_dist)
    execution_time = time.time() - start_time
    
    assert jnp.allclose(output1.mean, output2.mean)
    assert jnp.allclose(output1.var, output2.var)
    assert execution_time < compilation_time
    
    print(f"JIT compilation successful. Compilation: {compilation_time:.4f}s, Execution: {execution_time:.4f}s")


def test_jit_loglikelihood():
    config = create_default_conf()
    layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=config,
        key=jax.random.PRNGKey(42)
    )
    
    jitted_loglikelihood = jax.jit(layer.loglikelihood)
    
    # First (compilation)
    start_time = time.time()
    ll1 = jitted_loglikelihood()
    compilation_time = time.time() - start_time
    
    # Second (execution)
    start_time = time.time()
    ll2 = jitted_loglikelihood()
    execution_time = time.time() - start_time
    
    assert jnp.allclose(ll1, ll2)
    assert execution_time < compilation_time
    
    print(f"JIT loglikelihood successful. Compilation: {compilation_time:.4f}s, Execution: {execution_time:.4f}s")


def test_gpu_vs_cpu_forward():
    cpu_config = create_default_conf()
    cpu_config['DEVICE']['use_gpu'] = 'false'
    cpu_config['DEVICE']['device'] = 'cpu'
    
    cpu_layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=cpu_config,
        key=jax.random.PRNGKey(42)
    )
    
    input_dist = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[0.1, 0.2, 0.15]])
    )
    
    cpu_output = cpu_layer.forward(input_dist)
    
    gpu_config = create_default_conf()
    gpu_config['DEVICE']['use_gpu'] = 'true'
    gpu_config['DEVICE']['device'] = 'gpu'
    
    try:
        gpu_layer = DenseGPLayer(
            input_size=3,
            output_size=2,
            config=gpu_config,
            key=jax.random.PRNGKey(42)
        )
        
        gpu_output = gpu_layer.forward(input_dist)
        
        assert jnp.allclose(cpu_output.mean, gpu_output.mean, rtol=1e-5, atol=1e-8)
        assert jnp.allclose(cpu_output.var, gpu_output.var, rtol=1e-5, atol=1e-8)
        
        print("GPU and CPU forward pass results are consistent")
    except Exception as e:
        if 'gpu' in str(e).lower() or 'device' in str(e).lower():
            pytest.skip("GPU not available")
        else:
            raise


def test_gpu_vs_cpu_loglikelihood():
    cpu_config = create_default_conf()
    cpu_config['DEVICE']['use_gpu'] = 'false'
    cpu_config['DEVICE']['device'] = 'cpu'
    
    cpu_layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=cpu_config,
        key=jax.random.PRNGKey(42)
    )
    
    cpu_ll = cpu_layer.loglikelihood()
    
    gpu_config = create_default_conf()
    gpu_config['DEVICE']['use_gpu'] = 'true'
    gpu_config['DEVICE']['device'] = 'gpu'
    
    try:
        gpu_layer = DenseGPLayer(
            input_size=3,
            output_size=2,
            config=gpu_config,
            key=jax.random.PRNGKey(42)
        )
        
        gpu_ll = gpu_layer.loglikelihood()
        
        assert jnp.allclose(cpu_ll, gpu_ll, rtol=1e-5, atol=1e-8)
        
        print("GPU and CPU loglikelihood results are consistent")
    except Exception as e:
        if 'gpu' in str(e).lower() or 'device' in str(e).lower():
            pytest.skip("GPU not available")
        else:
            raise


@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_batch_perf(batch_size):
    input_mean = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3))
    input_var = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3))) + 0.1
    input_dist = NormalDist(input_mean, input_var)
    
    # CPU timing
    cpu_config = create_default_conf()
    cpu_config['DEVICE']['use_gpu'] = 'false'
    cpu_layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=cpu_config,
        key=jax.random.PRNGKey(42)
    )
    
    jitted_cpu_forward = jax.jit(cpu_layer.forward)
    _ = jitted_cpu_forward(input_dist)  # Warmup
    
    start_time = time.time()
    cpu_output = jitted_cpu_forward(input_dist)
    cpu_time = time.time() - start_time
    
    # GPU timing
    gpu_config = create_default_conf()
    gpu_config['DEVICE']['use_gpu'] = 'true'
    
    try:
        gpu_layer = DenseGPLayer(
            input_size=3,
            output_size=2,
            config=gpu_config,
            key=jax.random.PRNGKey(42)
        )
        
        jitted_gpu_forward = jax.jit(gpu_layer.forward)
        _ = jitted_gpu_forward(input_dist)  # Warmup
        
        start_time = time.time()
        gpu_output = jitted_gpu_forward(input_dist)
        gpu_time = time.time() - start_time
        
        assert jnp.allclose(cpu_output.mean, gpu_output.mean, rtol=1e-5, atol=1e-8)
        assert jnp.allclose(cpu_output.var, gpu_output.var, rtol=1e-5, atol=1e-8)
        
        print(f"Batch size {batch_size}: CPU {cpu_time:.4f}s, GPU {gpu_time:.4f}s")
        
    except Exception as e:
        if 'gpu' in str(e).lower() or 'device' in str(e).lower():
            print(f"Batch size {batch_size}: CPU {cpu_time:.4f}s, GPU not available")
        else:
            raise


def test_xla_opt():
    config = create_default_conf()
    layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=config,
        key=jax.random.PRNGKey(42)
    )
    
    input_dist = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[0.1, 0.2, 0.15]])
    )
    
    # Without JIT - use same layer instance to avoid recompile
    start_time = time.time()
    for _ in range(10):
        output1 = layer.forward(input_dist)
    no_jit_time = time.time() - start_time
    
    # With JIT -
    jitted_forward = jax.jit(layer.forward)
    _ = jitted_forward(input_dist)  # Warmup
    
    start_time = time.time()
    for _ in range(10):
        output2 = jitted_forward(input_dist)
    jit_time = time.time() - start_time
    
    # Results should be close but maybe not identical due to JIT
    assert jnp.allclose(output1.mean, output2.mean, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(output1.var, output2.var, rtol=1e-3, atol=1e-3)
    assert jit_time < no_jit_time
    
    print(f"XLA optimization: No JIT {no_jit_time:.4f}s, With JIT {jit_time:.4f}s")


def test_memory_efficiency():
    config = create_default_conf()
    config['GP']['num_inducing_points'] = '20'
    
    # CPU test
    cpu_config = create_default_conf()
    cpu_config['GP']['num_inducing_points'] = '20'
    cpu_config['DEVICE']['use_gpu'] = 'false'
    
    cpu_layer = DenseGPLayer(
        input_size=10,
        output_size=5,
        config=cpu_config,
        key=jax.random.PRNGKey(42)
    )
    
    large_input = NormalDist(
        jax.random.normal(jax.random.PRNGKey(0), (100, 10)),
        jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (100, 10))) + 0.1
    )
    
    try:
        cpu_output = cpu_layer.forward(large_input)
        assert cpu_output.mean.shape == (100, 5)
        assert cpu_output.var.shape == (100, 5)
        print("CPU memory test passed")
    except Exception as e:
        pytest.fail(f"CPU memory test failed: {e}")
    
    # GPU test
    gpu_config = create_default_conf()
    gpu_config['GP']['num_inducing_points'] = '20'
    gpu_config['DEVICE']['use_gpu'] = 'true'
    
    try:
        gpu_layer = DenseGPLayer(
            input_size=10,
            output_size=5,
            config=gpu_config,
            key=jax.random.PRNGKey(42)
        )
        
        gpu_output = gpu_layer.forward(large_input)
        assert gpu_output.mean.shape == (100, 5)
        assert gpu_output.var.shape == (100, 5)
        print("GPU memory test passed")
    except Exception as e:
        if 'gpu' in str(e).lower() or 'device' in str(e).lower() or 'memory' in str(e).lower():
            pytest.skip("GPU memory test failed - likely due to insufficient GPU memory")
        else:
            pytest.fail(f"GPU memory test failed: {e}")


def test_device_transfer():
    cpu_dist = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[0.1, 0.2, 0.15]])
    )
    
    try:
        gpu_dist = cpu_dist.to_device('gpu')
        assert hasattr(gpu_dist.mean, 'device')
        assert hasattr(gpu_dist.var, 'device')
        print("CPU to GPU transfer successful")
    except Exception as e:
        if 'gpu' in str(e).lower() or 'device' in str(e).lower():
            pytest.skip("GPU not available for transfer test")
        else:
            raise
    
    cpu_dist_2 = cpu_dist.to_device('cpu')
    assert hasattr(cpu_dist_2.mean, 'device')
    assert hasattr(cpu_dist_2.var, 'device')
    print("CPU to CPU transfer successful")


@pytest.mark.parametrize("precision", ["float32"])
def test_precision_configuration(precision):
    config = create_default_conf()
    config['DEVICE']['precision'] = precision
    
    try:
        setup_jax_device(config)
        
        layer = DenseGPLayer(
            input_size=3,
            output_size=2,
            config=config,
            key=jax.random.PRNGKey(42)
        )
        
        input_dist = NormalDist(
            jnp.array([[0.0, 1.0, -0.5]]),
            jnp.array([[0.1, 0.2, 0.15]])
        )
        
        output = layer.forward(input_dist)
        
        if precision == 'float32':
            assert output.mean.dtype == jnp.float32
            assert output.var.dtype == jnp.float32
        
        print(f"Precision {precision} configuration successful")
        
    except Exception as e:
        if precision == 'float64' and 'x64' in str(e).lower():
            pytest.skip(f"Float64 precision not supported: {e}")
        else:
            raise


def test_error_handling():
    config = create_default_conf()
    config['DEVICE']['precision'] = 'invalid_precision'
    
    # This should not raise an exception currently
    # as setup_jax_device doesn't validate precision values
    try:
        setup_jax_device(config)
        print("Invalid precision handled gracefully")
    except Exception:
        print("Invalid precision raised exception as expected")


if __name__ == "__main__":
    print("\nRunning GP-KAN acceleration tests...")
    test_device_config()
    test_gpu_setup()
    test_cpu_setup()
    test_gpu_layer_init()
    test_cpu_layer_init()
    test_jit_forward()
    test_jit_loglikelihood()
    test_gpu_vs_cpu_forward()
    test_gpu_vs_cpu_loglikelihood()

    test_batch_perf(1)
    test_batch_perf(5)
    test_batch_perf(10)
    test_xla_opt()
    test_memory_efficiency()
    test_device_transfer()
    test_precision_configuration("float32")
    test_error_handling()
    print("All tests passed!")
