import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core.kernels import compute_kernel


@pytest.fixture
def sample_kernel_data():
    np.random.seed(42)
    X1 = np.random.randn(100, 3)
    X2 = np.random.randn(50, 3)
    sigma = np.random.uniform(0.1, 2.0, 3)
    return X1, X2, sigma


def test_output_is_jax_array(sample_kernel_data):
    X1, X2, sigma = sample_kernel_data
    result = compute_kernel("RBF", X1, X2, sigma)
    assert isinstance(result, jax.Array)


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ", "MATERN"])
def test_numpy_input_acceptance(kernel_type, sample_kernel_data):
    X1, X2, sigma = sample_kernel_data
    alpha = 2.0 if kernel_type == "RQ" else 1.5 if kernel_type == "MATERN" else None
    result = compute_kernel(kernel_type, X1, X2, sigma, alpha=alpha)
    assert result.shape == (100, 50)
    assert not jnp.any(jnp.isnan(result))


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ", "MATERN"])
def test_jax_input_acceptance(kernel_type, sample_kernel_data):
    X1, X2, sigma = sample_kernel_data
    X1_jax = jnp.asarray(X1)
    X2_jax = jnp.asarray(X2)
    sigma_jax = jnp.asarray(sigma)
    alpha = 2.0 if kernel_type == "RQ" else 1.5 if kernel_type == "MATERN" else None
    result = compute_kernel(kernel_type, X1_jax, X2_jax, sigma_jax, alpha=alpha)
    assert result.shape == (100, 50)
    assert not jnp.any(jnp.isnan(result))


def test_kernel_consistency_across_types(sample_kernel_data):
    """All kernel types should produce PSD matrices."""
    X1, _, sigma = sample_kernel_data
    X = X1[:20]
    for kernel_type, alpha in [("RBF", None), ("RQ", 2.0), ("MATERN", 1.5)]:
        K = np.asarray(compute_kernel(kernel_type, X, X, sigma[:3], alpha=alpha))
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-8), f"{kernel_type} kernel not PSD"


def test_gpu_device_placement():
    """If a GPU is available, verify the result lives on GPU."""
    devices = jax.devices("gpu") if jax.devices("gpu") else []
    if not devices:
        pytest.skip("No GPU available")

    X1 = jnp.ones((10, 3))
    X2 = jnp.ones((5, 3))
    sigma = jnp.ones(3)
    result = compute_kernel("RBF", X1, X2, sigma)
    assert result.devices().pop().platform == "gpu"
