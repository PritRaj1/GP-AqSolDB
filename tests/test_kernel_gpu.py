import numpy as np
import pytest

from src.core.kernels import compute_kernel
from src.utils.kernel_utils import configure_parallel_settings


@pytest.fixture
def sample_kernel_data():
    np.random.seed(42)
    X1 = np.random.randn(500, 3)
    X2 = np.random.randn(250, 3)
    sigma = np.random.uniform(0.1, 2.0, 3)
    return X1, X2, sigma


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_sequential_vs_parallel(kernel_type, sample_kernel_data):
    X1, X2, sigma = sample_kernel_data
    alpha = 2.0 if kernel_type == "RQ" else None

    configure_parallel_settings(use_parallel=False, use_gpu=False)
    result_seq = compute_kernel(kernel_type, X1, X2, sigma, alpha=alpha)

    configure_parallel_settings(use_parallel=True, use_gpu=False, n_jobs=2)
    result_par = compute_kernel(kernel_type, X1, X2, sigma, alpha=alpha)

    assert np.allclose(result_seq, result_par, rtol=1e-10)
    assert result_seq.shape == result_par.shape


def test_gpu_vs_cpu():
    np.random.seed(42)
    X1 = np.random.randn(300, 3)
    X2 = np.random.randn(150, 3)
    sigma = np.random.uniform(0.1, 2.0, 3)

    configure_parallel_settings(use_parallel=False, use_gpu=False)
    result_cpu = compute_kernel("RBF", X1, X2, sigma)

    configure_parallel_settings(use_parallel=True, use_gpu=True, n_jobs=2)
    try:
        result_gpu = compute_kernel("RBF", X1, X2, sigma)
        assert np.allclose(result_cpu, result_gpu, rtol=1e-5, atol=1e-8)

    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ["gpu", "cuda", "device", "memory"]):
            pytest.skip(f"GPU not available: {e}")
        raise


@pytest.mark.parametrize("matrix_size", [100, 500, 1000])
def test_parallel_performance_scaling(matrix_size):
    np.random.seed(42)
    X1 = np.random.randn(matrix_size, 5)
    X2 = np.random.randn(matrix_size // 2, 5)
    sigma = np.random.uniform(0.1, 2.0, 5)

    configure_parallel_settings(use_parallel=False, use_gpu=False)
    result_seq = compute_kernel("RBF", X1, X2, sigma)

    configure_parallel_settings(use_parallel=True, use_gpu=False, n_jobs=2)
    result_par = compute_kernel("RBF", X1, X2, sigma)

    assert result_seq.shape == result_par.shape
    assert np.allclose(result_seq, result_par, rtol=1e-10)
