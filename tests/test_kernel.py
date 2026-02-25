import numpy as np
import pytest

from src.core.kernels import compute_kernel


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_kernel_symmetry(kernel_type):
    sigma = np.array([1.0, 1.0, 1.0])
    alpha = 1.0 if kernel_type == "RQ" else None

    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])

    K_xy = compute_kernel(kernel_type, X1, X2, sigma, alpha=alpha)
    K_yx = compute_kernel(kernel_type, X2, X1, sigma, alpha=alpha)

    assert np.allclose(K_xy, K_yx.T, rtol=1e-10)


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_kernel_identity(kernel_type):
    sigma = np.array([1.0, 1.0, 1.0])
    alpha = 1.0 if kernel_type == "RQ" else None

    X = np.array([[1.0, 2.0, 3.0]])
    K_xx = compute_kernel(kernel_type, X, X, sigma, alpha=alpha)

    assert np.allclose(np.diag(K_xx), 1.0, rtol=1e-10)


@pytest.mark.parametrize("sigma_val", [0.5, 1.0, 2.0])
def test_kernel_sigma_scaling(sigma_val):
    sigma = np.array([sigma_val, sigma_val, sigma_val])

    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])

    k_value = compute_kernel("RBF", X1, X2, sigma)[0, 0]

    if sigma_val == 0.5:
        assert k_value < 0.5

    elif sigma_val == 2.0:
        assert k_value > 0.5


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ", "MATERN"])
def test_kernel_positive_definite(kernel_type):
    np.random.seed(42)
    sigma = np.array([1.0, 1.0])
    X = np.random.randn(10, 2)
    alpha = 1.0 if kernel_type == "RQ" else 1.5 if kernel_type == "MATERN" else None

    K = compute_kernel(kernel_type, X, X, sigma, alpha=alpha)
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues > -1e-10), "Kernel should be PSD"
