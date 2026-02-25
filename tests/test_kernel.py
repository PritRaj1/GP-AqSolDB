import numpy as np
import pytest

from src.core.kernels import compute_kernel


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ", "TPS"])
def test_kernel_symmetry(kernel_type):
    sigma = np.array([1.0, 1.0, 1.0])
    alpha = 1.0 if kernel_type == "RQ" else None

    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])

    K_xy = np.asarray(compute_kernel(kernel_type, X1, X2, sigma, alpha=alpha))
    K_yx = np.asarray(compute_kernel(kernel_type, X2, X1, sigma, alpha=alpha))

    assert np.allclose(K_xy, K_yx.T, rtol=1e-10)


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_kernel_identity(kernel_type):
    sigma = np.array([1.0, 1.0, 1.0])
    alpha = 1.0 if kernel_type == "RQ" else None

    X = np.array([[1.0, 2.0, 3.0]])
    K_xx = np.asarray(compute_kernel(kernel_type, X, X, sigma, alpha=alpha))

    assert np.allclose(np.diag(K_xx), 1.0, rtol=1e-10)


def test_tps_self_kernel_zero():
    sigma = np.array([1.0, 1.0, 1.0])
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    K = np.asarray(compute_kernel("TPS", X, X, sigma))
    assert np.allclose(np.diag(K), 0.0, atol=1e-10), "TPS(x,x) should be 0"


@pytest.mark.parametrize("sigma_val", [0.5, 1.0, 2.0])
def test_kernel_sigma_scaling(sigma_val):
    sigma = np.array([sigma_val, sigma_val, sigma_val])

    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])

    k_value = float(np.asarray(compute_kernel("RBF", X1, X2, sigma))[0, 0])

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

    K = np.asarray(compute_kernel(kernel_type, X, X, sigma, alpha=alpha))
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues > -1e-10), "Kernel should be PSD"


def test_tps_conditionally_positive_definite():
    """TPS is CPD of order 2: c^T K c >= 0 when c perp degree-1 polys."""
    np.random.seed(42)
    sigma = np.array([1.0, 1.0])
    X = np.random.randn(10, 2)
    K = np.asarray(compute_kernel("TPS", X, X, sigma))

    # Project out the polynomial space of degree <= 1: [1, X]
    n = len(X)
    P_poly = np.column_stack([np.ones(n), X])
    Q, _ = np.linalg.qr(P_poly, mode="reduced")
    P = np.eye(n) - Q @ Q.T
    K_proj = P @ K @ P
    eigenvalues = np.linalg.eigvalsh(K_proj)
    assert np.all(eigenvalues > -1e-10), "TPS should be conditionally PSD"
