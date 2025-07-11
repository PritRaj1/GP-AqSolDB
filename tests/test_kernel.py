import configparser
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns

from src.multivar_gp.kernels import get_kernel

sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("paper", font_scale=1.2)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    }
)


figures_dir = os.path.join(os.path.dirname(__file__), "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Created test figures directory: {figures_dir}")


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_vectorized_kernel_shape(kernel_type):
    config = configparser.ConfigParser()
    config.read("config/test.ini")
    config.set("KERNEL", "type", kernel_type)
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))

    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])

    # Test with single points (population of one)
    result = kernel(X1, X2)
    assert result.shape == (1, 1)
    assert isinstance(result[0, 0], (int, float, np.number))
    assert result[0, 0] >= 0 and result[0, 0] <= 1

    # Test with multiple points
    X1_multi = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    X2_multi = np.array([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]])

    results = kernel(X1_multi, X2_multi)
    assert results.shape == (2, 2)
    assert np.all(results >= 0) and np.all(results <= 1)


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_vectorized_kernel_symmetry(kernel_type):
    config = configparser.ConfigParser()
    config.read("config/test.ini")
    config.set("KERNEL", "type", kernel_type)
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))

    # Test symmetry: k(x,y) = k(y,x)
    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])

    K_xy = kernel(X1, X2)
    K_yx = kernel(X2, X1)

    assert np.allclose(K_xy, K_yx.T, rtol=1e-10)


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_vectorized_kernel_identity(kernel_type):
    config = configparser.ConfigParser()
    config.read("config/test.ini")
    config.set("KERNEL", "type", kernel_type)
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))

    # Test identity: k(x,x) = 1
    X = np.array([[1.0, 2.0, 3.0]])
    K_xx = kernel(X, X)

    assert np.allclose(np.diag(K_xx), 1.0, rtol=1e-10)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_vectorized_kernel_sigma_scaling(sigma):
    config = configparser.ConfigParser()
    config.read("config/test.ini")
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config, np.array([sigma, sigma, sigma]))

    # Test that larger sigma gives broader kernel
    X1 = np.array([[1.0, 2.0, 3.0]])
    X2 = np.array([[0.5, 1.5, 2.5]])

    k_value = kernel(X1, X2)[0, 0]

    # For RBF, larger sigma should give higher values for same distance
    if sigma == 0.5:
        assert k_value < 0.5  # Should be small for small sigma
    elif sigma == 2.0:
        assert k_value > 0.5  # Should be larger for large sigma


def test_vectorized_kernel_matrix_shape():
    config = configparser.ConfigParser()
    config.read("config/test.ini")
    config.set("KERNEL", "type", "RBF")
    kernel = get_kernel(config, np.array([1.0, 1.0, 1.0]))

    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ]
    )
    n = len(X)

    # Kernel matrix using vectorized computation
    K = kernel(X, X)

    assert K.shape == (n, n)
    assert K.shape == (5, 5)

    # Test symmetry
    assert np.allclose(K, K.T)

    # Test diagonal elements are 1
    assert np.allclose(np.diag(K), 1.0)


@pytest.mark.parametrize("sigma_type", ["scalar", "array"])
def test_vectorized_multivariate_kernel(sigma_type):
    config = configparser.ConfigParser()
    config.read("config/test.ini")

    config.set("KERNEL", "type", "RBF")

    if sigma_type == "scalar":
        sigma = np.array([1.0, 1.0, 1.0])
        X1 = np.array([[1.0, 2.0, 3.0]])
        X2 = np.array([[0.5, 1.5, 2.5]])
    else:  # array
        sigma = np.array([1.0, 0.5, 2.0])
        X1 = np.array([[1.0, 2.0, 3.0]])
        X2 = np.array([[0.5, 1.5, 2.5]])

    kernel = get_kernel(config, sigma)

    result = kernel(X1, X2)
    assert result.shape == (1, 1)
    assert isinstance(result[0, 0], (int, float, np.number))
    assert result[0, 0] >= 0 and result[0, 0] <= 1

    # Test identity for multivariate
    K_xx = kernel(X1, X1)
    assert np.allclose(K_xx[0, 0], 1.0, rtol=1e-10)


def test_invalid_kernel_type():
    config = configparser.ConfigParser()
    config.read("config/test.ini")
    config.set("KERNEL", "type", "INVALID_KERNEL")

    with pytest.raises(ValueError, match="Unknown kernel type"):
        get_kernel(config, np.array([1.0, 1.0, 1.0]))


def test_visual():
    config = configparser.ConfigParser()
    config.read("config/test.ini")

    x = np.linspace(-5, 5, 100).reshape(-1, 1)  # Reshape for vectorized kernels
    x0 = np.array([[0]])  # Reference point (population of one)

    kernel_types = ["RBF", "RQ"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    colors = sns.color_palette("husl", 2)

    for i, kernel_type in enumerate(kernel_types):
        config.set("KERNEL", "type", kernel_type)

        kernel = get_kernel(config, np.array([1.0]))

        k_values = np.array([kernel(x_val.reshape(1, 1), x0)[0, 0] for x_val in x])

        if kernel_type == "RBF":
            label = (
                r"$k_{\text{RBF}}(x, x_0) = "
                + r"\exp\left(-\frac{(x-x_0)^2}{2\sigma^2}\right)$"
            )
        else:
            label = (
                r"$k_{\text{RQ}}(x, x_0) = "
                + r"\left(1 + \frac{(x-x_0)^2}{2\alpha\sigma^2}\right)^{-\alpha}$"
            )

        sns.lineplot(
            x=x.flatten(),
            y=k_values,
            color=colors[i],
            linewidth=2.5,
            label=label,
            ax=axes[i],
        )
        axes[i].axvline(
            x=0,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=r"$x_0 = 0$",
        )
        axes[i].set_xlabel(r"$x - x_0$", fontweight="bold")
        axes[i].set_ylabel(r"$k(x, x_0)$", fontweight="bold")
        axes[i].set_title(
            f"{kernel_type} Kernel (Vectorized)", fontweight="bold", pad=15
        )
        axes[i].legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1.1)

        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.savefig(
        os.path.join(figures_dir, "kernels.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


if __name__ == "__main__":
    print("\nRunning vectorized kernel tests...")

    test_vectorized_kernel_shape("RBF")
    test_vectorized_kernel_shape("RQ")

    test_vectorized_kernel_symmetry("RBF")
    test_vectorized_kernel_symmetry("RQ")
    test_vectorized_kernel_identity("RBF")
    test_vectorized_kernel_identity("RQ")

    test_vectorized_kernel_sigma_scaling(0.5)
    test_vectorized_kernel_sigma_scaling(1.0)
    test_vectorized_kernel_sigma_scaling(2.0)
    test_vectorized_kernel_matrix_shape()
    test_vectorized_multivariate_kernel("scalar")
    test_vectorized_multivariate_kernel("array")
    test_invalid_kernel_type()

    print("All vectorized kernel tests passed!")

    print(f"Creating kernel visualizations in {figures_dir}...")
    test_visual()
    print("Kernel visualizations saved.")
