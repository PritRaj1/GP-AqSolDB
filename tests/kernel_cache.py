import time
from configparser import ConfigParser

import numpy as np
import pytest

from src.multivar_gp.dense_gp import DenseGP
from src.core.kernels import clear_kernel_cache, get_cache_stats


def create_config(
    kernel_type="RBF", lmbda=0.1, alpha=1.0, use_cache=True, cache_size=100
):
    config = ConfigParser()
    config["KERNEL"] = {
        "type": kernel_type,
        "lmbda": str(lmbda),
        "alpha": str(alpha),
        "use_cache": str(use_cache).lower(),
        "cache_size": str(cache_size),
    }
    return config


def test_repeated_predictions_caching():
    np.random.seed(42)
    X_train = np.random.randn(200, 5)
    y_train = np.random.randn(200)
    X_test = np.random.randn(100, 5)
    sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    # 1. Without caching
    clear_kernel_cache()
    config_no_cache = create_config("RBF", lmbda=0.1, use_cache=False)

    start_time = time.time()
    gp_no_cache = DenseGP(config_no_cache, sigma)
    gp_no_cache.fit(X_train, y_train)

    for i in range(20):
        gp_no_cache.predict(X_test)

    time_no_cache = time.time() - start_time

    # 2. With caching
    clear_kernel_cache()
    config_cache = create_config("RBF", lmbda=0.1, use_cache=True)

    start_time = time.time()
    gp_cache = DenseGP(config_cache, sigma)
    gp_cache.fit(X_train, y_train)

    for i in range(20):
        gp_cache.predict(X_test)

    time_with_cache = time.time() - start_time
    cache_stats = get_cache_stats()
    assert (
        cache_stats is not None
    ), "Cache stats should be available when caching is enabled"
    assert cache_stats["hits"] > 0, "Should have cache hits for repeated predictions"
    assert cache_stats["hit_rate"] > 0, "Hit rate should be positive"

    if time_no_cache > 0 and time_with_cache > 0:
        assert (
            time_with_cache <= time_no_cache
        ), "Caching should not be slower than no caching"


def test_large_dataset_caching():
    np.random.seed(42)
    X_train = np.random.randn(500, 3)
    y_train = np.random.randn(500)
    X_test = np.random.randn(200, 3)
    sigma = np.array([1.0, 1.0, 1.0])

    # 1. Without caching
    clear_kernel_cache()
    config_no_cache = create_config("RBF", lmbda=0.1, use_cache=False)

    gp_no_cache = DenseGP(config_no_cache, sigma)
    gp_no_cache.fit(X_train, y_train)

    for i in range(5):
        gp_no_cache.predict(X_test)

    # 2. With caching
    clear_kernel_cache()
    config_cache = create_config("RBF", lmbda=0.1, use_cache=True)

    gp_cache = DenseGP(config_cache, sigma)
    gp_cache.fit(X_train, y_train)

    for i in range(5):
        gp_cache.predict(X_test)

    cache_stats = get_cache_stats()
    assert (
        cache_stats is not None
    ), "Cache stats should be available when caching is enabled"
    assert cache_stats["hits"] > 0, "Should have cache hits for repeated predictions"


def test_cache_config_options():
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.random.randn(100)
    X_test = np.random.randn(50, 2)
    sigma = np.array([1.0, 1.0])

    clear_kernel_cache()
    config_cache = create_config("RBF", lmbda=0.1, use_cache=True, cache_size=50)

    gp_cache = DenseGP(config_cache, sigma)
    gp_cache.fit(X_train, y_train)

    # Multiple predictions required to make cache hits (after first misses)
    for i in range(10):
        gp_cache.predict(X_test)

    cache_stats = gp_cache.get_cache_stats()
    assert (
        cache_stats is not None
    ), "Cache stats should be available when caching is enabled"
    assert cache_stats["cache_size"] > 0, "Cache should have some entries"
    assert (
        cache_stats["cache_size"] <= 50
    ), "Cache size should not exceed configured limit"

    # Test with cache disabled
    clear_kernel_cache()
    config_no_cache = create_config("RBF", lmbda=0.1, use_cache=False)

    gp_no_cache = DenseGP(config_no_cache, sigma)
    gp_no_cache.fit(X_train, y_train)

    for i in range(10):
        gp_no_cache.predict(X_test)

    cache_stats = gp_no_cache.get_cache_stats()
    assert cache_stats is None, "Cache stats should be None when caching is disabled"


def test_cache_clearing():
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    X_test = np.random.randn(20, 2)
    sigma = np.array([1.0, 1.0])

    config = create_config("RBF", lmbda=0.1, use_cache=True)
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)

    for i in range(5):
        gp.predict(X_test)

    initial_stats = gp.get_cache_stats()
    assert initial_stats["cache_size"] > 0, "Cache should have entries before clearing"

    gp.clear_cache()


@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_cache_with_different_kernels(kernel_type):
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.random.randn(100)
    X_test = np.random.randn(30, 3)
    sigma = np.array([1.0, 1.0, 1.0])

    clear_kernel_cache()
    config = create_config(kernel_type, lmbda=0.1, alpha=2.0, use_cache=True)

    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)

    for i in range(5):
        gp.predict(X_test)

    cache_stats = gp.get_cache_stats()
    assert cache_stats is not None, f"Cache should work with {kernel_type} kernel"
    assert cache_stats["hits"] > 0, f"Should have cache hits with {kernel_type} kernel"


@pytest.mark.parametrize("cache_size", [5, 50, 100])
def test_cache_size_configuration(cache_size):
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    X_test = np.random.randn(20, 2)
    sigma = np.array([1.0, 1.0])

    clear_kernel_cache()
    config = create_config("RBF", lmbda=0.1, use_cache=True, cache_size=cache_size)

    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)

    for i in range(10):
        gp.predict(X_test)

    cache_stats = gp.get_cache_stats()
    assert (
        cache_stats["cache_size"] <= cache_size
    ), f"Cache size should respect configuration limit of {cache_size}"


def test_cache_hit_rate():
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50)
    X_test = np.random.randn(20, 2)
    sigma = np.array([1.0, 1.0])

    config = create_config("RBF", lmbda=0.1, use_cache=True)
    gp = DenseGP(config, sigma)
    gp.fit(X_train, y_train)

    # First prediction (should be cache miss)
    gp.predict(X_test)
    stats_1 = gp.get_cache_stats()
    hit_rate_1 = stats_1["hit_rate"]

    # Multiple repeated predictions (should increase hit rate)
    for i in range(10):
        gp.predict(X_test)

    stats_2 = gp.get_cache_stats()
    hit_rate_2 = stats_2["hit_rate"]

    assert (
        hit_rate_2 >= hit_rate_1
    ), "Hit rate should increase with repeated predictions"
    assert (
        stats_2["hits"] > stats_1["hits"]
    ), "Number of hits should increase with repeated predictions"


if __name__ == "__main__":
    print("\nRunning cache tests...")

    test_repeated_predictions_caching()
    test_large_dataset_caching()
    test_cache_config_options()
    test_cache_with_different_kernels("RBF")
    test_cache_with_different_kernels("RQ")
    test_cache_size_configuration(50)
    test_cache_hit_rate()

    print("All cache tests completed!")
