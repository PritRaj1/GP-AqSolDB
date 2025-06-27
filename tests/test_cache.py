import numpy as np
import time
import sys
import os
from configparser import ConfigParser
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.GP_fcns import GP
from src.kernels import get_cache_stats, clear_kernel_cache

def create_config(kernel_type="RBF", lmbda=0.1, alpha=1.0, use_cache=True, cache_size=100):
    config = ConfigParser()
    config['KERNEL'] = {
        'type': kernel_type,
        'lmbda': str(lmbda),
        'alpha': str(alpha),
        'use_cache': str(use_cache).lower(),
        'cache_size': str(cache_size)
    }
    return config

def test_repeated_predictions():
    """Test caching benefits for repeated predictions on same test points"""
    print("Testing caching for repeated predictions...")
    
    np.random.seed(42)
    X_train = np.random.randn(200, 5)  
    y_train = np.random.randn(200)
    X_test = np.random.randn(100, 5)  
    
    config = create_config("RBF", lmbda=0.1, use_cache=False)
    sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    # 1. Without caching
    print("\n1. Repeated predictions WITHOUT caching:")
    clear_kernel_cache()
    
    start_time = time.time()
    gp_no_cache = GP(config, sigma)
    gp_no_cache.fit(X_train, y_train)
    
    for i in range(50):  
        y_pred = gp_no_cache.predict(X_test)
    
    time_no_cache = time.time() - start_time
    print(f"   Time without caching: {time_no_cache:.4f} seconds")
    
    # 2. With caching
    print("\n2. Repeated predictions WITH caching:")
    clear_kernel_cache()
    
    config = create_config("RBF", lmbda=0.1, use_cache=True)
    start_time = time.time()
    gp_cache = GP(config, sigma)
    gp_cache.fit(X_train, y_train)
    
    for i in range(50):  
        y_pred = gp_cache.predict(X_test)
    
    time_with_cache = time.time() - start_time
    print(f"   Time with caching: {time_with_cache:.4f} seconds")
    
    cache_stats = get_cache_stats()
    print(f"\n3. Cache statistics:")
    print(f"   Cache hits: {cache_stats['hits']}")
    print(f"   Cache misses: {cache_stats['misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Cache size: {cache_stats['cache_size']}")
    
    # Calculate speedup
    if time_no_cache > 0 and time_with_cache > 0:
        speedup = time_no_cache / time_with_cache
        print(f"\n4. Performance improvement:")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Time saved: {time_no_cache - time_with_cache:.4f} seconds")
    elif time_with_cache == 0:
        print(f"\n4. Performance improvement:")
        print(f"   Speedup: ∞ (caching eliminated computation time)")
        print(f"   Time saved: {time_no_cache:.4f} seconds")
    else:
        print(f"\n4. Performance improvement:")
        print(f"   Unable to calculate speedup (both times are zero)")

def test_large_dataset_caching():
    """Test caching benefits with larger datasets"""
    print("\n" + "="*60)
    print("Testing caching with larger datasets...")
    
    np.random.seed(42)
    X_train = np.random.randn(500, 3)  
    y_train = np.random.randn(500)
    X_test = np.random.randn(200, 3)
    
    # 1. Without caching
    print("\n1. Large dataset WITHOUT caching:")
    clear_kernel_cache()
    
    config = create_config("RBF", lmbda=0.1, use_cache=False)
    sigma = np.array([1.0, 1.0, 1.0])
    
    start_time = time.time()
    gp_no_cache = GP(config, sigma)
    gp_no_cache.fit(X_train, y_train)
    
    for i in range(10):
        y_pred = gp_no_cache.predict(X_test)
    
    time_no_cache = time.time() - start_time
    print(f"   Time without caching: {time_no_cache:.4f} seconds")
    
    # 2. With caching
    print("\n2. Large dataset WITH caching:")
    clear_kernel_cache()
    
    config = create_config("RBF", lmbda=0.1, use_cache=True)
    start_time = time.time()
    gp_cache = GP(config, sigma)
    gp_cache.fit(X_train, y_train)
    
    for i in range(10):
        y_pred = gp_cache.predict(X_test)
    
    time_with_cache = time.time() - start_time
    print(f"   Time with caching: {time_with_cache:.4f} seconds")
    
    # Report cache statistics
    cache_stats = get_cache_stats()
    print(f"\n3. Cache statistics:")
    print(f"   Cache hits: {cache_stats['hits']}")
    print(f"   Cache misses: {cache_stats['misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Cache size: {cache_stats['cache_size']}")
    
    # Calculate speedup
    if time_no_cache > 0 and time_with_cache > 0:
        speedup = time_no_cache / time_with_cache
        print(f"\n4. Performance improvement:")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Time saved: {time_no_cache - time_with_cache:.4f} seconds")

def test_cache_config_options():
    """Test that cache config options work correctly"""
    print("\n" + "="*60)
    print("Testing cache configuration options...")
    
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.random.randn(100)
    X_test = np.random.randn(50, 2)
    sigma = np.array([1.0, 1.0])
    
    # Test with cache enabled
    print("\n1. Testing with cache enabled:")
    clear_kernel_cache()
    config = create_config("RBF", lmbda=0.1, use_cache=True, cache_size=50)
    
    gp = GP(config, sigma)
    gp.fit(X_train, y_train)
    
    # Multiple predictions to make cache hits
    for i in range(20):
        y_pred = gp.predict(X_test)
    
    cache_stats = gp.get_cache_stats()
    print(f"   Cache enabled: {cache_stats is not None}")
    if cache_stats:
        print(f"   Cache hits: {cache_stats['hits']}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Test with cache disabled
    print("\n2. Testing with cache disabled:")
    clear_kernel_cache()
    config = create_config("RBF", lmbda=0.1, use_cache=False)
    
    gp = GP(config, sigma)
    gp.fit(X_train, y_train)
    
    for i in range(20):
        y_pred = gp.predict(X_test)
    
    cache_stats = gp.get_cache_stats()
    print(f"   Cache enabled: {cache_stats is not None}")
    
    # Test cache clearing
    print("\n3. Testing cache clearing:")
    config = create_config("RBF", lmbda=0.1, use_cache=True)
    gp = GP(config, sigma)
    gp.fit(X_train, y_train)
    
    # Generate some cache entries
    for i in range(10):
        y_pred = gp.predict(X_test)
    
    initial_stats = gp.get_cache_stats()
    print(f"   Cache size before clearing: {initial_stats['cache_size']}")
    
    gp.clear_cache()
    final_stats = gp.get_cache_stats()
    print(f"   Cache size after clearing: {final_stats['cache_size']}")

def test_cache_with_different_kernels():
    """Test caching with different kernel types"""
    print("\n" + "="*60)
    print("Testing caching with different kernel types...")
    
    np.random.seed(42)
    X_train = np.random.randn(150, 3)
    y_train = np.random.randn(150)
    X_test = np.random.randn(75, 3)
    sigma = np.array([1.0, 1.0, 1.0])
    
    kernels = ["RBF", "RQ"]
    
    for kernel_type in kernels:
        print(f"\nTesting {kernel_type} kernel:")
        clear_kernel_cache()
        
        # Without caching
        config = create_config(kernel_type, lmbda=0.1, alpha=2.0, use_cache=False)
        start_time = time.time()
        gp_no_cache = GP(config, sigma)
        gp_no_cache.fit(X_train, y_train)
        
        for i in range(15):
            y_pred = gp_no_cache.predict(X_test)
        
        time_no_cache = time.time() - start_time
        
        # With caching
        clear_kernel_cache()
        config = create_config(kernel_type, lmbda=0.1, alpha=2.0, use_cache=True)
        start_time = time.time()
        gp_cache = GP(config, sigma)
        gp_cache.fit(X_train, y_train)
        
        for i in range(15):
            y_pred = gp_cache.predict(X_test)
        
        time_with_cache = time.time() - start_time
        
        cache_stats = gp_cache.get_cache_stats()
        
        print(f"   Time without caching: {time_no_cache:.4f}s")
        print(f"   Time with caching: {time_with_cache:.4f}s")
        print(f"   Cache hits: {cache_stats['hits']}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
        
        if time_no_cache > 0 and time_with_cache > 0:
            speedup = time_no_cache / time_with_cache
            print(f"   Speedup: {speedup:.2f}x")

def test_cache_size_configuration():
    """Test that different cache sizes work correctly"""
    print("\n" + "="*60)
    print("Testing cache size configuration...")
    
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.random.randn(100)
    X_test = np.random.randn(50, 2)
    sigma = np.array([1.0, 1.0])
    
    cache_sizes = [10, 50, 100]
    
    for cache_size in cache_sizes:
        print(f"\nTesting cache size: {cache_size}")
        clear_kernel_cache()
        
        config = create_config("RBF", lmbda=0.1, use_cache=True, cache_size=cache_size)
        gp = GP(config, sigma)
        gp.fit(X_train, y_train)
        
        # Generate many predictions to test cache eviction
        for i in range(30):
            y_pred = gp.predict(X_test)
        
        cache_stats = gp.get_cache_stats()
        print(f"   Final cache size: {cache_stats['cache_size']}")
        print(f"   Cache hits: {cache_stats['hits']}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")

if __name__ == "__main__":
    test_repeated_predictions()
    test_large_dataset_caching()
    test_cache_config_options()
    test_cache_with_different_kernels()
    test_cache_size_configuration() 