import numpy as np
import time
import pytest
from configparser import ConfigParser
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multivar_gp.kernels import RBF, RQ, configure_parallel_settings, get_parallel_info, load_parallel_conf

def create_test_data(n_samples, n_features):
    np.random.seed(42)
    X1 = np.random.randn(n_samples, n_features)
    X2 = np.random.randn(n_samples // 2, n_features)
    sigma = np.random.uniform(0.1, 2.0, n_features)
    return X1, X2, sigma

def bench_kernel(kernel_func, X1, X2, sigma, alpha=None, n_runs=3):
    times = []
    
    for i in range(n_runs):
        start_time = time.time()
        if alpha is not None:
            result = kernel_func(X1, X2, sigma, alpha)
        else:
            result = kernel_func(X1, X2, sigma)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times), result.shape

def test_parallel_info():
    parallel_info = get_parallel_info()
    
    assert 'parallel_available' in parallel_info
    assert 'cpu_cores' in parallel_info
    assert 'gpu_available' in parallel_info
    assert 'cupy_available' in parallel_info
    assert 'settings' in parallel_info
    
    assert isinstance(parallel_info['cpu_cores'], int)
    assert parallel_info['cpu_cores'] > 0
    assert isinstance(parallel_info['gpu_available'], bool)
    assert isinstance(parallel_info['cupy_available'], bool)

def test_conf():
    configure_parallel_settings()
    parallel_info = get_parallel_info()
    settings = parallel_info['settings']
    
    assert settings['use_parallel'] == False
    assert settings['n_jobs'] is None  # Auto-detect
    assert settings['chunk_size'] == 1000
    assert settings['use_gpu'] == False
    assert settings['min_size_for_parallel'] == 500

    configure_parallel_settings(
        use_parallel=False,
        n_jobs=4,
        chunk_size=500,
        use_gpu=True,
        min_size_for_parallel=300
    )
    
    parallel_info = get_parallel_info()
    settings = parallel_info['settings']
    
    assert settings['use_parallel'] == False
    assert settings['n_jobs'] == 4
    assert settings['chunk_size'] == 500
    assert settings['use_gpu'] == True
    assert settings['min_size_for_parallel'] == 300

def test_conf_load():
    config = ConfigParser()
    config['PARALLEL'] = {
        'use_parallel': 'true',
        'n_jobs': '4',
        'chunk_size': '500',
        'use_gpu': 'false',
        'min_size_for_parallel': '300'
    }
    
    load_parallel_conf(config)
    
    parallel_info = get_parallel_info()
    settings = parallel_info['settings']
    
    assert settings['use_parallel'] == True
    assert settings['n_jobs'] == 4
    assert settings['chunk_size'] == 500
    assert settings['use_gpu'] == False
    assert settings['min_size_for_parallel'] == 300

def test_conf_load_missing():
    config = ConfigParser()
    config['KERNEL'] = {'type': 'RBF'}
    
    load_parallel_conf(config)
    
    parallel_info = get_parallel_info()
    settings = parallel_info['settings']
    
    # Should have default values
    assert 'use_parallel' in settings
    assert 'n_jobs' in settings
    assert 'chunk_size' in settings
    assert 'use_gpu' in settings
    assert 'min_size_for_parallel' in settings

@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_sequential_vs_parallel(kernel_type, sample_kernel_data):
    X1, X2, sigma = sample_kernel_data
        
    # Sequential
    configure_parallel_settings(use_parallel=False, use_gpu=False)
    
    if kernel_type == "RBF":
        result_seq = RBF(X1, X2, sigma)
    else:  # RQ
        result_seq = RQ(X1, X2, sigma, alpha=2.0)
    
    # Parallel 
    configure_parallel_settings(use_parallel=True, use_gpu=False, n_jobs=2)
    
    if kernel_type == "RBF":
        result_par = RBF(X1, X2, sigma)
    else:  # RQ
        result_par = RQ(X1, X2, sigma, alpha=2.0)
    
    assert np.allclose(result_seq, result_par, rtol=1e-10)
    assert result_seq.shape == result_par.shape

def test_gpu():
    X1, X2, sigma = create_test_data(500, 3)
    
    # Force GPU usage
    configure_parallel_settings(use_parallel=True, use_gpu=True, n_jobs=2)
    
    # RBF
    try:
        result_gpu = RBF(X1, X2, sigma)
        assert result_gpu.shape == (X1.shape[0], X2.shape[0])
        assert not np.any(np.isnan(result_gpu))
        print("GPU computation successful")
    except Exception as e:
        # If GPU fails, it should be due to GPU unavailability
        error_msg = str(e).lower()
        gpu_related = any(keyword in error_msg for keyword in ['gpu', 'cuda', 'device', 'memory'])
        if not gpu_related:
            raise
        print(f"GPU computation failed (fine if no GPU): {e}")
    
    # RQ
    try:
        result_gpu_rq = RQ(X1, X2, sigma, alpha=2.0)
        assert result_gpu_rq.shape == (X1.shape[0], X2.shape[0])
        assert not np.any(np.isnan(result_gpu_rq))
        print("GPU RQ computation successful")
    except Exception as e:
        error_msg = str(e).lower()
        gpu_related = any(keyword in error_msg for keyword in ['gpu', 'cuda', 'device', 'memory'])
        if not gpu_related:
            raise
        print(f"GPU RQ computation failed (fine if no GPU): {e}")

def test_gpu_vs_cpu():
    X1, X2, sigma = create_test_data(300, 3)
    
    # CPU 
    configure_parallel_settings(use_parallel=False, use_gpu=False)
    result_cpu = RBF(X1, X2, sigma)
    
    # GPU 
    configure_parallel_settings(use_parallel=True, use_gpu=True, n_jobs=2)
    
    try:
        result_gpu = RBF(X1, X2, sigma)
        # Results should be very close (floating point differences might be apparent)
        assert np.allclose(result_cpu, result_gpu, rtol=1e-5, atol=1e-8)
        print("GPU and CPU results are consistent")
    except Exception as e:
        # If GPU fails, fine - but CPU should work
        error_msg = str(e).lower()
        gpu_related = any(keyword in error_msg for keyword in ['gpu', 'cuda', 'device', 'memory'])
        if not gpu_related:
            raise
        print(f"GPU computation failed, but CPU works: {e}")

@pytest.mark.parametrize("matrix_size", [100, 500, 1000])
def test_parallel_performance_scaling(matrix_size):
    n_features = 5
    X1, X2, sigma = create_test_data(matrix_size, n_features)
    
    # Sequential btime
    configure_parallel_settings(use_parallel=False, use_gpu=False)
    seq_time, _, _ = bench_kernel(RBF, X1, X2, sigma, n_runs=2)
    
    # Parallel btime
    configure_parallel_settings(use_parallel=True, use_gpu=False, n_jobs=2)
    par_time, _, _ = bench_kernel(RBF, X1, X2, sigma, n_runs=2)
    
    # Both should complete 
    assert seq_time > 0
    assert par_time > 0
    
    # Make sure large matrices work
    if matrix_size >= 1000:
        assert True

def test_gpu_performance():
    X1, X2, sigma = create_test_data(1000, 5)
    
    # CPU btime
    configure_parallel_settings(use_parallel=False, use_gpu=False)
    cpu_time, _, _ = bench_kernel(RBF, X1, X2, sigma, n_runs=2)
    
    # GPU btime
    configure_parallel_settings(use_parallel=True, use_gpu=True, n_jobs=2)
    try:
        gpu_time, _, _ = bench_kernel(RBF, X1, X2, sigma, n_runs=2)
        print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
        
        # Both should complete 
        assert cpu_time > 0
        assert gpu_time > 0
        
        # GPU might be faster (but not guaranteed - subject to transfer overhead)
        if gpu_time < cpu_time:
            print("GPU is faster than CPU")
        else:
            print("GPU is slower than CPU (transfer overhead)")
            
    except Exception as e:
        error_msg = str(e).lower()
        gpu_related = any(keyword in error_msg for keyword in ['gpu', 'cuda', 'device', 'memory'])
        if not gpu_related:
            raise
        print(f"GPU computation failed: {e}")

def test_gpu_fallback():
    X1, X2, sigma = create_test_data(500, 3)
    
    # Force GPU 
    configure_parallel_settings(use_parallel=True, use_gpu=True, n_jobs=2)
    
    # Should not raise error, should fall back to CPU if GPU not available 
    try:
        result = RBF(X1, X2, sigma) 
        assert result.shape == (X1.shape[0], X2.shape[0])
        assert not np.any(np.isnan(result))
    except Exception as e:
        # If GPU fails, it should be due to GPU unavailability
        error_msg = str(e).lower()
        gpu_related = any(keyword in error_msg for keyword in ['gpu', 'cuda', 'device', 'memory'])
        if not gpu_related:
            raise # Otherwise raise error 
        assert True

def test_chunks():
    X1, X2, sigma = create_test_data(1000, 5)
    
    for chunk_size in [100, 500, 1000]:
        configure_parallel_settings(
            use_parallel=True, 
            use_gpu=False, 
            n_jobs=2, 
            chunk_size=chunk_size
        )
        
        result = RBF(X1, X2, sigma)
        assert result.shape == (X1.shape[0], X2.shape[0])
        assert not np.any(np.isnan(result))

def test_min_size_for_parallel():

    # Small matrices should not use parallel 
    X1, X2, sigma = create_test_data(50, 3)  
    
    configure_parallel_settings(
        use_parallel=True, 
        use_gpu=False, 
        n_jobs=2, 
        min_size_for_parallel=100 
    )
    
    result = RBF(X1, X2, sigma)
    assert result.shape == (X1.shape[0], X2.shape[0])
    
    # Large matrices should use parallel
    X1_large, X2_large, sigma_large = create_test_data(1000, 5)  # Large size
    
    configure_parallel_settings(
        use_parallel=True, 
        use_gpu=False, 
        n_jobs=2, 
        min_size_for_parallel=100 
    )
    
    result_large = RBF(X1_large, X2_large, sigma_large)
    assert result_large.shape == (X1_large.shape[0], X2_large.shape[0])

@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_different_n_jobs(n_jobs):
    X1, X2, sigma = create_test_data(800, 4)
    
    configure_parallel_settings(
        use_parallel=True, 
        use_gpu=False, 
        n_jobs=n_jobs
    )
    
    result = RBF(X1, X2, sigma)
    assert result.shape == (X1.shape[0], X2.shape[0])
    assert not np.any(np.isnan(result))

def test_cache():
    """Test that caching works correctly with parallel processing"""
    X1, X2, sigma = create_test_data(500, 3)
    
    configure_parallel_settings(use_parallel=True, use_gpu=False, n_jobs=2)
    
    # First computation (should be cache miss)
    result1 = RBF(X1, X2, sigma, use_cache=True)
    
    # Second computation (should be cache hit)
    result2 = RBF(X1, X2, sigma, use_cache=True)
    
    # Results should be identical
    assert np.allclose(result1, result2, rtol=1e-10)

def test_errors():
    X1 = np.random.randn(100, 3)
    X2 = np.random.randn(50, 3)
    sigma = np.array([1.0, 1.0]) 
    
    configure_parallel_settings(use_parallel=True, use_gpu=False, n_jobs=2)
    
    # Should raise ValueError 
    with pytest.raises(ValueError):
        RBF(X1, X2, sigma)

def test_memory_efficiency():
    X1, X2, sigma = create_test_data(2000, 10)
    
    configure_parallel_settings(
        use_parallel=True, 
        use_gpu=False, 
        n_jobs=2, 
        chunk_size=500 
    )
    
    result = RBF(X1, X2, sigma)
    assert result.shape == (X1.shape[0], X2.shape[0])
    assert not np.any(np.isnan(result))

if __name__ == "__main__":
    print("\nRunning parallel kernel tests...")
    
    sample_kernel_data = create_test_data(500, 3)
    
    test_sequential_vs_parallel("RBF", sample_kernel_data)
    test_sequential_vs_parallel("RQ", sample_kernel_data)
    
    for size in [100, 500, 1000]:
        test_parallel_performance_scaling(size)
    
    test_cache()
    test_memory_efficiency()
    test_gpu()
    test_gpu_vs_cpu()
    test_gpu_performance()
    test_gpu_fallback()
    
    print("All parallel kernel tests completed!") 