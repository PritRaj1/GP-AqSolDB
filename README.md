# GaussianProcess-Solubility
UQ with Gaussian Processes modeling drug solubility.

## Setup

Need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Choose your favourite installer and run: 

```bash
bash <conda-installer-name>-latest-Linux-x86_64.sh
```

```bash
bash setup/auto.sh
```

## Data

The [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1) dataset in this repository comprises curated experimental solubility values, made openly accessible by the Autonomous Energy Materials Discovery research group.

## Features

### Unified Gaussian Process System
A unified GP class that automatically switches between dense and sparse implementations based on configuration:

**Configuration:**
```ini
[KERNEL]
type = "RBF"
lmbda = 1.0
alpha = 1.0
use_cache = true
cache_size = 100

[SPARSE]
use_sparse = false
num_inducing = 20
inducing_method = random
```

**Usage:**
```python
from src.gp import GP

# Create GP (automatically chooses dense or sparse based on config)
gp = GP(config, sigma)

# Fit and predict (same interface for both types)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)

# Get model information
info = gp.get_model_info()
print(f"Model type: {info['model_type']}")
```

### Kernel Caching
Intelligent caching for kernel computations to improve performance during:
- **Repeated predictions**: Multiple predictions on the same test points
- **Large datasets**: When kernel computations become expensive
- **Multiple kernel evaluations**: When the same kernel parameters are used repeatedly

**Benefits:**
- Significant speedup for repeated computations
- Automatic cache management with LRU eviction
- Configurable cache size
- Cache statistics for performance monitoring

### Sparse Gaussian Processes (FITC)
Implements the Fully Independent Training Conditional (FITC) approximation for scaling to large datasets:

**Features:**
- **Inducing points**: Uses a subset of training data as inducing points
- **Computational scaling**: Reduces complexity from O(N³) to O(NM²) where M << N
- **Memory efficient**: Only stores M×M kernel matrix instead of N×N
- **Uncertainty quantification**: Maintains proper uncertainty estimates
- **Multiple selection methods**: Random or uniform inducing point selection

**Benefits:**
- Scales to large datasets (1000+ points)
- Maintains uncertainty quantification
- Configurable accuracy/speed trade-off
- Automatic fallback to full dataset when num_inducing >= N

### Enhanced Auto-tuning with BIC
Advanced hyperparameter optimization that automatically chooses between dense and sparse GPs using Bayesian Information Criterion (BIC):

**Features:**
- **Model selection**: Automatically chooses between dense and sparse GPs
- **BIC optimization**: Uses BIC to balance accuracy vs complexity
- **Comprehensive search**: Optimizes kernel type, parameters, and inducing points
- **Cross-validation**: Robust evaluation with k-fold cross-validation

**Usage:**
```python
from src.auto_tune import GPAutoTuner

# Create auto-tuner
tuner = GPAutoTuner(X_train, y_train)

# Run optimization
best_params = tuner.optimize(n_trials=100)

# Load optimized parameters
config, sigmas = tuner.load_optimized_parameters()
```

**BIC Calculation:**
```
BIC = n * log(MSE) + k * log(n)
```
where n = number of samples, k = number of parameters

Lower BIC indicates better model (penalizes complexity).

## Optimisations

- **Optuna** - used to automate hyperparameter search.
- **Kernel caching** - to speed up repeated predictions on the same data, the kernel matrix is cached and evicted with LRU. If the dataset is small, turn this off. It also does not help with autotuning since optuna proposes different hyperparameters each time.

## TODO

- Parallelize

## References

- [GPs](https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning)
- [Sparse FITC GPs](https://arxiv.org/abs/1606.04820)
- [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1)