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

### Kernel Caching
The implementation includes intelligent caching for kernel computations to improve performance during:
- **Repeated predictions**: Multiple predictions on the same test points
- **Large datasets**: When kernel computations become expensive
- **Multiple kernel evaluations**: When the same kernel parameters are used repeatedly

**Usage:**
```ini
[KERNEL]
type = "RBF"
lmbda = 1.0
alpha = 1.0
use_cache = true
cache_size = 100
```

**Benefits:**
- Significant speedup for repeated computations
- Automatic cache management with LRU eviction
- Configurable cache size
- Cache statistics for performance monitoring

## TODO

- Parallelize