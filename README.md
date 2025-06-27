# GaussianProcess-Solubility
Gaussian Processes modeling drug solubility with various optimizations.

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

## Optimizations

### Kernel Caching
Caching for kernel computations to improve performance during:
- **Repeated predictions**: Multiple predictions on the same test points
- **Large datasets**: When (vectorized) kernel computations become expensive
- **Multiple kernel evaluations**: When the same kernel parameters are used repeatedly

**Benefits:**
- Significant speedup for repeated computations
- Automatic cache management with LRU eviction
- Configurable cache size
- Cache stat for performance monitoring/benchmarking

## TODO

- Parallelize