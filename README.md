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

## Scripts

### Testing

```bash 
pytest tests
```

## Data

The [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1) dataset in this repository comprises curated experimental solubility values, made openly accessible by the Autonomous Energy Materials Discovery research group.


## TODO

- Parallelize auto tune cross-validation
- Parallelize hp opt

## References

- [GPs](https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning)
- [Sparse FITC GPs](https://arxiv.org/abs/1606.04820)
- [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1)