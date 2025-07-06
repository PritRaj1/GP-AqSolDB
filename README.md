# GaussianProcess-Solubility
UQ with Gaussian Processes modeling drug solubility.

<p align="center">
  <img src="figures/learning_evolution.gif" alt="Active Learning">
  <br>
  <em>Active Learning of Gaussian Process</em>
</p>

Understanding solubility is important in drug discovery because it reflects a drug's ADME, (absorption, distribution, metabolism, excretion), and thus its effectiveness in therapy.

Uncertainty quantification is also important, (in pretty much any clinical setting), since you want drug development to progress quickly and in a trustworthy manner. It's not enough to simply make decisions, you must also quantify the validity of them.


## Setup

Need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Choose your favourite installer and run: 

```bash
bash <conda-installer-name>-latest-Linux-x86_64.sh
```

Run:
```bash
bash setup/auto.sh
```

NOTE: This will create a conda environment for the Python dependencies called 'GP_sol'. The terminal multiplexer 'tmux' will also be installed in GP_sol, which is my personally preferred method of running long programs.

## Scripts

```bash
conda activate GP_sol
```

To test:

```bash 
pytest tests
```

To run:

```bash
# In terminal
python main.py 
```

```bash
# In pseudo terminal (preferred)
bash run.sh
```

## Data

The [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1) dataset in this repository comprises curated experimental solubility values, made openly accessible by the Autonomous Energy Materials Discovery research group.


## TODO

- Parallelize auto tune cross-validation
- Parallelize hp opt

## References

- [GPs](https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning)
- [Sparse FITC GPs](https://arxiv.org/abs/1606.04820)
- [GP-Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.18397)
- [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1)