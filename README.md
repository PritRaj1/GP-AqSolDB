# Gaussian Processes - AqSolDB

<p align="center">
  <img src="figures/learning_evolution.gif" alt="Active Learning">
  <br>
  <em>Epistemic uncertainty reduction</em>
</p>

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync          # install deps
uv sync --group dev   # + dev tools
```

## Usage

```bash
uv run gp-kan gp             # train standard GP
uv run gp-kan kan            # train GP-KAN hybrid
uv run gp-kan stats          # dataset statistics
uv run gp-kan clean          # remove generated artifacts
```

## Dev

```bash
uv run pytest                        # tests
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # format
uv run mypy src/                     # type check
```

## Data

The [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1) dataset in this repository comprises curated experimental solubility values, made openly accessible by the Autonomous Energy Materials Discovery research group.

## References

- [GPs](https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning)
- [Sparse FITC GPs](https://arxiv.org/abs/1606.04820)
- [GP-Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.18397)
- [AqSolDB](https://doi.org/10.1038/s41597-019-0151-1)
