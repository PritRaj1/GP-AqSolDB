from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..core.kernels import compute_kernel


def load_aqsol_data(
    csv_path: str = "data/solubility-dataset.csv",
    scale: bool = False,
    return_frame: bool = False,
    return_scaler: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[str]],
    Tuple[pd.DataFrame, np.ndarray, List[str]],
    Tuple[np.ndarray, np.ndarray, List[str], StandardScaler],
    Tuple[pd.DataFrame, np.ndarray, List[str], StandardScaler],
]:
    """
    Load and preprocess the AqSolDB dataset with feature engineering.
    Returns (X, y, feature_names) and optionally the scaler used.
    """
    sol = pd.read_csv(csv_path)

    base_features = [
        "MolWt",
        "MolLogP",
        "MolMR",
        "TPSA",
        "LabuteASA",
        "BalabanJ",
        "BertzCT",
        "HeavyAtomCount",
        "NumHAcceptors",
        "NumHDonors",
        "NumHeteroatoms",
        "NumRotatableBonds",
        "NumValenceElectrons",
        "NumAromaticRings",
        "NumSaturatedRings",
        "NumAliphaticRings",
        "RingCount",
    ]

    missing_columns = set(base_features + ["Solubility"]) - set(sol.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in dataset: {sorted(missing_columns)}"
        )

    X_df = sol[base_features].copy()

    # Log / stabilising transforms for positively skewed descriptors
    skewed_features = {
        "log_MolWt": np.log1p(X_df["MolWt"]),
        "log_TPSA": np.log1p(X_df["TPSA"]),
        "log_LabuteASA": np.log1p(X_df["LabuteASA"]),
        "log_BertzCT": np.log1p(X_df["BertzCT"]),
    }

    # Chemically motivated ratios/fractions
    total_rings = (
        sol["NumAromaticRings"] + sol["NumSaturatedRings"] + sol["NumAliphaticRings"]
    )
    fraction_features = {
        "HBond_Total": sol["NumHAcceptors"] + sol["NumHDonors"],
        "HBond_Donor_Acceptor_Ratio": (sol["NumHDonors"] + 1.0)
        / (sol["NumHAcceptors"] + 1.0),
        "Heteroatom_Fraction": sol["NumHeteroatoms"] / (sol["HeavyAtomCount"] + 1.0),
        "Rotatable_Bond_Fraction": sol["NumRotatableBonds"]
        / (sol["HeavyAtomCount"] + 1.0),
        "Aromatic_Ring_Fraction": np.where(
            total_rings > 0, sol["NumAromaticRings"] / total_rings, 0.0
        ),
    }

    for name, values in {**skewed_features, **fraction_features}.items():
        X_df[name] = values.astype(np.float64)

    feature_names = list(X_df.columns)
    y = sol["Solubility"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_processed: Union[np.ndarray, pd.DataFrame]

    if scale:
        X_array = scaler.fit_transform(X_df)
        if return_frame:
            X_processed = pd.DataFrame(X_array, columns=feature_names, index=sol.index)

        else:
            X_processed = X_array
    else:
        if return_frame:
            X_processed = X_df.copy()

        else:
            X_processed = X_df.to_numpy(dtype=np.float64)

    if return_scaler:
        return X_processed, y, feature_names, scaler

    return X_processed, y, feature_names


def _median_pairwise_dists(X: np.ndarray) -> np.ndarray:
    """Median absolute pairwise distance per feature column."""
    d = X.shape[1]
    medians = np.ones(d)
    for dim in range(d):
        col = X[:, dim]
        dists = np.abs(col[:, None] - col[None, :])
        nonzero = dists[dists > 0]
        if len(nonzero):
            medians[dim] = float(np.median(nonzero))

    return medians


def infer_defaults(
    X: np.ndarray,
    csv_path: str = "data/solubility-dataset.csv",
) -> Dict[str, object]:
    """Infer GP hyperparameter defaults from data properties."""
    n, d = X.shape
    rng = np.random.RandomState(42)

    # Subsample for expensive ops
    sub_n = min(500, n)
    sub_idx = rng.choice(n, sub_n, replace=False)
    X_sub = X[sub_idx]

    # Median-heuristic sigmas (1/median_dist per dim)
    medians = _median_pairwise_dists(X_sub)
    sigmas_heuristic = 1.0 / np.maximum(medians, 1e-8)

    # Nystrom residuals for num_inducing
    sigma_jnp = jnp.array(sigmas_heuristic)
    X_sub_jnp = jnp.array(X_sub)
    K = np.asarray(compute_kernel("RBF", X_sub_jnp, X_sub_jnp, sigma_jnp))
    tr_K = np.trace(K)

    num_inducing = int(np.sqrt(n))
    for m in [10, 20, 50, 100, 200]:
        if m >= sub_n:
            continue

        km = KMeans(n_clusters=m, random_state=42, n_init=3, max_iter=50)
        Z_jnp = jnp.array(km.fit(X_sub).cluster_centers_)
        Knm = np.asarray(compute_kernel("RBF", X_sub_jnp, Z_jnp, sigma_jnp))
        Kmm = np.asarray(
            compute_kernel("RBF", Z_jnp, Z_jnp, sigma_jnp)
        ) + 1e-6 * np.eye(m)
        Q = Knm @ np.linalg.inv(Kmm) @ Knm.T
        if (tr_K - np.trace(Q)) / max(tr_K, 1e-12) < 0.05:
            num_inducing = m
            break

    # Lambda range from measurement noise in CSV
    lmbda_lo = 1e-3
    try:
        sol = pd.read_csv(csv_path)
        if "SD" in sol.columns:
            mean_sd_sq = float(np.mean(sol["SD"].to_numpy(dtype=np.float64) ** 2))
            if np.isfinite(mean_sd_sq) and mean_sd_sq > 0:
                lmbda_lo = mean_sd_sq

    except (FileNotFoundError, KeyError):
        pass

    # Sigma ranges per dimension: [0.5, 2.0] * heuristic, clamped to [0.1, 5.0]
    sigma_ranges = []
    for dim in range(d):
        lo = float(np.clip(0.5 * sigmas_heuristic[dim], 0.1, 5.0))
        hi = float(np.clip(2.0 * sigmas_heuristic[dim], 0.1, 5.0))
        if lo > hi:
            lo, hi = hi, lo
        sigma_ranges.append((lo, hi))

    return {
        "use_sparse": n > 2000,
        "num_inducing": num_inducing,
        "inducing_method": "kmeans",
        "lmbda_range": (lmbda_lo, 0.1),
        "sigma_ranges": sigma_ranges,
    }
