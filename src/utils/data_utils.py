from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
