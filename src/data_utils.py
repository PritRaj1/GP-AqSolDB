from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_aqsol_data(
    csv_path: str = "data/solubility-dataset.csv", scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess the AqSolDB dataset with feature engineering.
    Returns (X, y, feature_names).
    """
    proplist = [
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
    sol = pd.read_csv(csv_path)
    sol["NumHAcceptors"] = sol["NumHAcceptors"] + 1

    # FEATURE ENGINEERING
    X_basic = np.array([list(sol[prop] / sol["MolWt"]) for prop in proplist]).T
    X_basic = np.insert(X_basic, 0, list(np.log(sol["MolWt"])), axis=1)
    X_enhanced = X_basic.copy()

    # Skewed distributions are log-transformed
    log_features = []
    for i, prop in enumerate(proplist):
        if prop in ["NumSaturatedRings", "NumAliphaticRings"]:
            log_feat = np.log1p(sol[prop] / sol["MolWt"])
            X_enhanced = np.column_stack([X_enhanced, log_feat])
            log_features.append(f"log_{prop}/MolWt")

    # Interaction terms for highly correlated features
    heavy_atom_idx = 1
    valence_idx = 6
    interaction = X_basic[:, heavy_atom_idx] * X_basic[:, valence_idx]
    X_enhanced = np.column_stack([X_enhanced, interaction])

    # Polynomial features
    log_molwt_sq = X_basic[:, 0] ** 2
    X_enhanced = np.column_stack([X_enhanced, log_molwt_sq])

    # Ratio features
    acceptors_idx = 2
    donors_idx = 3
    hbond_ratio = np.where(
        X_basic[:, donors_idx] > 0,
        X_basic[:, acceptors_idx] / X_basic[:, donors_idx],
        0,
    )
    X_enhanced = np.column_stack([X_enhanced, hbond_ratio])

    # Molecular complexity features
    total_rings = (
        sol["NumAromaticRings"] + sol["NumSaturatedRings"] + sol["NumAliphaticRings"]
    ) / sol["MolWt"]
    X_enhanced = np.column_stack([X_enhanced, total_rings])

    basic_names = ["log_MolWt"] + [f"{prop}/MolWt" for prop in proplist]
    enhanced_names = (
        basic_names
        + log_features
        + [
            "HeavyAtom_Valence_Interaction",
            "log_MolWt_squared",
            "HBond_Acceptor_Donor_Ratio",
            "Total_Rings_per_MolWt",
        ]
    )

    # Remove highly correlated features (keep only one)
    corr_with_target = np.corrcoef(X_enhanced.T, sol["Solubility"])[:-1, -1]
    heavy_atom_corr = abs(corr_with_target[1])
    valence_corr = abs(corr_with_target[6])
    if heavy_atom_corr < valence_corr:
        X_final = np.delete(X_enhanced, 1, axis=1)
        feature_names = [name for i, name in enumerate(enhanced_names) if i != 1]
    else:
        X_final = np.delete(X_enhanced, 6, axis=1)
        feature_names = [name for i, name in enumerate(enhanced_names) if i != 6]

    if scale:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_final)

    y = np.array(sol["Solubility"])
    return X_final, y, feature_names
