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
    sol = pd.read_csv(csv_path)

    basic_props = [
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

    # Normalize to . / MolWt
    X_basic = np.array([list(sol[prop] / sol["MolWt"]) for prop in basic_props]).T
    X_basic = np.insert(X_basic, 0, list(np.log(sol["MolWt"])), axis=1)

    X_enhanced = X_basic.copy()
    X_enhanced = np.column_stack([X_enhanced, sol["MolLogP"]])
    X_enhanced = np.column_stack([X_enhanced, sol["MolMR"] / sol["MolWt"]])
    X_enhanced = np.column_stack([X_enhanced, sol["TPSA"] / sol["MolWt"]])
    X_enhanced = np.column_stack([X_enhanced, sol["LabuteASA"] / sol["MolWt"]])
    X_enhanced = np.column_stack([X_enhanced, sol["BertzCT"] / sol["MolWt"]])
    X_enhanced = np.column_stack([X_enhanced, sol["BalabanJ"]])

    # FEATURE ENGINEERING

    # Log transformations for skewed distributions
    log_features = []
    for i, prop in enumerate(basic_props):
        if prop in ["NumSaturatedRings", "NumAliphaticRings", "RingCount"]:
            log_feat = np.log1p(sol[prop] / sol["MolWt"])
            X_enhanced = np.column_stack([X_enhanced, log_feat])
            log_features.append(f"log_{prop}/MolWt")

    # Interaction terms for highly correlated features
    heavy_atom_idx = 1
    valence_idx = 6
    heavy_valence_interaction = X_basic[:, heavy_atom_idx] * X_basic[:, valence_idx]
    X_enhanced = np.column_stack([X_enhanced, heavy_valence_interaction])

    # Lipophilicity-polarity interaction
    # How soluble is it in non-polar vs polar solvents?
    mol_logp_idx = len(basic_props) + 1  # Index after adding MolLogP
    tpsa_idx = len(basic_props) + 3  # Index after adding TPSA
    lipo_polar_interaction = X_enhanced[:, mol_logp_idx] * X_enhanced[:, tpsa_idx]
    X_enhanced = np.column_stack([X_enhanced, lipo_polar_interaction])

    # Molecular complexity features
    total_rings = (
        sol["NumAromaticRings"] + sol["NumSaturatedRings"] + sol["NumAliphaticRings"]
    ) / sol["MolWt"]
    X_enhanced = np.column_stack([X_enhanced, total_rings])

    # Hydrogen bonding capacity
    hbond_capacity = (sol["NumHAcceptors"] + sol["NumHDonors"]) / sol["MolWt"]
    X_enhanced = np.column_stack([X_enhanced, hbond_capacity])

    # Flexibility index
    flexibility = sol["NumRotatableBonds"] / sol["MolWt"]
    X_enhanced = np.column_stack([X_enhanced, flexibility])

    # Aromaticity index
    aromaticity = sol["NumAromaticRings"] / (
        sol["RingCount"] + 1e-8
    )  # Avoid division by zero
    X_enhanced = np.column_stack([X_enhanced, aromaticity])

    # Polynomial versions (helps capture nonlinear relationships if any)
    log_molwt_sq = X_basic[:, 0] ** 2
    X_enhanced = np.column_stack([X_enhanced, log_molwt_sq])

    mol_logp_sq = sol["MolLogP"] ** 2
    X_enhanced = np.column_stack([X_enhanced, mol_logp_sq])

    # Ratio features
    acceptors_idx = 2
    donors_idx = 3
    hbond_ratio = np.where(
        X_basic[:, donors_idx] > 1e-8,  # Avoid division by zero
        X_basic[:, acceptors_idx] / X_basic[:, donors_idx],
        0,
    )
    X_enhanced = np.column_stack([X_enhanced, hbond_ratio])

    # Surface area to volume ratio (approximation)
    surface_volume_ratio = sol["LabuteASA"] / (sol["MolWt"] ** (2 / 3))
    X_enhanced = np.column_stack([X_enhanced, surface_volume_ratio])

    basic_names = ["log_MolWt"] + [f"{prop}/MolWt" for prop in basic_props]
    enhanced_names = (
        basic_names
        + [
            "MolLogP",
            "MolMR/MolWt",
            "TPSA/MolWt",
            "LabuteASA/MolWt",
            "BertzCT/MolWt",
            "BalabanJ",
        ]
        + log_features
        + [
            "HeavyAtom_Valence_Interaction",
            "Lipophilicity_Polarity_Interaction",
            "Total_Rings_per_MolWt",
            "HBond_Capacity_per_MolWt",
            "Flexibility_per_MolWt",
            "Aromaticity_Index",
            "log_MolWt_squared",
            "MolLogP_squared",
            "HBond_Acceptor_Donor_Ratio",
            "Surface_Volume_Ratio",
        ]
    )

    # Find highly correlated features (|correlation| > 0.95).
    corr_matrix = np.corrcoef(X_enhanced.T)

    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > 0.95:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    # Remove one feature from each highly correlated pair
    # (keep the one with higher correlation to solubility)
    features_to_remove = set()
    for i, j, corr in high_corr_pairs:
        corr_i = abs(np.corrcoef(X_enhanced[:, i], sol["Solubility"])[0, 1])
        corr_j = abs(np.corrcoef(X_enhanced[:, j], sol["Solubility"])[0, 1])

        if corr_i < corr_j:
            features_to_remove.add(i)
        else:
            features_to_remove.add(j)

    if features_to_remove:
        X_final = np.delete(X_enhanced, list(features_to_remove), axis=1)
        feature_names = [
            name for i, name in enumerate(enhanced_names) if i not in features_to_remove
        ]
        print(f"Removed {len(features_to_remove)} highly correlated features")
    else:
        X_final = X_enhanced
        feature_names = enhanced_names

    if scale:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_final)

    y = np.array(sol["Solubility"])

    print(f"Final feature set: {len(feature_names)} features")
    print(f"Dataset shape: {X_final.shape}")

    return X_final, y, feature_names