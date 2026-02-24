import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("data/solubility-dataset.csv")

    proplist = [
        "HeavyAtomCount", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
        "NumRotatableBonds", "NumValenceElectrons", "NumAromaticRings",
        "NumSaturatedRings", "NumAliphaticRings", "RingCount",
    ]

    feature_names = ["log_MolWt"] + [f"{prop}/MolWt" for prop in proplist]
    X = np.array([list(df[prop] / df["MolWt"]) for prop in proplist]).T
    X = np.insert(X, 0, list(np.log(df["MolWt"])), axis=1)
    y = np.array(df["Solubility"])

    print("=== DATA ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Target mean: {y.mean():.3f}")
    print(f"Target std: {y.std():.3f}")

    print("\n=== FEATURE STATS ===")
    for i, name in enumerate(feature_names):
        print(f"{name}:")
        print(f"  Range: [{X[:, i].min():.3f}, {X[:, i].max():.3f}]")
        print(f"  Mean: {X[:, i].mean():.3f}")
        print(f"  Std: {X[:, i].std():.3f}")
        print(f"  Skewness: {pd.Series(X[:, i]).skew():.3f}")

    print("\n=== CORRELATION ===")
    corr_matrix = np.corrcoef(X.T)
    target_corrs = np.corrcoef(X.T, y)[:-1, -1]
    print("Correlations with target:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {target_corrs[i]:.3f}")

    print("\nFeature correlations (absolute > 0.8):")
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.8:
                print(f"  {feature_names[i]} <-> {feature_names[j]}: "
                      f"{corr_matrix[i, j]:.3f}")

    print("\n=== OUTLIERS ===")
    for i, name in enumerate(feature_names):
        Q1 = np.percentile(X[:, i], 25)
        Q3 = np.percentile(X[:, i], 75)
        IQR = Q3 - Q1
        outliers = np.sum((X[:, i] < Q1 - 1.5 * IQR) | (X[:, i] > Q3 + 1.5 * IQR))
        print(f"{name}: {outliers} outliers ({outliers / len(X) * 100:.1f}%)")

    print("\n=== DATA QUALITY ===")
    print(f"Missing values in features: {np.sum(np.isnan(X))}")
    print(f"Missing values in target: {np.sum(np.isnan(y))}")
    print(f"Infinite values in features: {np.sum(np.isinf(X))}")
    print(f"Infinite values in target: {np.sum(np.isinf(y))}")
