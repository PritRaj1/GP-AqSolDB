import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 16

FIGURE_DIR = "figures"


def make_feature_grid(X, x_idx, y_idx, grid_size=60, pct_low=1, pct_high=99):
    """Create a 2D meshgrid over two features, filling others via nearest neighbor."""
    x1 = np.linspace(
        np.percentile(X[:, x_idx], pct_low),
        np.percentile(X[:, x_idx], pct_high),
        grid_size,
    )
    x2 = np.linspace(
        np.percentile(X[:, y_idx], pct_low),
        np.percentile(X[:, y_idx], pct_high),
        grid_size,
    )
    X1g, X2g = np.meshgrid(x1, x2)
    X_grid = np.zeros((X1g.size, X.shape[1]))
    X_grid[:, x_idx] = X1g.ravel()
    X_grid[:, y_idx] = X2g.ravel()

    subset_size = min(1000, len(X))
    if len(X) > subset_size:
        np.random.seed(42)
        X_sample = X[np.random.choice(len(X), subset_size, replace=False)]
    else:
        X_sample = X

    for i in range(X.shape[1]):
        if i not in (x_idx, y_idx):
            distances = cdist(
                X_grid[:, [x_idx, y_idx]], X_sample[:, [x_idx, y_idx]]
            )
            X_grid[:, i] = X_sample[np.argmin(distances, axis=1), i]

    return X1g, X2g, X_grid


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_predictions(y_test, y_pred, y_std, title, save_path):
    """Scatter plot of predictions vs actual with uncertainty error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))

    threshold = np.mean(y_std) + 2 * np.std(y_std)
    high = y_std > threshold
    low = ~high

    ax.scatter(
        y_test[low], y_pred[low], alpha=0.7, s=30, label="Predictions", color="C0"
    )
    ax.errorbar(
        y_test[low], y_pred[low], yerr=2 * y_std[low],
        fmt="none", alpha=0.3, capsize=2, color="C0",
    )

    if np.any(high):
        ax.scatter(
            y_test[high], y_pred[high], alpha=0.9, s=40,
            label="High Uncertainty", color="red", edgecolor="black", zorder=5,
        )
        ax.errorbar(
            y_test[high], y_pred[high], yerr=2 * y_std[high],
            fmt="none", alpha=0.7, capsize=2, color="red",
        )

    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Actual Solubility")
    ax.set_ylabel("Predicted Solubility")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(X1g, X2g, y_std_grid, x_name, y_name, X_train, x_idx, y_idx, save_path):
    """2D uncertainty heatmap with optional training data overlay."""
    fig, (ax_heat, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))
    cf = ax_heat.contourf(X1g, X2g, y_std_grid, levels=30, cmap="plasma")
    fig.colorbar(cf, ax=ax_heat, label="Predicted Uncertainty")

    ax_heat.set_xlabel(x_name)
    ax_heat.set_ylabel(y_name)
    ax_heat.set_title("2D Uncertainty Heatmap (Top 2 Features)")
    ax_heat.set_xlim(X1g[0].min(), X1g[0].max())
    ax_heat.set_ylim(X2g[:, 0].min(), X2g[:, 0].max())

    if X_train is not None:
        n = min(500, len(X_train))
        if len(X_train) > n:
            np.random.seed(42)
            X_train = X_train[np.random.choice(len(X_train), n, replace=False)]
        ax_heat.scatter(
            X_train[:, x_idx], X_train[:, y_idx], c="lime", marker="x",
            s=18, alpha=0.7, label="Training Data", zorder=5, linewidth=1.2,
        )
        ax_heat.legend()

    ax_hist.hist(y_std_grid.ravel(), bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax_hist.set_xlabel("Predicted Uncertainty")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Distribution of Predicted Uncertainties")

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_surface(X1g, X2g, y_pred_grid, x_idx, y_idx, X, y, feature_names, label, save_path):
    """3D surface plot with data point overlay."""
    n = min(200, len(X))
    if len(X) > n:
        np.random.seed(42)
        idx = np.random.choice(len(X), n, replace=False)
        X_s, y_s = X[idx], y[idx]
    else:
        X_s, y_s = X, y

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X1g, X2g, y_pred_grid, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=label)

    ax.scatter(
        X_s[:, x_idx], X_s[:, y_idx], y_s, c="red", s=20, alpha=0.8, marker="x",
        label=f"Actual Data ({len(X_s)} points)", edgecolor="black", linewidth=0.5,
    )

    ax.set_xlabel(feature_names[x_idx])
    ax.set_ylabel(feature_names[y_idx])
    ax.set_zlabel("Solubility")
    ax.set_title("3D Solubility Surface Plot")
    ax.set_xlim(np.percentile(X[:, x_idx], 5), np.percentile(X[:, x_idx], 95))
    ax.set_ylim(np.percentile(X[:, y_idx], 5), np.percentile(X[:, y_idx], 95))
    ax.legend()
    ax.view_init(elev=10, azim=30)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_length_scales(length_scales, feature_names, sorted_indices, save_path):
    """Horizontal bar chart of kernel length scales."""
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_ls = length_scales[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(sorted_features)), sorted_ls)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Length Scale (Feature Importance)")
    ax.set_title("Kernel Length Scales - Feature Importance")
    ax.grid(True, alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
