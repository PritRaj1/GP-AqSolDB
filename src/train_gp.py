import glob
import os
import pickle
from configparser import ConfigParser
from typing import Any, List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.core.models import GP
from src.optimization import GPAutoTuner
from src.plotting import (
    FIGURE_DIR,
    make_feature_grid,
    plot_heatmap,
    plot_length_scales,
    plot_predictions,
    plot_surface,
)
from src.utils.data_utils import infer_defaults, load_aqsol_data

CONFIG_PATH = "config/gp.ini"
SIGMA_PATH = "config/gp_sigmas.pkl"


def _top2_from_sigmas(sigmas: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    length_scales = 1.0 / sigmas
    sorted_indices = np.argsort(length_scales)[::-1]
    return sorted_indices[:2], length_scales, sorted_indices


def learning_evolution(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Any,
    config: ConfigParser,
    sigmas: np.ndarray,
    full_X: np.ndarray,
    n_init: int = 1,
    n_steps: int = 300,
    gif_path: str = f"{FIGURE_DIR}/learning_evolution.gif",
) -> None:
    np.random.seed(42)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    frames: List[np.ndarray] = []
    gif_config = ConfigParser()

    for section in config.sections():
        gif_config[section] = dict(config[section])

    top2_idx, _, _ = _top2_from_sigmas(sigmas)
    x_idx, y_idx = top2_idx[0], top2_idx[1]
    x_name, y_name = feature_names[x_idx], feature_names[y_idx]

    grid_size = 60
    x1 = np.linspace(
        np.percentile(X[:, x_idx], 1), np.percentile(X[:, x_idx], 99), grid_size
    )
    x2 = np.linspace(
        np.percentile(X[:, y_idx], 1), np.percentile(X[:, y_idx], 99), grid_size
    )
    X1g, X2g = np.meshgrid(x1, x2)
    X_grid = np.zeros((X1g.size, X.shape[1]))
    X_grid[:, x_idx] = X1g.ravel()
    X_grid[:, y_idx] = X2g.ravel()
    for i in range(X.shape[1]):
        if i not in (x_idx, y_idx):
            X_grid[:, i] = np.mean(X[:, i])

    # Pre-scan to determine axis limits
    pool_idx = np.arange(len(X))
    init_idx = np.random.choice(pool_idx, size=n_init, replace=False)
    temp_train_idx = list(init_idx)
    temp_pool_idx = np.setdiff1d(np.arange(len(X)), temp_train_idx)
    temp_mean_uncertainties = []
    grid_unc_min, grid_unc_max = np.inf, -np.inf

    for _ in range(n_steps):
        if len(temp_pool_idx) == 0:
            break
        X_train, y_train = X[temp_train_idx], y[temp_train_idx]
        gp = GP(gif_config, sigmas)
        gp.fit(X_train, y_train)
        _, y_std_grid = gp.predict(X_grid, return_std=True)
        grid_unc_min = min(grid_unc_min, np.min(y_std_grid))
        grid_unc_max = max(grid_unc_max, np.max(y_std_grid))
        X_pool = X[temp_pool_idx]
        _, y_std_pool = gp.predict(X_pool, return_std=True)
        temp_mean_uncertainties.append(np.mean(y_std_pool))

        if len(temp_pool_idx) > 0:
            next_idx = temp_pool_idx[np.argmax(y_std_pool)]
            temp_train_idx.append(next_idx)
            temp_pool_idx = np.setdiff1d(temp_pool_idx, [next_idx])

    if (
        len(temp_mean_uncertainties) == 0
        or not np.isfinite(temp_mean_uncertainties).all()
    ):
        unc_ylim = (0, 1)
        grid_unc_min, grid_unc_max = 0, 1

    else:
        unc_ylim = (
            min(temp_mean_uncertainties) * 0.95,
            max(temp_mean_uncertainties) * 1.05,
        )

    del temp_train_idx, temp_pool_idx, temp_mean_uncertainties

    # Actual run with frame generation
    mean_uncertainties = []
    pool_idx = np.arange(len(X))
    init_idx = np.random.choice(pool_idx, size=n_init, replace=False)
    train_idx = list(init_idx)
    pool_idx = np.setdiff1d(pool_idx, train_idx)

    for step in range(n_steps):
        X_train, y_train = X[train_idx], y[train_idx]
        X_pool = X[pool_idx]

        gp = GP(gif_config, sigmas)
        gp.fit(X_train, y_train)

        _, y_std_pool = gp.predict(X_pool, return_std=True)
        mean_uncertainties.append(np.mean(y_std_pool))

        if len(pool_idx) > 0:
            next_idx = pool_idx[np.argmax(y_std_pool)]
            train_idx.append(next_idx)
            pool_idx = np.setdiff1d(pool_idx, [next_idx])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(
            full_X[:, x_idx],
            full_X[:, y_idx],
            c="black",
            s=40,
            marker="x",
            label="All Data",
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            X[train_idx, x_idx],
            X[train_idx, y_idx],
            c="lime",
            s=40,
            marker="x",
            label="Train",
            alpha=0.9,
            zorder=3,
        )

        if len(train_idx) > n_init:
            last = train_idx[-1]
            ax.scatter(
                X[last, x_idx],
                X[last, y_idx],
                c="red",
                s=120,
                marker="x",
                label="Newly Added",
                edgecolor="black",
                linewidth=3,
                zorder=4,
            )

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"Data Seen {step + 1}/{n_steps}")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(np.percentile(X[:, x_idx], 1), np.percentile(X[:, x_idx], 99))
        ax.set_ylim(np.percentile(X[:, y_idx], 1), np.percentile(X[:, y_idx], 99))

        _, y_std_grid = gp.predict(X_grid, return_std=True)
        y_std_grid = y_std_grid.reshape(X1g.shape)
        levels = np.linspace(grid_unc_min, grid_unc_max, 40)
        im = ax.contourf(
            X1g,
            X2g,
            y_std_grid,
            levels=levels,
            cmap="plasma",
            alpha=0.7,
            vmin=grid_unc_min,
            vmax=grid_unc_max,
        )
        fig.colorbar(im, ax=ax, label="Uncertainty")

        ax2 = axes[1]
        ax2.plot(np.arange(1, step + 2), mean_uncertainties, "-o", color="purple")
        ax2.set_xlabel("Data Seen")
        ax2.set_ylabel("Mean Predictive Variance")
        ax2.set_title("Uncertainty Reduction")
        ax2.set_xlim(1, n_steps)
        ax2.set_ylim(unc_ylim)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        frame_path = f"{FIGURE_DIR}/_al_frame_{step:03d}.png"
        plt.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        frames.append(imageio.v2.imread(frame_path))
        os.remove(frame_path)

    imageio.mimsave(gif_path, frames, duration=3)  # type: ignore[arg-type]
    print(f"Active learning GIF saved to {gif_path}")

    for f in glob.glob(f"{FIGURE_DIR}/_al_frame_*.png"):
        try:
            os.remove(f)
        except Exception:
            pass


def main() -> None:
    print("=" * 80)
    print("Tuning/training on AqSolDB")
    print("=" * 80)

    X_df, y, feature_names, scaler = load_aqsol_data(  # type: ignore[misc]
        scale=False,
        return_frame=True,
        return_scaler=True,
    )
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.1,
        random_state=69,
    )
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    X = scaler.transform(X_df)

    if os.path.exists(CONFIG_PATH) and os.path.exists(SIGMA_PATH):
        print("Loading previously optimized hyperparameters...")
        config = ConfigParser()
        config.read(CONFIG_PATH)
        with open(SIGMA_PATH, "rb") as f:
            sigmas = np.array(pickle.load(f))

    else:
        print("No optimized hyperparameters found. Running auto-tuning...")
        defaults = infer_defaults(X_train)
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=CONFIG_PATH,
            sigma_save_path=SIGMA_PATH,
            gp_mode="dense",
            metric="MSE",
            use_gpu=True,
            sampler="tpe",
            max_samples=4000,
            data_defaults=defaults,
        )
        tuner.optimize(n_trials=1000)
        config, sigmas = tuner.load_optimized_parameters()

    gp = GP(config, sigmas)
    gp.fit(X_train, y_train)

    y_pred_raw, y_std_raw = gp.predict(X_test, return_std=True)
    y_pred = np.asarray(y_pred_raw)
    y_std = np.asarray(y_std_raw)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Mean uncertainty: {np.mean(y_std):.4f}")

    # Plots
    top2, length_scales, sorted_indices = _top2_from_sigmas(sigmas)
    x_idx, y_idx = top2[0], top2[1]

    plot_predictions(
        y_test,
        y_pred,
        y_std,
        "GP Predictions with 95% Confidence Intervals",
        f"{FIGURE_DIR}/solubility_uncertainty.png",
    )
    plot_length_scales(
        length_scales,
        feature_names,
        sorted_indices,
        f"{FIGURE_DIR}/kernel_length_scales.png",
    )

    X1g, X2g, X_grid = make_feature_grid(X, x_idx, y_idx, grid_size=60)
    _, y_std_grid_raw = gp.predict(X_grid, return_std=True)
    y_std_grid = np.asarray(y_std_grid_raw)
    plot_heatmap(
        X1g,
        X2g,
        y_std_grid.reshape(X1g.shape),
        feature_names[x_idx],
        feature_names[y_idx],
        X_train,
        x_idx,
        y_idx,
        f"{FIGURE_DIR}/kernel_uncertainty_heatmap.png",
    )

    X1g_s, X2g_s, X_grid_s = make_feature_grid(
        X, x_idx, y_idx, grid_size=80, pct_low=5, pct_high=95
    )
    y_pred_grid = np.asarray(gp.predict(X_grid_s)).reshape(X1g_s.shape)
    plot_surface(
        X1g_s,
        X2g_s,
        y_pred_grid,
        x_idx,
        y_idx,
        X,
        y,
        feature_names,
        "GP Solubility",
        f"{FIGURE_DIR}/solubility_surface.png",
    )

    subset_size = min(500, len(X))
    subset_idx = np.random.choice(len(X), subset_size, replace=False)
    learning_evolution(
        X[subset_idx, :2],
        y[subset_idx],
        feature_names[:2],
        config,
        sigmas[:2],
        full_X=X,
        gif_path=f"{FIGURE_DIR}/learning_evolution.gif",
    )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print("Files:")
    print(f"- {FIGURE_DIR}/solubility_uncertainty.png")
    print(f"- {FIGURE_DIR}/kernel_length_scales.png")
    print(f"- {FIGURE_DIR}/kernel_uncertainty_heatmap.png")
    print(f"- {FIGURE_DIR}/solubility_surface.png")
    print(f"- {FIGURE_DIR}/learning_evolution.gif")
    print(f"- {CONFIG_PATH}")
    print(f"- {SIGMA_PATH}")
