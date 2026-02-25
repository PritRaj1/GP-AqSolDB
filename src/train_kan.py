import os
from configparser import ConfigParser

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.core.models.gp_kan import NormalDist
from src.optimization import GPKANAutoTuner, create_optimized_network
from src.plotting import (
    FIGURE_DIR,
    make_feature_grid,
    plot_heatmap,
    plot_predictions,
    plot_surface,
)
from src.utils.data_utils import load_aqsol_data

CONFIG_PATH = "config/gp_kan.ini"
PARAMS_PATH = "config/gp_kan_params.pkl"


def _predict_kan(gp_kan, X):
    """Forward pass through GP-KAN, returning (mean, std)."""
    dist = NormalDist(X, np.zeros_like(X))
    out = gp_kan.forward(dist)
    mean = out.mean.flatten()
    std = np.sqrt(np.maximum(out.var.flatten(), 1e-6))
    return mean, std


def _top2_from_layers(gp_kan):
    first_layer = gp_kan.layers[0]
    variances = np.var(first_layer.get_z(), axis=(1, 2))
    sorted_indices = np.argsort(variances)[::-1]
    return sorted_indices[:2]


def main():
    print("=" * 80)
    print("Tuning/training GP-KAN on AqSolDB")
    print("=" * 80)

    X_df, y, feature_names, scaler = load_aqsol_data(
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

    if os.path.exists(CONFIG_PATH) and os.path.exists(PARAMS_PATH):
        print("Loading previously optimized hyperparameters...")
        config = ConfigParser()
        config.read(CONFIG_PATH)
        gp_kan = create_optimized_network(CONFIG_PATH, PARAMS_PATH)

    else:
        print("No optimized hyperparameters found. Running auto-tuning...")
        tuner = GPKANAutoTuner(
            X_train,
            y_train,
            config_path=CONFIG_PATH,
            params_save_path=PARAMS_PATH,
            metric="MSE",
            use_gpu=True,
            max_hidden_layers=4,
            max_hidden_size=100,
            num_epochs=60,
            pretrain_iters=30,
            patience=100,
            sampler="tpe",
            available_acts=["NormaliseGaussian", "ReduceSumGaussian", "None"],
        )
        tuner.optimize(n_trials=200)
        gp_kan = create_optimized_network(CONFIG_PATH, PARAMS_PATH)

    print("Training GP-KAN network...")
    gp_kan.train(
        X_train,
        y_train,
        X_test,
        y_test,
        num_epochs=60,
        patience=300,
        pretrain_iters=30,
    )

    y_pred, y_std = _predict_kan(gp_kan, X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Mean uncertainty: {np.mean(y_std):.4f}")

    # Plots
    top2 = _top2_from_layers(gp_kan)
    x_idx, y_idx = top2[0], top2[1]

    plot_predictions(
        y_test,
        y_pred,
        y_std,
        "GP-KAN Predictions with 95% Confidence Intervals",
        f"{FIGURE_DIR}/solubility_uncertainty_kan.png",
    )

    X1g, X2g, X_grid = make_feature_grid(X, x_idx, y_idx, grid_size=60)
    _, y_std_grid = _predict_kan(gp_kan, X_grid)
    plot_heatmap(
        X1g,
        X2g,
        y_std_grid.reshape(X1g.shape),
        feature_names[x_idx],
        feature_names[y_idx],
        X_train,
        x_idx,
        y_idx,
        f"{FIGURE_DIR}/kernel_uncertainty_heatmap_kan.png",
    )

    X1g_s, X2g_s, X_grid_s = make_feature_grid(
        X, x_idx, y_idx, grid_size=80, pct_low=5, pct_high=95
    )
    y_pred_grid, _ = _predict_kan(gp_kan, X_grid_s)
    plot_surface(
        X1g_s,
        X2g_s,
        y_pred_grid.reshape(X1g_s.shape),
        x_idx,
        y_idx,
        X,
        y,
        feature_names,
        "GP-KAN Solubility",
        f"{FIGURE_DIR}/solubility_surface_kan.png",
    )

    gp_kan.save_fig(f"{FIGURE_DIR}/gp_kan_architecture.png", max_neurons_per_layer=3)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print("Files:")
    print(f"- {FIGURE_DIR}/solubility_uncertainty_kan.png")
    print(f"- {FIGURE_DIR}/kernel_uncertainty_heatmap_kan.png")
    print(f"- {FIGURE_DIR}/solubility_surface_kan.png")
    print(f"- {FIGURE_DIR}/gp_kan_architecture.png")
    print(f"- {CONFIG_PATH}")
    print(f"- {PARAMS_PATH}")
