import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.gp_kan.auto_tune_kan import GPKANAutoTuner, create_optimized_network
from src.gp_kan.normal_dist import NormalDist

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16

CONFIG_PATH = "config/gp_kan.ini"
PARAMS_PATH = "config/gp_kan_params.pkl"
FIGURE_DIR = "figures"


def load_data():
    proplist = [
        'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
        'NumRotatableBonds', 'NumValenceElectrons', 'NumAromaticRings',
        'NumSaturatedRings', 'NumAliphaticRings', 'RingCount'
    ]
    sol = pd.read_csv("data/solubility-dataset.csv")
    sol["NumHAcceptors"] = sol["NumHAcceptors"] + 1

    # FEATURE ENGINEERING - run data_stats.py to see why
    X_basic = np.array([list(sol[prop] / sol['MolWt']) for prop in proplist]).T
    X_basic = np.insert(X_basic, 0, list(np.log(sol['MolWt'])), axis=1)
    X_enhanced = X_basic.copy()  # Extra for better modeling

    # Skewed distributins are log-transformed
    log_features = []
    for i, prop in enumerate(proplist):
        if prop in ['NumSaturatedRings', 'NumAliphaticRings']:  # Highly skewed features - run data_stats.py
            log_feat = np.log1p(sol[prop] / sol['MolWt'])  # log1p is log(1+x) because zeros are problematic
            X_enhanced = np.column_stack([X_enhanced, log_feat])
            log_features.append(f'log_{prop}/MolWt')

    # Iteraction terms for highly correlated features - (correlation ~0.9, see data_stats.py)
    heavy_atom_idx = 1  # After log_MolWt
    valence_idx = 6     # NumValenceElectrons/MolWt
    interaction = (
        X_basic[:, heavy_atom_idx] * X_basic[:, valence_idx]
    )
    X_enhanced = np.column_stack([X_enhanced, interaction])

    # Add polynomial features for important features (log_MolWt squared)
    log_molwt_sq = X_basic[:, 0] ** 2
    X_enhanced = np.column_stack([X_enhanced, log_molwt_sq])

    # Add ratio features (H-bond acceptors to donors ratio)
    acceptors_idx = 2  # NumHAcceptors/MolWt
    donors_idx = 3     # NumHDonors/MolWt
    hbond_ratio = np.where(
        X_basic[:, donors_idx] > 0,
        X_basic[:, acceptors_idx] / X_basic[:, donors_idx],
        0,
    )  # Avoid scalar division
    X_enhanced = np.column_stack([X_enhanced, hbond_ratio])

    # Add molecular complexity features (total rings normalized by molecular weight)
    total_rings = (
        sol['NumAromaticRings'] + sol['NumSaturatedRings'] + sol['NumAliphaticRings']
    ) / sol['MolWt']
    X_enhanced = np.column_stack([X_enhanced, total_rings])

    basic_names = ['log_MolWt'] + [f'{prop}/MolWt' for prop in proplist]
    enhanced_names = basic_names + log_features + [
        'HeavyAtom_Valence_Interaction',
        'log_MolWt_squared',
        'HBond_Acceptor_Donor_Ratio',
        'Total_Rings_per_MolWt'
    ]

    # Remove highly correlated features (only keep one, the other is redundant because ~90% correlated acc. data_stats.py)
    corr_with_target = np.corrcoef(X_enhanced.T, sol['Solubility'])[:-1, -1]
    heavy_atom_corr = abs(corr_with_target[1])  # HeavyAtomCount/MolWt
    valence_corr = abs(corr_with_target[6])     # NumValenceElectrons/MolWt

    if heavy_atom_corr < valence_corr:
        X_final = np.delete(X_enhanced, 1, axis=1)
        feature_names = [name for i, name in enumerate(enhanced_names) if i != 1]
    else:
        X_final = np.delete(X_enhanced, 6, axis=1)
        feature_names = [name for i, name in enumerate(enhanced_names) if i != 6]

    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_final)

    y = np.array(sol['Solubility'])

    print("Feature engineering summary:")
    print(f"  Original features: {len(basic_names)}")
    print(f"  Enhanced features: {len(enhanced_names)}")
    print(f"  Final features (after correlation removal): {len(feature_names)}")
    print("  Features removed due to high correlation: 1")
    print("  Features standardized: Yes")

    return X_final, y, feature_names


def uncertainty_plot(gp_kan, X_train, X_test, y_test):
    X_test_mean = X_test
    X_test_var = np.zeros_like(X_test)
    X_test_dist = NormalDist(X_test_mean, X_test_var)

    # Forward pass through GP-KAN
    output_dist = gp_kan.forward(X_test_dist)
    y_pred = output_dist.mean.flatten()
    y_std = np.sqrt(np.maximum(output_dist.var.flatten(), 1e-6))  # Ensure positive variance

    fig, ax = plt.subplots(figsize=(8, 6))

    # 2 std away from mean uncertainty get differently coloured
    threshold = np.mean(y_std) + 2 * np.std(y_std)
    high_uncertainty = y_std > threshold
    low_uncertainty = ~high_uncertainty

    ax.scatter(y_test[low_uncertainty], y_pred[low_uncertainty], alpha=0.7, s=30, label='Predictions', color='C0')
    ax.errorbar(y_test[low_uncertainty], y_pred[low_uncertainty], yerr=2*y_std[low_uncertainty], fmt='none', alpha=0.3, capsize=2, color='C0')

    if np.any(high_uncertainty):
        ax.scatter(
            y_test[high_uncertainty],
            y_pred[high_uncertainty],
            alpha=0.9,
            s=40,
            label='High Uncertainty',
            color='red',
            edgecolor='black',
            zorder=5,
        )
        ax.errorbar(
            y_test[high_uncertainty],
            y_pred[high_uncertainty],
            yerr=2*y_std[high_uncertainty],
            fmt='none',
            alpha=0.7,
            capsize=2,
            color='red',
        )

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        lw=2,
    )
    ax.set_xlabel('Actual Solubility')
    ax.set_ylabel('Predicted Solubility')
    ax.set_title('GP-KAN Predictions with 95% Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(f'{FIGURE_DIR}/solubility_uncertainty_kan.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_uncertainty_heatmap(gp_kan, X, y, feature_names, X_train, save_path):
    first_layer = gp_kan.layers[0]
    layer_variances = np.var(first_layer.get_z(), axis=(1, 2))  # Variance across inducing points for each input
    sorted_indices = np.argsort(layer_variances)[::-1]
    top2_idx = sorted_indices[:2]
    x_idx, y_idx = top2_idx[0], top2_idx[1]
    x_name, y_name = feature_names[x_idx], feature_names[y_idx]

    x1_min, x1_max = np.percentile(X[:, x_idx], 1), np.percentile(X[:, x_idx], 99)
    x2_min, x2_max = np.percentile(X[:, y_idx], 1), np.percentile(X[:, y_idx], 99)
    x1 = np.linspace(x1_min, x1_max, 60)
    x2 = np.linspace(x2_min, x2_max, 60)
    X1g, X2g = np.meshgrid(x1, x2)
    X_grid = np.zeros((X1g.size, X.shape[1]))
    X_grid[:, x_idx] = X1g.ravel()
    X_grid[:, y_idx] = X2g.ravel()

    subset_size = min(1000, len(X))
    if len(X) > subset_size:
        np.random.seed(42)
        sample_indices = np.random.choice(len(X), subset_size, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X

    # For each grid point, use the closest real data point for other dimensions
    for i in range(X.shape[1]):
        if i not in top2_idx:
            grid_2d = X_grid[:, [x_idx, y_idx]]
            sample_2d = X_sample[:, [x_idx, y_idx]]
            distances = cdist(grid_2d, sample_2d)
            closest_indices = np.argmin(distances, axis=1)
            X_grid[:, i] = X_sample[closest_indices, i]

    X_grid_mean = X_grid
    X_grid_var = np.zeros_like(X_grid)
    X_grid_dist = NormalDist(X_grid_mean, X_grid_var)
    output_dist = gp_kan.forward(X_grid_dist)
    y_std_grid = np.sqrt(np.maximum(output_dist.var.flatten(), 1e-6))
    y_std_grid = y_std_grid.reshape(X1g.shape)

    fig, (ax_heat, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))
    cf = ax_heat.contourf(X1g, X2g, y_std_grid, levels=30, cmap='plasma')
    fig.colorbar(cf, ax=ax_heat, label='Predicted Uncertainty')

    ax_heat.set_xlabel(x_name)
    ax_heat.set_ylabel(y_name)
    ax_heat.set_title('2D Uncertainty Heatmap (Top 2 Features)')
    ax_heat.set_xlim(x1.min(), x1.max())
    ax_heat.set_ylim(x2.min(), x2.max())

    if X_train is not None:
        train_subset_size = min(500, len(X_train))
        if len(X_train) > train_subset_size:
            np.random.seed(42)
            train_sample_indices = np.random.choice(
                len(X_train), train_subset_size, replace=False
            )
            X_train_sample = X_train[train_sample_indices]
        else:
            X_train_sample = X_train

        ax_heat.scatter(
            X_train_sample[:, x_idx],
            X_train_sample[:, y_idx],
            c='lime',
            marker='x',
            s=18,
            alpha=0.7,
            label='Training Data',
            zorder=5,
            linewidth=1.2,
        )
        ax_heat.legend()

    ax_hist.hist(y_std_grid.ravel(), bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel('Predicted Uncertainty')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Distribution of Predicted Uncertainties')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def surface_plot(gp_kan, X, y, feature_names, save_path):
    first_layer = gp_kan.layers[0]
    layer_variances = np.var(first_layer.get_z(), axis=(1, 2))
    sorted_indices = np.argsort(layer_variances)[::-1]
    x_idx, y_idx = sorted_indices[0], sorted_indices[1]

    x1_min, x1_max = np.percentile(X[:, x_idx], 5), np.percentile(X[:, x_idx], 95)
    x2_min, x2_max = np.percentile(X[:, y_idx], 5), np.percentile(X[:, y_idx], 95)
    x1 = np.linspace(x1_min, x1_max, 80)
    x2 = np.linspace(x2_min, x2_max, 80)
    X1g, X2g = np.meshgrid(x1, x2)
    X_grid = np.zeros((X1g.size, X.shape[1]))
    X_grid[:, x_idx] = X1g.ravel()
    X_grid[:, y_idx] = X2g.ravel()

    subset_size = min(1000, len(X))
    if len(X) > subset_size:
        np.random.seed(42)
        sample_indices = np.random.choice(len(X), subset_size, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X

    for i in range(X.shape[1]):
        if i not in (x_idx, y_idx):
            grid_2d = X_grid[:, [x_idx, y_idx]]
            sample_2d = X_sample[:, [x_idx, y_idx]]
            distances = cdist(grid_2d, sample_2d)
            closest_indices = np.argmin(distances, axis=1)
            X_grid[:, i] = X_sample[closest_indices, i]

    X_grid_mean = X_grid
    X_grid_var = np.zeros_like(X_grid)
    X_grid_dist = NormalDist(X_grid_mean, X_grid_var)
    output_dist = gp_kan.forward(X_grid_dist)
    y_pred_grid = output_dist.mean.flatten()
    y_pred_grid = y_pred_grid.reshape(X1g.shape)

    n_data_points = min(200, len(X))
    if len(X) > n_data_points:
        np.random.seed(42)
        sample_indices = np.random.choice(len(X), n_data_points, replace=False)
        X_sample = X[sample_indices]
        y_sample = y[sample_indices]
    else:
        X_sample = X
        y_sample = y

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X1g, X2g, y_pred_grid, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='GP-KAN Solubility')

    ax.scatter(
        X_sample[:, x_idx],
        X_sample[:, y_idx],
        y_sample,
        c='red',
        s=20,
        alpha=0.8,
        marker='x',
        label=f'Actual Data ({len(X_sample)} points)',
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_xlabel(feature_names[x_idx])
    ax.set_ylabel(feature_names[y_idx])
    ax.set_zlabel('Solubility')
    ax.set_title('3D Solubility Surface Plot')
    ax.set_xlim(
        np.percentile(X[:, x_idx], 5), np.percentile(X[:, x_idx], 95)
    )
    ax.set_ylim(
        np.percentile(X[:, y_idx], 5), np.percentile(X[:, y_idx], 95)
    )
    ax.legend()

    ax.view_init(elev=10, azim=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("="*80)
    print("Tuning/training GP-KAN on AqSolDB")
    print("="*80)

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)

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
            metric='MSE',
            n_jobs=4,
            use_gpu=True,
            max_hidden_layers=4,
            max_hidden_size=30,
            num_epochs=20,
            pretrain_iters=10,
            patience=100,
            sampler='tpe',
        )
        tuner.optimize(n_trials=200)
        gp_kan = create_optimized_network(CONFIG_PATH, PARAMS_PATH)

    print("Training GP-KAN network...")
    gp_kan.train(
        X_train, y_train,
        X_test, y_test,
        num_epochs=50,
        patience=100,
        pretrain_iters=20,
    )

    X_test_mean = X_test
    X_test_var = np.zeros_like(X_test)
    X_test_dist = NormalDist(X_test_mean, X_test_var)
    test_output = gp_kan.forward(X_test_dist)
    y_pred = test_output.mean.flatten()
    y_std = np.sqrt(np.maximum(test_output.var.flatten(), 1e-6))

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Mean uncertainty: {np.mean(y_std):.4f}")

    uncertainty_plot(gp_kan, X_train, X_test, y_test)
    plot_uncertainty_heatmap(gp_kan, X, y, feature_names, X_train, f'{FIGURE_DIR}/kernel_uncertainty_heatmap_kan.png')
    surface_plot(gp_kan, X, y, feature_names, f'{FIGURE_DIR}/solubility_surface_kan.png')
    gp_kan.save_fig(f'{FIGURE_DIR}/gp_kan_architecture.png', max_neurons_per_layer=3)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)

    print("Files:")
    print(f"- {FIGURE_DIR}/solubility_uncertainty_kan.png")
    print(f"- {FIGURE_DIR}/kernel_uncertainty_heatmap_kan.png")
    print(f"- {FIGURE_DIR}/solubility_surface_kan.png")
    print(f"- {FIGURE_DIR}/gp_kan_architecture.png")
    print(f"- {CONFIG_PATH}")
    print(f"- {PARAMS_PATH}")


if __name__ == "__main__":
    main()
