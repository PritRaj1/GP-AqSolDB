import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.auto_tune import GPAutoTuner
from src.gp import GP
from src.kernels import configure_parallel_settings

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

CONFIG_PATH = "config/solubility_gp.ini"
SIGMA_PATH = "config/solubility_sigmas.pkl"
FIGURE_DIR = "figures"

def load_data():
    proplist = [
        'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
        'NumRotatableBonds', 'NumValenceElectrons', 'NumAromaticRings',
        'NumSaturatedRings', 'NumAliphaticRings', 'RingCount'
    ]
    sol = pd.read_csv("data/solubility-dataset.csv")
    sol["NumHAcceptors"] = sol["NumHAcceptors"] + 1
    feature_names = ['log_MolWt'] + [f'{prop}/MolWt' for prop in proplist]
    X = np.array([list(sol[prop] / sol['MolWt']) for prop in proplist]).T
    X = np.insert(X, 0, list(np.log(sol['MolWt'])), axis=1)
    y = np.array(sol['Solubility'])
    return X, y, feature_names

def uncertainty_plot(gp, X_train, X_test, y_test):
    y_pred, y_std = gp.predict(X_test, return_std=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1.5x median uncertainty get differently coloured
    threshold = np.median(y_std) * 1.5
    high_uncertainty = y_std > threshold
    low_uncertainty = ~high_uncertainty

    ax.scatter(y_test[low_uncertainty], y_pred[low_uncertainty], alpha=0.7, s=30, label='Predictions', color='C0')
    ax.errorbar(y_test[low_uncertainty], y_pred[low_uncertainty], yerr=2*y_std[low_uncertainty], fmt='none', alpha=0.3, capsize=2, color='C0')
    if np.any(high_uncertainty):
        ax.scatter(y_test[high_uncertainty], y_pred[high_uncertainty], alpha=0.9, s=40, label='High Uncertainty', color='red', edgecolor='black', zorder=5)
        ax.errorbar(y_test[high_uncertainty], y_pred[high_uncertainty], yerr=2*y_std[high_uncertainty], fmt='none', alpha=0.7, capsize=2, color='red')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Solubility')
    ax.set_ylabel('Predicted Solubility')
    ax.set_title('GP Predictions with 95% Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(f'{FIGURE_DIR}/solubility_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_length_scales(length_scales, feature_names, sorted_indices, save_path):
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_length_scales = length_scales[sorted_indices]
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(sorted_features)), sorted_length_scales)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Length Scale (Feature Importance)')
    ax.set_title('Kernel Length Scales - Feature Importance')
    ax.grid(True, alpha=0.3)
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sigmas(length_scales, feature_names, sorted_indices, save_path):
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_length_scales = length_scales[sorted_indices]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_features)), 1/sorted_length_scales, color='lightcoral')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Sigma (Length Scale)')
    ax.set_title('Kernel Sigma Values')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_relative_importance(length_scales, feature_names, sorted_indices, save_path):
    sorted_features = [feature_names[i] for i in sorted_indices]
    importance_ratio = length_scales / np.max(length_scales)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_features)), importance_ratio[sorted_indices], color='lightcoral')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Relative Importance')
    ax.set_title('Relative Feature Importance')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_heatmap(gp, X, y, feature_names, sigmas, X_train, save_path):
    length_scales = 1.0 / sigmas
    sorted_indices = np.argsort(length_scales)[::-1]
    top2_idx = sorted_indices[:2]
    
    x1_name, x2_name = feature_names[top2_idx[0]], feature_names[top2_idx[1]]
    x1 = np.linspace(np.percentile(X[:, top2_idx[0]], 1), np.percentile(X[:, top2_idx[0]], 99), 60)
    x2 = np.linspace(np.percentile(X[:, top2_idx[1]], 1), np.percentile(X[:, top2_idx[1]], 99), 60)
    X1g, X2g = np.meshgrid(x1, x2)
    X_grid = np.zeros((X1g.size, X.shape[1]))
    X_grid[:, top2_idx[0]] = X1g.ravel()
    X_grid[:, top2_idx[1]] = X2g.ravel()
    
    for i in range(X.shape[1]):
        if i not in top2_idx:
            X_grid[:, i] = np.mean(X[:, i])
    _, y_std_grid = gp.predict(X_grid, return_std=True)
    y_std_grid = y_std_grid.reshape(X1g.shape)
    
    fig, (ax_heat, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))
    cf = ax_heat.contourf(X1g, X2g, y_std_grid, levels=30, cmap='plasma')
    fig.colorbar(cf, ax=ax_heat, label='Predicted Uncertainty')
    
    ax_heat.set_xlabel(x1_name)
    ax_heat.set_ylabel(x2_name)
    ax_heat.set_title('2D Uncertainty Heatmap (Top 2 Features)')
    ax_heat.set_xlim(x1.min(), x1.max())
    ax_heat.set_ylim(x2.min(), x2.max())
    
    if X_train is not None:
        ax_heat.scatter(X_train[:, top2_idx[0]], X_train[:, top2_idx[1]], 
                       c='lime', marker='x', s=18, alpha=0.7, 
                       label='Training Data', zorder=5, linewidth=1.2)
        ax_heat.legend()
    
    ax_hist.hist(y_std_grid.ravel(), bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel('Predicted Uncertainty')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Distribution of Predicted Uncertainties (Grid)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*80)
    print("Tuning/training on AqSolDB")
    print("="*80)

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    configure_parallel_settings(use_parallel=True, n_jobs=4, use_gpu=True)
    
    # Check if optimal parameters already exist
    if os.path.exists(CONFIG_PATH) and os.path.exists(SIGMA_PATH):
        print("Loading previously optimized hyperparameters...")
        from src.auto_tune import load_sigmas_from_file
        from configparser import ConfigParser
        config = ConfigParser()
        config.read(CONFIG_PATH)
        sigmas = load_sigmas_from_file(SIGMA_PATH)
    else:
        print("No optimized hyperparameters found. Running auto-tuning...")
        tuner = GPAutoTuner(X_train, y_train, config_path=CONFIG_PATH, sigma_save_path=SIGMA_PATH)
        tuner.optimize(n_trials=50)
        config, sigmas = tuner.load_optimized_parameters()

    gp = GP(config, sigmas)
    gp.fit(X_train, y_train)

    y_pred, y_std = gp.predict(X_test, return_std=True)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Mean uncertainty: {np.mean(y_std):.4f}")

    uncertainty_plot(gp, X_train, X_test, y_test)
    length_scales = 1.0 / sigmas
    sorted_indices = np.argsort(length_scales)[::-1]
    plot_length_scales(length_scales, feature_names, sorted_indices, f'{FIGURE_DIR}/kernel_length_scales.png')
    plot_sigmas(length_scales, feature_names, sorted_indices, f'{FIGURE_DIR}/kernel_sigmas.png')
    plot_relative_importance(length_scales, feature_names, sorted_indices, f'{FIGURE_DIR}/kernel_relative_importance.png')
    plot_uncertainty_heatmap(gp, X, y, feature_names, sigmas, X_train, f'{FIGURE_DIR}/kernel_uncertainty_heatmap.png')

    print("\n" + "="*80)
    print("Done!")
    print("="*80)

    print("Files:")
    print(f"- {FIGURE_DIR}/solubility_uncertainty.png")
    print(f"- {FIGURE_DIR}/kernel_length_scales.png")
    print(f"- {FIGURE_DIR}/kernel_sigmas.png")
    print(f"- {FIGURE_DIR}/kernel_relative_importance.png")
    print(f"- {FIGURE_DIR}/kernel_uncertainty_heatmap.png")
    print(f"- {CONFIG_PATH}")
    print(f"- {SIGMA_PATH}")

if __name__ == "__main__":
    main()
