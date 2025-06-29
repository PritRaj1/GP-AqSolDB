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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred, alpha=0.7, s=30, label='Predictions')
    axes[0].errorbar(y_test, y_pred, yerr=2*y_std, fmt='none', alpha=0.3, capsize=2)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Solubility')
    axes[0].set_ylabel('Predicted Solubility')
    axes[0].set_title('GP Predictions with 95% Confidence Intervals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(y_std, bins=30, alpha=0.8, edgecolor='black')
    axes[1].axvline(np.mean(y_std), color='red', linestyle='--', label=f'Mean: {np.mean(y_std):.3f}')
    axes[1].set_xlabel('Prediction Uncertainty (σ)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Prediction Uncertainties')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(f'{FIGURE_DIR}/solubility_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()

def kernel_plot(sigmas, feature_names):
    
    length_scales = 1.0 / sigmas
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sorted_indices = np.argsort(length_scales)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_length_scales = length_scales[sorted_indices]
    bars = axes[0, 0].barh(range(len(sorted_features)), sorted_length_scales)
    
    axes[0, 0].set_yticks(range(len(sorted_features)))
    axes[0, 0].set_yticklabels(sorted_features)
    axes[0, 0].set_xlabel('Length Scale (Feature Importance)')
    axes[0, 0].set_title('Kernel Length Scales - Feature Importance')
    axes[0, 0].grid(True, alpha=0.3)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    axes[0, 1].barh(range(len(sorted_features)), 1/sorted_length_scales)
    axes[0, 1].set_yticks(range(len(sorted_features)))
    axes[0, 1].set_yticklabels(sorted_features)
    axes[0, 1].set_xlabel('Sigma (Length Scale)')
    axes[0, 1].set_title('Kernel Sigma Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    importance_ratio = length_scales / np.max(length_scales)
    
    axes[1, 0].barh(range(len(sorted_features)), importance_ratio)
    axes[1, 0].set_yticks(range(len(sorted_features)))
    axes[1, 0].set_yticklabels(sorted_features)
    axes[1, 0].set_xlabel('Relative Importance')
    axes[1, 0].set_title('Relative Feature Importance')
    axes[1, 0].grid(True, alpha=0.3)
    
    cumulative_importance = np.cumsum(sorted_length_scales) / np.sum(length_scales)
    
    axes[1, 1].plot(range(len(sorted_features)), cumulative_importance, 'o-', linewidth=2)
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('Cumulative Importance')
    axes[1, 1].set_title('Cumulative Feature Importance')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80% Importance')
    axes[1, 1].legend()
    
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(f'{FIGURE_DIR}/kernel_interpretability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTop 5 Important Features:")
    
    for i, (feature, importance) in enumerate(zip(sorted_features[:5], sorted_length_scales[:5])):
        print(f"{i+1}. {feature}: {importance:.3f}")

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
    kernel_plot(sigmas, feature_names)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)

    print("Files:")
    print(f"- {FIGURE_DIR}/solubility_uncertainty.png")
    print(f"- {FIGURE_DIR}/solubility_kernel_interpretability.png")
    print(f"- {CONFIG_PATH}")
    print(f"- {SIGMA_PATH}")

if __name__ == "__main__":
    main()
