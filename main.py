import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from configparser import ConfigParser
import imageio
import glob
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from src.multivar_gp.auto_tune import GPAutoTuner
from src.multivar_gp.gp import GP
from src.multivar_gp.kernels import configure_parallel_settings

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16

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
    
    ## FEATURE ENGINEERING - run data_stats.py to see why
    X_basic = np.array([list(sol[prop] / sol['MolWt']) for prop in proplist]).T
    X_basic = np.insert(X_basic, 0, list(np.log(sol['MolWt'])), axis=1)    
    X_enhanced = X_basic.copy() # Extra for better modeling
    
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
    interaction = X_basic[:, heavy_atom_idx] * X_basic[:, valence_idx]
    X_enhanced = np.column_stack([X_enhanced, interaction])
    
    # Add polynomial features for important features (log_MolWt squared)
    log_molwt_sq = X_basic[:, 0] ** 2
    X_enhanced = np.column_stack([X_enhanced, log_molwt_sq])
    
    # Add ratio features (H-bond acceptors to donors ratio)
    acceptors_idx = 2  # NumHAcceptors/MolWt
    donors_idx = 3     # NumHDonors/MolWt
    hbond_ratio = np.where(X_basic[:, donors_idx] > 0, 
                          X_basic[:, acceptors_idx] / X_basic[:, donors_idx], 
                          0)  # Avoid scalar division
    X_enhanced = np.column_stack([X_enhanced, hbond_ratio])
    
    # Add molecular complexity features (total rings normalized by molecular weight)
    total_rings = (sol['NumAromaticRings'] + sol['NumSaturatedRings'] + 
                  sol['NumAliphaticRings']) / sol['MolWt']
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
    
    # I think stadnardizing also helps performance - need to check tho
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_final)
    
    y = np.array(sol['Solubility'])
    
    print(f"Feature engineering summary:")
    print(f"  Original features: {len(basic_names)}")
    print(f"  Enhanced features: {len(enhanced_names)}")
    print(f"  Final features (after correlation removal): {len(feature_names)}")
    print(f"  Features removed due to high correlation: 1")
    print(f"  Features standardized: Yes")
    
    return X_final, y, feature_names

def uncertainty_plot(gp, X_train, X_test, y_test):
    y_pred, y_std = gp.predict(X_test, return_std=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 2 std away from mean uncertainty get differently coloured
    threshold = np.mean(y_std) + 2 * np.std(y_std)
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

def plot_uncertainty_heatmap(gp, X, y, feature_names, sigmas, X_train, save_path):
    length_scales = 1.0 / sigmas
    sorted_indices = np.argsort(length_scales)[::-1]
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
    
    _, y_std_grid = gp.predict(X_grid, return_std=True)
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
            train_sample_indices = np.random.choice(len(X_train), train_subset_size, replace=False)
            X_train_sample = X_train[train_sample_indices]
        else:
            X_train_sample = X_train
            
        ax_heat.scatter(X_train_sample[:, x_idx], X_train_sample[:, y_idx], 
                       c='lime', marker='x', s=18, alpha=0.7, 
                       label='Training Data', zorder=5, linewidth=1.2)
        ax_heat.legend()
    
    ax_hist.hist(y_std_grid.ravel(), bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel('Predicted Uncertainty')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Distribution of Predicted Uncertainties')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def learning_evolution(X, y, feature_names, config, sigmas, full_X, n_init=1, n_steps=300, gif_path=f'{FIGURE_DIR}/learning_evolution.gif'):
    np.random.seed(42)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    frames = []
    gif_config = ConfigParser()
    
    for section in config.sections():
        gif_config[section] = dict(config[section])
    if 'PARALLEL' not in gif_config:
        gif_config['PARALLEL'] = {}
    
    gif_config['PARALLEL']['use_parallel'] = 'false'
    gif_config['PARALLEL']['n_jobs'] = '1'
    gif_config['PARALLEL']['use_gpu'] = 'false'

    length_scales = 1.0 / sigmas
    sorted_indices = np.argsort(length_scales)[::-1]
    top2_idx = sorted_indices[:2]
    x_idx, y_idx = top2_idx[0], top2_idx[1]
    x_name, y_name = feature_names[x_idx], feature_names[y_idx]

    grid_size = 60
    x1 = np.linspace(np.percentile(X[:, x_idx], 1), np.percentile(X[:, x_idx], 99), grid_size)
    x2 = np.linspace(np.percentile(X[:, y_idx], 1), np.percentile(X[:, y_idx], 99), grid_size)
    X1g, X2g = np.meshgrid(x1, x2)
    X_grid = np.zeros((X1g.size, X.shape[1]))
    X_grid[:, x_idx] = X1g.ravel()
    X_grid[:, y_idx] = X2g.ravel()
    for i in range(X.shape[1]):
        if i not in (x_idx, y_idx):
            X_grid[:, i] = np.mean(X[:, i])

    pool_idx = np.arange(len(X))
    init_idx = np.random.choice(pool_idx, size=n_init, replace=False)
    train_idx = list(init_idx)
    pool_idx = np.setdiff1d(pool_idx, train_idx)
    temp_train_idx = list(train_idx)
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
            next_idx_in_pool = np.argmax(y_std_pool)
            next_idx = temp_pool_idx[next_idx_in_pool]
            temp_train_idx.append(next_idx)
            temp_pool_idx = np.setdiff1d(temp_pool_idx, [next_idx])

    if len(temp_mean_uncertainties) == 0 or not np.isfinite(temp_mean_uncertainties).all():
        unc_ylim = (0, 1)
        grid_unc_min, grid_unc_max = 0, 1
    else:
        unc_ylim = (min(temp_mean_uncertainties)*0.95, max(temp_mean_uncertainties)*1.05)
        
    del temp_train_idx, temp_pool_idx, temp_mean_uncertainties

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
            next_idx_in_pool = np.argmax(y_std_pool)
            next_idx = pool_idx[next_idx_in_pool]
            train_idx.append(next_idx)
            pool_idx = np.setdiff1d(pool_idx, [next_idx])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        ax.scatter(full_X[:, x_idx], full_X[:, y_idx], c='black', s=40, marker='x', label='All Data', alpha=0.8, zorder=1)
        ax.scatter(X[train_idx, x_idx], X[train_idx, y_idx], c='lime', s=40, marker='x', label='Train', alpha=0.9, zorder=3)
        
        if len(train_idx) > n_init:
            last_added_idx = train_idx[-1]
            ax.scatter(X[last_added_idx, x_idx], X[last_added_idx, y_idx], c='red', s=120, marker='x', label='Newly Added', edgecolor='black', linewidth=3, zorder=4)
        
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f'Learning Step {step+1}/{n_steps}')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(np.percentile(X[:, x_idx], 1), np.percentile(X[:, x_idx], 99))
        ax.set_ylim(np.percentile(X[:, y_idx], 1), np.percentile(X[:, y_idx], 99))
        
        _, y_std_grid = gp.predict(X_grid, return_std=True)
        y_std_grid = y_std_grid.reshape(X1g.shape)
        levels = np.linspace(grid_unc_min, grid_unc_max, 40)
        im = ax.contourf(X1g, X2g, y_std_grid, levels=levels, cmap='plasma', alpha=0.7, vmin=grid_unc_min, vmax=grid_unc_max)
        
        fig.colorbar(im, ax=ax, label='Uncertainty')
        
        ax2 = axes[1]
        ax2.plot(np.arange(1, step+2), mean_uncertainties, '-o', color='purple')
        ax2.set_xlabel('Learning Step')
        ax2.set_ylabel('Mean Predictive Variance')
        ax2.set_title('Uncertainty Reduction')
        ax2.set_xlim(1, n_steps)
        ax2.set_ylim(unc_ylim)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        frame_path = f'{FIGURE_DIR}/_al_frame_{step:03d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        frames.append(imageio.v2.imread(frame_path))
        os.remove(frame_path)
    
    imageio.mimsave(gif_path, frames, duration=3)
    print(f"Active learning GIF saved to {gif_path}")
    
    for f in glob.glob(f'{FIGURE_DIR}/_al_frame_*.png'):
        try:
            os.remove(f)
        except Exception:
            pass

def surface_plot(gp, X, y, sigmas, feature_names, save_path):
    length_scales = 1.0 / sigmas
    sorted_indices = np.argsort(length_scales)[::-1]
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

    y_pred_grid = gp.predict(X_grid)
    y_pred_grid = y_pred_grid.reshape(X1g.shape)

    # Sample actual data points for overlay
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
    surf = ax.plot_surface(X1g, X2g, y_pred_grid, cmap='viridis', 
                          alpha=0.8, linewidth=0, antialiased=True)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='GP Solubility')
    
    ax.scatter(X_sample[:, x_idx], X_sample[:, y_idx], y_sample, c='red', s=20, alpha=0.8, 
               marker='x', label=f'Actual Data ({len(X_sample)} points)', edgecolor='black', linewidth=0.5)

    ax.set_xlabel(feature_names[x_idx])
    ax.set_ylabel(feature_names[y_idx])
    ax.set_zlabel('Solubility')
    ax.set_title(f'3D Solubility Surface Plot')
    ax.set_xlim(np.percentile(X[:, x_idx], 5), np.percentile(X[:, x_idx], 95))
    ax.set_ylim(np.percentile(X[:, y_idx], 5), np.percentile(X[:, y_idx], 95))
    ax.legend()
    
    ax.view_init(elev=10, azim=30)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("="*80)
    print("Tuning/training on AqSolDB")
    print("="*80)

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=69)
    
    try:
        configure_parallel_settings(use_parallel=True, n_jobs=4, use_gpu=True)
    except Exception as e:
        print(f"Warning: Could not configure parallel processing: {e}")
        print("Continuing with sequential processing...")
    
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
        tuner = GPAutoTuner(
            X_train, 
            y_train, 
            config_path=CONFIG_PATH, 
            sigma_save_path=SIGMA_PATH, 
            force_dense=True, 
            metric='MSE',
            n_jobs=4, 
            use_gpu=True, 
            chunk_size=500, 
            min_size_for_parallel=500  
            )
        tuner.optimize(n_trials=3000)
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
    plot_uncertainty_heatmap(gp, X, y, feature_names, sigmas, X_train, f'{FIGURE_DIR}/kernel_uncertainty_heatmap.png')
    
    surface_plot(gp, X, y, sigmas, feature_names, f'{FIGURE_DIR}/solubility_surface.png')

    subset_size = min(500, len(X))
    subset_idx = np.random.choice(len(X), subset_size, replace=False)
    X_subset = X[subset_idx, :2]
    y_subset = y[subset_idx]
    
    learning_evolution(X_subset, y_subset, feature_names[:2], config, sigmas[:2], full_X=X, gif_path=f'{FIGURE_DIR}/learning_evolution.gif')

    print("\n" + "="*80)
    print("Done!")
    print("="*80)

    print("Files:")
    print(f"- {FIGURE_DIR}/solubility_uncertainty.png")
    print(f"- {FIGURE_DIR}/kernel_length_scales.png")
    print(f"- {FIGURE_DIR}/kernel_uncertainty_heatmap.png")
    print(f"- {FIGURE_DIR}/solubility_surface.png")
    print(f"- {FIGURE_DIR}/learning_evolution.gif")
    print(f"- {CONFIG_PATH}")
    print(f"- {SIGMA_PATH}")

if __name__ == "__main__":
    main()
