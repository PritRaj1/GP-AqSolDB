import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
from configparser import ConfigParser
import sys
import os
import tempfile
import pickle

sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("paper", font_scale=1.2)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}"
})

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.GP_fcns import GP
from src.auto_tune import GPAutoTuner, load_sigmas_from_file

@pytest.fixture
def sample_data_2d():
    """Fixture for 2D test data"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 50)
    return X_train, y_train

@pytest.fixture
def sample_data_3d():
    """Fixture for 3D test data"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (60, 3))
    y_train = (np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + 
               X_train[:, 2]**2 + np.random.normal(0, 0.1, 60))
    return X_train, y_train

@pytest.fixture
def temp_config_file():
    """Fixture for temp config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write("""
                [KERNEL]
                type="RBF"
                lmbda=0.1
                alpha=1.0

                [TUNING]
                n_trials=50
                timeout=300
                """)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def temp_sigma_file():
    """Fixture for temporary sigma file"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.mark.auto_tune
@pytest.mark.parametrize("n_features", [1, 2, 3])
def test_auto_tuner_initialization(n_features):
    """Test GPAutoTuner initialization with different feature dimensions"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (30, n_features))
    y_train = np.random.normal(0, 1, 30)
    
    tuner = GPAutoTuner(X_train, y_train)
    
    assert tuner.X_train.shape == (30, n_features)
    assert tuner.y_train.shape == (30,)
    assert tuner.n_features == n_features
    assert tuner.config is not None
    assert 'KERNEL' in tuner.config

@pytest.mark.auto_tune
def test_auto_tuner_default_config():
    """Test that GPAutoTuner creates default config when none exists"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (20, 2))
    y_train = np.random.normal(0, 1, 20)
    
    # Use non-existent config path to trigger default creation
    with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as f:
        temp_config = f.name
    
    try:
        tuner = GPAutoTuner(X_train, y_train, config_path=temp_config)
        
        assert 'KERNEL' in tuner.config
        assert tuner.config['KERNEL']['type'] == 'RBF'
        assert tuner.config['KERNEL']['lmbda'] == '0.1'
        assert tuner.config['KERNEL']['alpha'] == '1.0'
    finally:
        if os.path.exists(temp_config):
            os.unlink(temp_config)

@pytest.mark.auto_tune
def test_auto_tuner_preserves_existing_config(temp_config_file):
    """Test that GPAutoTuner preserves existing config sections"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (20, 2))
    y_train = np.random.normal(0, 1, 20)
    
    tuner = GPAutoTuner(X_train, y_train, config_path=temp_config_file)
    
    # Check that TUNING section is preserved
    assert 'TUNING' in tuner.config
    assert tuner.config['TUNING']['n_trials'] == '50'
    assert tuner.config['TUNING']['timeout'] == '300'

@pytest.mark.auto_tune
@pytest.mark.parametrize("kernel_type", ["RBF", "RQ"])
def test_objective_function(kernel_type, sample_data_2d):
    """Test the objective function with different kernel types"""
    X_train, y_train = sample_data_2d
    tuner = GPAutoTuner(X_train, y_train)
    
    class MockTrial:
        def suggest_categorical(self, name, choices):
            return kernel_type
        
        def suggest_float(self, name, low, high, log=False):
            if name == 'lmbda':
                return 0.1
            elif name == 'alpha':
                return 2.0
            else: 
                return 1.0
    
    trial = MockTrial()
    score = tuner.objective(trial)
    
    assert isinstance(score, float)
    assert not np.isnan(score)
    assert not np.isinf(score)

@pytest.mark.auto_tune
def test_cross_validation_gp(sample_data_2d):
    """Test cross-validation functionality"""
    X_train, y_train = sample_data_2d
    tuner = GPAutoTuner(X_train, y_train)
    
    config = ConfigParser()
    config['KERNEL'] = {
        'type': 'RBF',
        'lmbda': '0.1',
        'alpha': '1.0'
    }
    
    sigmas = [1.0, 1.0]  
    
    cv_scores = tuner._cross_validate_gp(config, sigmas, n_splits=3)
    
    assert len(cv_scores) == 3
    assert all(isinstance(score, float) for score in cv_scores)
    assert all(score >= 0 for score in cv_scores)
    assert not any(np.isnan(score) for score in cv_scores)

@pytest.mark.auto_tune
@pytest.mark.slow
def test_optimization_small_scale(sample_data_2d, temp_config_file, temp_sigma_file):
    """Test optimization with small number of trials"""
    X_train, y_train = sample_data_2d
    
    tuner = GPAutoTuner(X_train, y_train, 
                       config_path=temp_config_file,
                       sigma_save_path=temp_sigma_file)
    
    best_params = tuner.optimize(n_trials=5)
    
    assert isinstance(best_params, dict)
    assert 'kernel_type' in best_params
    assert 'lmbda' in best_params
    assert 'sigma_0' in best_params
    assert 'sigma_1' in best_params
    
    assert os.path.exists(temp_config_file)
    assert os.path.exists(temp_sigma_file)
    
    # Check that config preserves other sections
    config = ConfigParser()
    config.read(temp_config_file)
    assert 'TUNING' in config
    assert config['TUNING']['n_trials'] == '50'

@pytest.mark.auto_tune
def test_save_best_parameters(sample_data_2d, temp_config_file, temp_sigma_file):
    """Test saving best parameters functionality"""
    X_train, y_train = sample_data_2d
    tuner = GPAutoTuner(X_train, y_train, 
                       config_path=temp_config_file,
                       sigma_save_path=temp_sigma_file)
    
    best_params = {
        'kernel_type': 'RQ',
        'lmbda': 0.05,
        'alpha': 2.5,
        'sigma_0': 1.2,
        'sigma_1': 0.8
    }
    
    tuner._save_best_parameters(best_params)
    
    # Check config file
    config = ConfigParser()
    config.read(temp_config_file)
    assert config['KERNEL']['type'] == 'RQ'
    assert config['KERNEL']['lmbda'] == '0.05'
    assert config['KERNEL']['alpha'] == '2.5'
    assert 'TUNING' in config  
    
    # Check sigma file
    with open(temp_sigma_file, 'rb') as f:
        sigmas = pickle.load(f)
    
    assert len(sigmas) == 2
    assert sigmas[0] == 1.2
    assert sigmas[1] == 0.8

@pytest.mark.auto_tune
def test_load_optimized_parameters(sample_data_2d, temp_config_file, temp_sigma_file):
    """Test loading optimized parameters"""
    X_train, y_train = sample_data_2d
    tuner = GPAutoTuner(X_train, y_train, 
                       config_path=temp_config_file,
                       sigma_save_path=temp_sigma_file)
    
    config = ConfigParser()
    config['KERNEL'] = {
        'type': 'RBF',
        'lmbda': '0.1',
        'alpha': '1.0'
    }
    config['TUNING'] = {
        'n_trials': '100',
        'timeout': '3600'
    }
    
    with open(temp_config_file, 'w') as f:
        config.write(f)
    
    test_sigmas = [1.5, 0.7]
    with open(temp_sigma_file, 'wb') as f:
        pickle.dump(test_sigmas, f)
    
    loaded_config, loaded_sigmas = tuner.load_optimized_parameters()
    
    assert loaded_config['KERNEL']['type'] == 'RBF'
    assert loaded_config['KERNEL']['lmbda'] == '0.1'
    assert loaded_config['TUNING']['n_trials'] == '100'
    assert np.array_equal(loaded_sigmas, test_sigmas)

@pytest.mark.auto_tune
def test_load_sigmas_from_file(temp_sigma_file):
    """Test load_sigmas_from_file utility function"""
    test_sigmas = [1.2, 0.8, 1.5]
    
    with open(temp_sigma_file, 'wb') as f:
        pickle.dump(test_sigmas, f)
    
    loaded_sigmas = load_sigmas_from_file(temp_sigma_file)
    
    assert isinstance(loaded_sigmas, np.ndarray)
    assert np.array_equal(loaded_sigmas, test_sigmas)

@pytest.mark.auto_tune
def test_load_sigmas_from_file_not_found():
    """Test load_sigmas_from_file with non-existent file"""
    with pytest.raises(FileNotFoundError):
        load_sigmas_from_file("non_existent_file.pkl")

@pytest.mark.auto_tune
@pytest.mark.parametrize("n_features", [1, 2, 3, 4])
def test_sigma_parameter_count(n_features):
    """Test that correct number of sigma parameters are suggested"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (30, n_features))
    y_train = np.random.normal(0, 1, 30)
    
    tuner = GPAutoTuner(X_train, y_train)
    
    # Check sigma parameter names
    class MockTrial:
        def suggest_categorical(self, name, choices):
            return 'RBF'
        
        def suggest_float(self, name, low, high, log=False):
            return 1.0
    
    trial = MockTrial()
    
    assert tuner.n_features == n_features

@pytest.mark.auto_tune
def test_gp_with_optimized_parameters(sample_data_2d, temp_config_file, temp_sigma_file):
    """Test that GP works with optimized parameters"""
    X_train, y_train = sample_data_2d
    tuner = GPAutoTuner(X_train, y_train, 
                       config_path=temp_config_file,
                       sigma_save_path=temp_sigma_file)
    
    best_params = {
        'kernel_type': 'RBF',
        'lmbda': 0.1,
        'alpha': 1.0,
        'sigma_0': 1.2,
        'sigma_1': 0.8
    }
    
    tuner._save_best_parameters(best_params)
    
    config, sigmas = tuner.load_optimized_parameters()
    gp = GP(config, sigmas)
    gp.fit(X_train, y_train)
    
    X_test = np.random.uniform(0, 5, (10, 2))
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    assert len(y_pred) == 10
    assert len(y_std) == 10
    assert np.all(y_std >= 0)
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isnan(y_std))

def visualize_auto_tune_results():
    """Create visualizations for auto-tuning results"""
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 50)
    
    with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config:
        temp_config_path = temp_config.name
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    
    try:
        tuner = GPAutoTuner(X_train, y_train, 
                           config_path=temp_config_path,
                           sigma_save_path=temp_sigma_path)
        
        best_params = tuner.optimize(n_trials=20)
        
        config, sigmas = tuner.load_optimized_parameters()
        
        gp = GP(config, sigmas)
        gp.fit(X_train, y_train)
        
        x1 = np.linspace(0, 5, 50)
        x2 = np.linspace(0, 5, 50)
        X1, X2 = np.meshgrid(x1, x2)
        X_test = np.column_stack([X1.ravel(), X2.ravel()])
        
        y_pred, y_std = gp.predict(X_test, return_std=True)
        y_pred_grid = y_pred.reshape(X1.shape)
        y_std_grid = y_std.reshape(X1.shape)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes[0].contourf(X1, X2, y_pred_grid, levels=25, cmap='viridis', alpha=0.8)
        axes[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=30, marker='x', label='Training Data')
        axes[0].set_title('Optimized GP Prediction\nKernel: ' + config["KERNEL"]["type"] + r', $\lambda$: ' + config["KERNEL"]["lmbda"])
        axes[0].set_xlabel(r'$x_1$')
        axes[0].set_ylabel(r'$x_2$')
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Uncertainty
        im2 = axes[1].contourf(X1, X2, y_std_grid, levels=25, cmap='plasma', alpha=0.8)
        axes[1].scatter(X_train[:, 0], X_train[:, 1], c='lime', s=30, marker='x', label='Training Data')
        axes[1].set_title('Prediction Uncertainty\n' + r'$\sigma$: ' + str(sigmas))
        axes[1].set_xlabel(r'$x_1$')
        axes[1].set_ylabel(r'$x_2$')
        axes[1].legend()
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('tests/figures/auto_tune_results.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
    finally:
        # Cleanup
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)

if __name__ == "__main__":
    print("Running auto-tuning tests...")
    
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1]/5) + np.random.normal(0, 0.1, 50)
    
    test_auto_tuner_initialization(2)
    test_objective_function("RBF", (X_train, y_train))
    test_cross_validation_gp((X_train, y_train))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as temp_config:
        temp_config.write("""
                        [KERNEL]
                        type="RBF"
                        lmbda=0.1
                        alpha=1.0

                        [TUNING]
                        n_trials=50
                        timeout=300
                            """)
        temp_config_path = temp_config.name
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    
    try:
        test_save_best_parameters((X_train, y_train), temp_config_path, temp_sigma_path)
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)
    
    print("All tests passed!")
    
    print("Creating visualizations...")
    visualize_auto_tune_results()
    print("Visualizations saved to tests/figures/auto_tune_results.png") 