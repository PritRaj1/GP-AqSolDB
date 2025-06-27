import optuna
import numpy as np
import os
import sys
from configparser import ConfigParser
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from src.gp import GP

class GPAutoTuner:
    def __init__(
            self, 
            X_train, 
            y_train, 
            config_path="../config/GP.ini", 
            sigma_save_path="../config/optimized_sigmas.pkl"
        ):
        """
        Initialize the GP Auto Tuner
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        config_path : str
            Path to save non-vector hyperparameters
        sigma_save_path : str
            Path to save vector sigma hyperparameters
        """
        self.X_train = X_train
        self.y_train = y_train
        self.config_path = config_path
        self.sigma_save_path = sigma_save_path
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
        
        # Load existing config
        self.config = ConfigParser()
        if os.path.exists(config_path):
            self.config.read(config_path)
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        self.config['KERNEL'] = {
            'type': 'RBF',
            'lmbda': '0.1',
            'alpha': '1.0',
            'use_cache': 'true',
            'cache_size': '100'
        }
        self.config['SPARSE'] = {
            'use_sparse': 'false',
            'num_inducing': '20',
            'inducing_method': 'random'
        }
    
    def calculate_bic(self, mse, n_params, n_samples):
        """
        Calculate Bayesian Information Criterion (BIC)
        
        BIC = n * log(MSE) + k * log(n)
        where n = number of samples, k = number of parameters
        
        Lower BIC is better (penalizes complexity)
        """
        return n_samples * np.log(mse) + n_params * np.log(n_samples)
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
            
        Returns:
        --------
        float : Cross-validation score (negative BIC for minimization)
        """
        # Suggest model type (dense vs sparse)
        use_sparse = trial.suggest_categorical('use_sparse', [True, False])
        
        # Suggest hyperparameters
        kernel_type = trial.suggest_categorical('kernel_type', ['RBF', 'RQ'])
        lmbda = trial.suggest_float('lmbda', 1e-4, 1.0, log=True)
        alpha = trial.suggest_float('alpha', 0.1, 10.0) if kernel_type == 'RQ' else 1.0
        sigmas = [trial.suggest_float(f'sigma_{i}', 0.1, 3.0) for i in range(self.n_features)]
        
        # Sparse GP specific parameters
        if use_sparse:
            # Suggest number of inducing points (between 10% and 50% of data size)
            min_inducing = max(10, int(0.1 * self.n_samples))
            max_inducing = min(int(0.5 * self.n_samples), self.n_samples - 1)
            num_inducing = trial.suggest_int('num_inducing', min_inducing, max_inducing)
            inducing_method = trial.suggest_categorical('inducing_method', ['random', 'uniform'])
        else:
            num_inducing = 20
            inducing_method = 'random'
        
        # Create config for this trial
        config = ConfigParser()
        config['KERNEL'] = {
            'type': kernel_type,
            'lmbda': str(lmbda),
            'alpha': str(alpha),
            'use_cache': 'true',
            'cache_size': '100'
        }
        config['SPARSE'] = {
            'use_sparse': str(use_sparse).lower(),
            'num_inducing': str(num_inducing),
            'inducing_method': inducing_method
        }
        
        # Perform cross-validation
        try:
            cv_results = self._cross_validate_gp(config, sigmas, n_splits=5)
            return -np.mean(cv_results['bic'])  # Negative because Optuna minimizes
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')  # Return large value for failed trials
    
    def _cross_validate_gp(self, config, sigmas, n_splits=5):
        """
        Perform cross-validation for GP with given hyperparameters
        
        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        sigmas : list
            Sigma values for each feature
        n_splits : int
            Number of CV folds
            
        Returns:
        --------
        dict : Cross-validation results with MSE and BIC
        """
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scores = []
        bic_scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            # Create GP and fit
            gp = GP(config, np.array(sigmas))
            gp.fit(X_train_fold, y_train_fold)
            
            # Get model complexity for BIC calculation
            n_params = gp.get_model_complexity()
            
            # Predict and calculate score
            y_pred = gp.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, y_pred)
            bic = self.calculate_bic(mse, n_params, len(y_val_fold))
            
            mse_scores.append(mse)
            bic_scores.append(bic)
        
        return {
            'mse': mse_scores,
            'bic': bic_scores
        }
    
    def optimize(self, n_trials=100, timeout=None):
        """
        Run hyperparameter optimization
        
        Parameters:
        -----------
        n_trials : int
            Number of optimization trials
        timeout : int
            Timeout in seconds (None for no timeout)
            
        Returns:
        --------
        dict : Best hyperparameters
        """
        print(f"Starting GP hyperparameter optimization with {n_trials} trials...")
        print(f"Features: {self.n_features}")
        print(f"Samples: {self.n_samples}")
        print("Models: Dense GP and Sparse GP (FITC)")
        print("Metric: BIC (Bayesian Information Criterion)")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\nOptimization completed!")
        print(f"Best CV BIC: {-best_value:.2f}")
        print(f"Best model type: {'Sparse' if best_params['use_sparse'] else 'Dense'}")
        print(f"Best parameters: {best_params}")
        
        # Report cache stats
        try:
            from src.kernels import get_cache_stats
            cache_stats = get_cache_stats()
            total_requests = cache_stats['hits'] + cache_stats['misses']
            if total_requests > 0:
                print(f"\nKernel cache statistics:")
                print(f"  Cache hits: {cache_stats['hits']}")
                print(f"  Cache misses: {cache_stats['misses']}")
                print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
                print(f"  Cache size: {cache_stats['cache_size']}")
        except:
            pass
        
        self._save_best_parameters(best_params)        
        return best_params
    
    def _save_best_parameters(self, best_params):
        """Save the best hyperparameters to files"""

        use_sparse = best_params['use_sparse']
        kernel_type = best_params['kernel_type']
        lmbda = best_params['lmbda']
        alpha = best_params.get('alpha', 1.0)
        
        if 'KERNEL' not in self.config:
            self.config['KERNEL'] = {}
        if 'SPARSE' not in self.config:
            self.config['SPARSE'] = {}
        
        self.config['KERNEL']['type'] = kernel_type
        self.config['KERNEL']['lmbda'] = str(lmbda)
        self.config['KERNEL']['alpha'] = str(alpha)
        
        # Save sparse GP parameters if applicable
        self.config['SPARSE']['use_sparse'] = str(use_sparse).lower()
        if use_sparse:
            num_inducing = best_params.get('num_inducing', 20)
            inducing_method = best_params.get('inducing_method', 'random')
            self.config['SPARSE']['num_inducing'] = str(num_inducing)
            self.config['SPARSE']['inducing_method'] = inducing_method
                
        with open(self.config_path, 'w') as f:
            self.config.write(f)
        
        sigmas = [best_params[f'sigma_{i}'] for i in range(self.n_features)]
        
        with open(self.sigma_save_path, 'wb') as f:
            pickle.dump(sigmas, f)
        
        print(f"\nSaved hyperparameters:")
        print(f"Config file: {self.config_path}")
        print(f"Sigma file: {self.sigma_save_path}")
        print(f"Model type: {'Sparse' if use_sparse else 'Dense'}")
        print(f"Kernel type: {kernel_type}")
        print(f"Lambda: {lmbda}")
        print(f"Alpha: {alpha}")
        print(f"Sigmas: {sigmas}")
        if use_sparse:
            print(f"Number of inducing points: {best_params.get('num_inducing', 20)}")
            print(f"Inducing method: {best_params.get('inducing_method', 'random')}")
        
        preserved_sections = [section for section in self.config.sections() if section not in ['KERNEL', 'SPARSE']]
        if preserved_sections:
            print(f"Preserved sections: {preserved_sections}")
    
    def load_optimized_parameters(self):
        """Load optimized parameters from saved files"""
        config = ConfigParser()
        config.read(self.config_path)
        
        if os.path.exists(self.sigma_save_path):
            with open(self.sigma_save_path, 'rb') as f:
                sigmas = pickle.load(f)
        else:
            raise FileNotFoundError(f"Sigma file not found: {self.sigma_save_path}")
        
        return config, np.array(sigmas)

def load_sigmas_from_file(file_path):
    """Load optimized sigmas from file"""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return np.array(pickle.load(f))
    else:
        raise FileNotFoundError(f"Sigma file not found: {file_path}")
