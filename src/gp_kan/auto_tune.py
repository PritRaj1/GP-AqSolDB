import optuna
import numpy as np
import os
from configparser import ConfigParser
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
import jax
import jax.numpy as jnp
from typing import List, Dict, Any, Optional
warnings.filterwarnings('ignore')

from src.gp_kan.fully_connected import GP_KAN, create_default_conf
from src.gp_kan.normal_dist import NormalDist

class GPKANAutoTuner:
    def __init__(
            self, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
            config_path: str = "../config/gp_kan.ini", 
            params_save_path: str = "../config/gp_kan_params.pkl",
            metric: str = 'BIC',
            n_jobs: int = 2,
            use_gpu: bool = False,
            max_hidden_layers: int = 3,
            max_hidden_size: int = 10,
            num_epochs: int = 50
        ):
        """
        Initialize the GP-KAN Auto Tuner
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        config_path : str
            Path to save configuration file
        params_save_path : str
            Path to save optimized parameters
        metric : str
            Optimization metric: 'BIC' or 'MSE'
        n_jobs : int
            Number of jobs for parallelization
        use_gpu : bool
            Whether to use GPU
        max_hidden_layers : int
            Maximum number of hidden layers to try
        max_hidden_size : int
            Maximum hidden layer size to try
        num_epochs : int
            Number of training epochs for each trial
        """
        self.X_train = X_train
        self.y_train = y_train
        self.config_path = config_path
        self.params_save_path = params_save_path
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
        self.metric = metric.upper()
        
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.max_hidden_layers = max_hidden_layers
        self.max_hidden_size = max_hidden_size
        self.num_epochs = num_epochs
        
        if self.metric not in ['BIC', 'MSE']:
            raise ValueError("metric must be 'BIC' or 'MSE'")
        
        self.config = ConfigParser()
        if os.path.exists(config_path):
            self.config.read(config_path)
        else:
            self._create_default_config()
        
        self._load_device_settings()
    
    def _load_device_settings(self):
        try:
            if 'DEVICE' in self.config:
                device_section = self.config['DEVICE']
                use_gpu = device_section.get('use_gpu', 'false').lower() == 'true'
                precision = device_section.get('precision', 'float32')
            else:
                use_gpu = False
                precision = 'float32'
            
            print(f"Device configuration:")
            print(f"  Use GPU: {use_gpu}")
            print(f"  Precision: {precision}")
            
            if use_gpu:
                try:
                    gpu_devices = jax.devices('gpu')
                    print(f"  GPU devices available: {len(gpu_devices)}")
                except:
                    print(f"  GPU devices available: 0")
            
        except Exception as e:
            print(f"Warning: Could not load device settings: {e}")
    
    def _create_default_config(self):
        self.config['NETWORK'] = {
            'input_size': str(self.n_features),
            'output_size': '1'
        }
        self.config['GP'] = {
            'num_inducing_points': '10',
            'z_init_low': '-2.0',
            'z_init_high': '2.0',
            'h_init_low': '-1.0',
            'h_init_high': '1.0',
            'global_length_scale': '0.4',
            'min_length_scale': '0.2',
            'global_covariance_scale': '1.0',
            'min_covariance_scale': '0.1',
            'global_jitter': '0.001',
            'baseline_jitter': '0.01'
        }
        self.config['NORMALIZATION'] = {
            'min_var': '0.2'
        }
        self.config['TRAINING'] = {
            'seed': '42'
        }
        self.config['DEVICE'] = {
            'use_gpu': str(self.use_gpu).lower(),
            'device': 'gpu' if self.use_gpu else 'cpu',
            'precision': 'float32'
        }
    
    def calculate_bic(self, mse: float, n_params: int, n_samples: int) -> float:
        """
        Calculate Bayesian Information Criterion (BIC)
        
        BIC = n * log(MSE) + k * log(n)
        where n = number of samples, k = number of parameters
        
        Lower BIC is better (penalizes complexity)
        """
        return n_samples * np.log(mse) + n_params * np.log(n_samples)
    
    def _count_network_parameters(self, hidden_sizes: List[int]) -> int:
        """Count total number of parameters in the network"""
        layer_sizes = [self.n_features] + hidden_sizes + [1]
        total_params = 0
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # z: (I, O, P), h: (I, O, P), l: (I, O), s: (I, O), jitter: (I, O)
            num_inducing = int(self.config['GP'].get('num_inducing_points', '10'))
            gp_params = input_size * output_size * (2 * num_inducing + 3)  # z, h, l, s, jitter
            total_params += gp_params
            
            # Normalizer parameters (if not last layer)
            if i < len(layer_sizes) - 2:
                total_params += 2  # mean and variance tracking
        
        return total_params
    
    def _objective(self, trial: optuna.Trial) -> float:
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
        try:
            num_hidden_layers = trial.suggest_int('num_hidden_layers', 0, self.max_hidden_layers)
            hidden_sizes = []
            
            for i in range(num_hidden_layers):
                hidden_size = trial.suggest_int(f'hidden_size_{i}', 2, self.max_hidden_size)
                hidden_sizes.append(hidden_size)
            
            num_inducing_points = trial.suggest_int('num_inducing_points', 5, 20)
            z_init_low = trial.suggest_float('z_init_low', -3.0, -1.0)
            z_init_high = trial.suggest_float('z_init_high', 1.0, 3.0)
            h_init_low = trial.suggest_float('h_init_low', -2.0, 0.0)
            h_init_high = trial.suggest_float('h_init_high', 0.0, 2.0)
            
            global_length_scale = trial.suggest_float('global_length_scale', 0.1, 1.0)
            min_length_scale = trial.suggest_float('min_length_scale', 0.05, 0.5)
            
            global_covariance_scale = trial.suggest_float('global_covariance_scale', 0.5, 2.0)
            min_covariance_scale = trial.suggest_float('min_covariance_scale', 0.05, 0.5)
            
            global_jitter = trial.suggest_float('global_jitter', 1e-4, 1e-2, log=True)
            baseline_jitter = trial.suggest_float('baseline_jitter', 1e-3, 1e-1, log=True)
            
            min_var = trial.suggest_float('min_var', 0.1, 0.5)
            
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            pretrain_iters = trial.suggest_int('pretrain_iters', 5, 20)
            
            config = ConfigParser()
            config['NETWORK'] = {
                'input_size': str(self.n_features),
                'output_size': '1'
            }
            config['GP'] = {
                'num_inducing_points': str(num_inducing_points),
                'z_init_low': str(z_init_low),
                'z_init_high': str(z_init_high),
                'h_init_low': str(h_init_low),
                'h_init_high': str(h_init_high),
                'global_length_scale': str(global_length_scale),
                'min_length_scale': str(min_length_scale),
                'global_covariance_scale': str(global_covariance_scale),
                'min_covariance_scale': str(min_covariance_scale),
                'global_jitter': str(global_jitter),
                'baseline_jitter': str(baseline_jitter)
            }
            config['NORMALIZATION'] = {
                'min_var': str(min_var)
            }
            config['TRAINING'] = {
                'seed': '42'
            }
            config['DEVICE'] = {
                'use_gpu': str(self.use_gpu).lower(),
                'device': 'gpu' if self.use_gpu else 'cpu',
                'precision': 'float32'
            }
            
            config['TRAINING']['learning_rate'] = str(learning_rate)
            config['TRAINING']['num_epochs'] = str(self.num_epochs)
            config['TRAINING']['batch_size'] = str(batch_size)
            config['TRAINING']['pretrain_iters'] = str(pretrain_iters)
            
            cv_results = self._cross_validate_gpkan(config, hidden_sizes, n_splits=3)  # Reduced for speed
            
            if self.metric == 'BIC':
                return -np.mean(cv_results['bic'])  # Negative because Optuna minimizes
            else:  
                return np.mean(cv_results['mse'])  # Direct minimization
                
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')  # Return large value for failed trials
    
    def _cross_validate_gpkan(self, config: ConfigParser, hidden_sizes: List[int], n_splits: int = 5):
        """
        Perform cross-validation for GP-KAN with given hyperparameters
        
        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        hidden_sizes : List[int]
            List of hidden layer sizes
        n_splits : int
            Number of CV folds
            
        Returns:
        --------
        dict : Cross-validation results with MSE and BIC
        """
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scores = []
        bic_scores = []
        
        learning_rate = float(config['TRAINING'].get('learning_rate', '0.001'))
        num_epochs = int(config['TRAINING'].get('num_epochs', '30'))
        batch_size = int(config['TRAINING'].get('batch_size', '32'))
        pretrain_iters = int(config['TRAINING'].get('pretrain_iters', '10'))
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            try:
                network = GP_KAN(config, hidden_sizes=hidden_sizes)
                
                network.train(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold,
                    learning_rate=learning_rate,
                    num_epochs=self.num_epochs,
                    batch_size=batch_size,
                    patience=10,
                    pretrain_iters=pretrain_iters
                )
                
                X_val_mean = X_val_fold.astype(np.float32)
                X_val_var = np.zeros_like(X_val_mean)
                X_val_dist = NormalDist(jnp.array(X_val_mean), jnp.array(X_val_var))
                
                output_dist = network.forward(X_val_dist)
                y_pred = np.array(output_dist.mean).flatten()
                
                mse = mean_squared_error(y_val_fold, y_pred)
                n_params = self._count_network_parameters(hidden_sizes)
                bic = self.calculate_bic(mse, n_params, len(y_val_fold))
                
                mse_scores.append(mse)
                bic_scores.append(bic)
                
            except Exception as e:
                print(f"CV fold failed: {e}")
                mse_scores.append(1e6)
                bic_scores.append(1e6)
        
        return {
            'mse': mse_scores,
            'bic': bic_scores
        }
    
    def optimize(self, n_trials: int = 100, timeout: Optional[int] = None):
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
        print(f"Starting GP-KAN hyperparameter optimization with {n_trials} trials...")
        print(f"Features: {self.n_features}")
        print(f"Samples: {self.n_samples}")
        print(f"Max hidden layers: {self.max_hidden_layers}")
        print(f"Max hidden size: {self.max_hidden_size}")
        print(f"Metric: {self.metric}")
        if self.metric == 'BIC':
            print("  BIC balances accuracy and model complexity")
        else:
            print("  MSE optimizes for pure prediction accuracy")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(self._objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\nOptimization completed!")
        if self.metric == 'BIC':
            print(f"Best CV BIC: {-best_value:.2f}")
        else:
            print(f"Best CV MSE: {best_value:.4f}")
        
        hidden_sizes = []
        for i in range(best_params.get('num_hidden_layers', 0)):
            hidden_sizes.append(best_params[f'hidden_size_{i}'])
        
        print(f"Best architecture: {[self.n_features] + hidden_sizes + [1]}")
        print(f"Best parameters: {best_params}")
        
        self._save_best_parameters(best_params)        
        return best_params
    
    def _save_best_parameters(self, best_params: dict):
        
        hidden_sizes = []
        for i in range(best_params.get('num_hidden_layers', 0)):
            hidden_sizes.append(best_params[f'hidden_size_{i}'])
        
        # Update config with best parameters
        if 'NETWORK' not in self.config:
            self.config['NETWORK'] = {}
        if 'GP' not in self.config:
            self.config['GP'] = {}
        if 'NORMALIZATION' not in self.config:
            self.config['NORMALIZATION'] = {}
        if 'TRAINING' not in self.config:
            self.config['TRAINING'] = {}
        if 'DEVICE' not in self.config:
            self.config['DEVICE'] = {}
        
        self.config['NETWORK']['input_size'] = str(self.n_features)
        self.config['NETWORK']['output_size'] = '1'
        
        self.config['GP']['num_inducing_points'] = str(best_params['num_inducing_points'])
        self.config['GP']['z_init_low'] = str(best_params['z_init_low'])
        self.config['GP']['z_init_high'] = str(best_params['z_init_high'])
        self.config['GP']['h_init_low'] = str(best_params['h_init_low'])
        self.config['GP']['h_init_high'] = str(best_params['h_init_high'])
        self.config['GP']['global_length_scale'] = str(best_params['global_length_scale'])
        self.config['GP']['min_length_scale'] = str(best_params['min_length_scale'])
        self.config['GP']['global_covariance_scale'] = str(best_params['global_covariance_scale'])
        self.config['GP']['min_covariance_scale'] = str(best_params['min_covariance_scale'])
        self.config['GP']['global_jitter'] = str(best_params['global_jitter'])
        self.config['GP']['baseline_jitter'] = str(best_params['baseline_jitter'])
        
        self.config['NORMALIZATION']['min_var'] = str(best_params['min_var'])
        
        self.config['TRAINING']['seed'] = '42'
        
        self.config['DEVICE']['use_gpu'] = str(self.use_gpu).lower()
        self.config['DEVICE']['device'] = 'gpu' if self.use_gpu else 'cpu'
        self.config['DEVICE']['precision'] = 'float32'
        
        with open(self.config_path, 'w') as f:
            self.config.write(f)
        
        params_to_save = {
            'hidden_sizes': hidden_sizes,
            'best_params': best_params
        }
        
        with open(self.params_save_path, 'wb') as f:
            pickle.dump(params_to_save, f)
        
        print(f"\nSaved hyperparameters:")
        print(f"Config file: {self.config_path}")
        print(f"Params file: {self.params_save_path}")
        print(f"Architecture: {[self.n_features] + hidden_sizes + [1]}")
        print(f"Num inducing points: {best_params['num_inducing_points']}")
        print(f"Length scale: {best_params['global_length_scale']}")
        print(f"Covariance scale: {best_params['global_covariance_scale']}")
        print(f"Jitter: {best_params['global_jitter']}")
        print(f"Min variance: {best_params['min_var']}")
    
    def load_optimized_parameters(self):
        config = ConfigParser()
        config.read(self.config_path)
        
        if os.path.exists(self.params_save_path):
            with open(self.params_save_path, 'rb') as f:
                params_data = pickle.load(f)
        else:
            raise FileNotFoundError(f"Params file not found: {self.params_save_path}")
        
        return config, params_data['hidden_sizes']

def load_gpkan_params_from_file(file_path: str) -> dict:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Params file not found: {file_path}")

def create_optimized_network(config_path: str, params_path: str) -> GP_KAN:
    config = ConfigParser()
    config.read(config_path)
    
    params_data = load_gpkan_params_from_file(params_path)
    hidden_sizes = params_data['hidden_sizes']
    
    return GP_KAN(config, hidden_sizes=hidden_sizes) 