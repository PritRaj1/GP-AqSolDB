import os
import pickle
from configparser import ConfigParser
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from ..core.models.gp_kan import GP_KAN, NormalDist
from ..utils.config_utils import create_default_config
from .base_autotuner import BaseAutoTuner


class GPKANAutoTuner(BaseAutoTuner):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config_path: str = "../config/gp_kan.ini",
        params_save_path: str = "../config/gp_kan_params.pkl",
        metric: str = "BIC",
        n_jobs: int = 2,
        use_gpu: bool = False,
        max_hidden_layers: int = 2,
        max_hidden_size: int = 8,
        num_epochs: int = 50,
        pretrain_iters: int = 10,
        patience: int = 10,
        sampler: str = "bayesian",
        available_acts: Optional[List[str]] = None,
    ) -> None:
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
        pretrain_iters : int
            Number of pretraining iterations for each trial
        patience : int
            Number of epochs to wait before early stopping
        sampler : str
            Optimization sampler: "bayesian" (GPSampler), "tpe" (TPESampler),
            "random" (RandomSampler), or "cmaes" (CmaEsSampler)
        available_acts : List[str], optional
            List of activation functions to consider during optimization
        """
        self.params_save_path = params_save_path
        self.max_hidden_layers = max_hidden_layers
        self.max_hidden_size = max_hidden_size
        self.num_epochs = num_epochs
        self.pretrain_iters = pretrain_iters
        self.patience = patience

        self.available_acts = available_acts or [
            "NormaliseGaussian",
            "ReshapeGaussian",
            "ReduceSumGaussian",
            "None",
        ]

        super().__init__(
            X_train, y_train, config_path, metric, n_jobs, use_gpu, sampler
        )

        self.config = ConfigParser()
        if os.path.exists(config_path):
            self.config.read(config_path)
        else:
            self._create_default_config()

        self._load_device_settings()

    def _load_device_settings(self) -> None:
        try:
            if "DEVICE" in self.config:
                device_section = self.config["DEVICE"]
                use_gpu = device_section.get("use_gpu", "false").lower() == "true"
                precision = device_section.get("precision", "float32")
            else:
                use_gpu = False
                precision = "float32"

            print("Device configuration:")
            print(f"  Use GPU: {use_gpu}")
            print(f"  Precision: {precision}")

            if use_gpu:
                try:
                    gpu_devices = jax.devices("gpu")
                    print(f"  GPU devices available: {len(gpu_devices)}")
                except Exception:
                    print("  GPU devices available: 0")

        except Exception as e:
            print(f"Warning: Could not load device settings: {e}")

    def _create_default_config(self) -> None:
        self.config["NETWORK"] = {
            "input_size": str(self.n_features),
            "output_size": "1",
        }
        self.config["GP"] = {
            "num_inducing_points": "10",
            "z_init_low": "-2.0",
            "z_init_high": "2.0",
            "h_init_low": "-1.0",
            "h_init_high": "1.0",
            "global_length_scale": "0.4",
            "min_length_scale": "0.2",
            "global_covariance_scale": "1.0",
            "min_covariance_scale": "0.1",
            "global_jitter": "0.001",
            "baseline_jitter": "0.01",
        }
        self.config["NORMALIZATION"] = {"min_var": "0.2"}
        self.config["TRAINING"] = {"seed": "42"}
        self.config["DEVICE"] = {
            "use_gpu": str(self.use_gpu).lower(),
            "device": "gpu" if self.use_gpu else "cpu",
            "precision": "float32",
        }

    def calculate_bic(self, mse: float, n_params: int, n_samples: int) -> float:
        """
        Calculate Bayesian Information Criterion (BIC)

        BIC = n * log(MSE) + k * log(n)
        where n = number of samples, k = number of parameters

        Lower BIC is better (penalizes complexity)
        """
        return float(n_samples * np.log(mse) + n_params * np.log(n_samples))

    def _count_network_parameters(self, hidden_sizes: List[int]) -> int:
        layer_sizes = [self.n_features] + hidden_sizes + [1]
        total_params = 0

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            # z: (I, O, P), h: (I, O, P), l: (I, O), s: (I, O), jitter: (I, O)
            num_inducing = int(self.config["GP"].get("num_inducing_points", "10"))
            gp_params = input_size * output_size * (2 * num_inducing + 3)
            total_params += gp_params

            # Normalizer parameters (if not last layer)
            if i < len(layer_sizes) - 2:
                total_params += 2  # mean and variance tracking

        return total_params

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            num_hidden_layers = trial.suggest_int(
                "num_hidden_layers", 0, self.max_hidden_layers
            )
            hidden_sizes = []
            activation_types = []
            activation_params = []

            for i in range(num_hidden_layers):
                hidden_size = trial.suggest_int(
                    f"hidden_size_{i}", 2, self.max_hidden_size
                )
                hidden_sizes.append(hidden_size)

                activation_type = trial.suggest_categorical(
                    f"activation_type_{i}", self.available_acts
                )
                activation_types.append(activation_type)

                activation_param: Dict[str, Any] = {}
                if activation_type == "ReshapeGaussian":
                    activation_param["new_shape"] = [
                        hidden_size,
                        1,
                    ]  # Simple reshape, to allow support (not recommended)
                elif activation_type == "ReduceSumGaussian":
                    activation_param["dim"] = trial.suggest_int(
                        f"reducesum_dim_{i}", -1, 0
                    )
                    activation_param["keep_dim"] = trial.suggest_categorical(
                        f"reducesum_keepdim_{i}", [True, False]
                    )

                activation_params.append(activation_param)

            num_inducing_points = trial.suggest_int("num_inducing_points", 1, 20)
            z_init_low = trial.suggest_float("z_init_low", -2.0, -0.1)
            z_init_high = trial.suggest_float("z_init_high", 0.1, 2.0)
            h_init_low = trial.suggest_float("h_init_low", -2.0, -0.1)
            h_init_high = trial.suggest_float("h_init_high", 0.1, 2.0)

            global_length_scale = trial.suggest_float("global_length_scale", 0.1, 1.0)
            min_length_scale = trial.suggest_float("min_length_scale", 0.1, 1.0)

            global_covariance_scale = trial.suggest_float(
                "global_covariance_scale", 0.1, 1.0
            )
            min_covariance_scale = trial.suggest_float("min_covariance_scale", 0.1, 1.0)

            global_jitter = trial.suggest_float("global_jitter", 1e-3, 1e-1, log=True)
            baseline_jitter = trial.suggest_float(
                "baseline_jitter", 1e-3, 1e-2, log=True
            )

            min_var = trial.suggest_float("min_var", 0.1, 1.0)

            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_int("batch_size", 16, 64)

            config = ConfigParser()
            config["NETWORK"] = {"input_size": str(self.n_features), "output_size": "1"}
            config["GP"] = {
                "num_inducing_points": str(num_inducing_points),
                "z_init_low": str(z_init_low),
                "z_init_high": str(z_init_high),
                "h_init_low": str(h_init_low),
                "h_init_high": str(h_init_high),
                "global_length_scale": str(global_length_scale),
                "min_length_scale": str(min_length_scale),
                "global_covariance_scale": str(global_covariance_scale),
                "min_covariance_scale": str(min_covariance_scale),
                "global_jitter": str(global_jitter),
                "baseline_jitter": str(baseline_jitter),
            }
            config["NORMALIZATION"] = {"min_var": str(min_var)}
            config["TRAINING"] = {"seed": "42"}
            config["DEVICE"] = {
                "use_gpu": str(self.use_gpu).lower(),
                "device": "gpu" if self.use_gpu else "cpu",
                "precision": "float32",
            }

            config["TRAINING"]["learning_rate"] = str(learning_rate)
            config["TRAINING"]["num_epochs"] = str(self.num_epochs)
            config["TRAINING"]["batch_size"] = str(batch_size)
            config["TRAINING"]["pretrain_iters"] = str(self.pretrain_iters)

            cv_results = self._cross_validate_gpkan(
                config, hidden_sizes, activation_types, activation_params, n_splits=3
            )

            if self.metric == "BIC":
                return float(
                    -np.mean(cv_results["bic"])
                )  # Negative because Optuna minimizes
            elif self.metric == "R2":
                return float(
                    -np.mean(cv_results["r2"])
                )  # Negative because Optuna minimizes
            else:
                return float(np.mean(cv_results["mse"]))  # Direct minimization

        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")  # Return large value for failed trials

    def _cross_validate_gpkan(
        self,
        config: ConfigParser,
        hidden_sizes: List[int],
        activation_types: List[str],
        activation_params: List[Dict[str, Any]],
        n_splits: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation for GP-KAN with given hyperparameters

        Parameters:
        -----------
        config : ConfigParser
            Configuration object
        hidden_sizes : List[int]
            List of hidden layer sizes
        activation_types : List[str]
            List of activation function types
        activation_params : List[Dict[str, Any]]
            List of activation function parameters
        n_splits : int
            Number of CV folds

        Returns:
        --------
        dict : Cross-validation results with MSE and BIC
        """

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scores = []
        bic_scores = []
        r2_scores = []

        learning_rate = float(config["TRAINING"].get("learning_rate", "0.001"))
        batch_size = int(config["TRAINING"].get("batch_size", "32"))
        pretrain_iters = int(config["TRAINING"].get("pretrain_iters", "10"))

        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]

            try:
                network = GP_KAN(
                    config,
                    hidden_sizes=hidden_sizes,
                    activation_types=activation_types,
                    activation_params=activation_params,
                )

                network.train(
                    jnp.array(X_train_fold),
                    jnp.array(y_train_fold),
                    jnp.array(X_val_fold),
                    jnp.array(y_val_fold),
                    learning_rate=learning_rate,
                    num_epochs=self.num_epochs,
                    batch_size=batch_size,
                    patience=self.patience,
                    pretrain_iters=pretrain_iters,
                )

                X_val_mean = X_val_fold.astype(np.float32)
                X_val_var = np.zeros_like(X_val_mean)
                X_val_dist = NormalDist(jnp.array(X_val_mean), jnp.array(X_val_var))

                output_dist = network.forward(X_val_dist)
                y_pred = np.array(output_dist.mean.squeeze())

                mse = mean_squared_error(y_val_fold, y_pred)
                r2 = r2_score(y_val_fold, y_pred)
                n_params = self._count_network_parameters(hidden_sizes)
                bic = self.calculate_bic(mse, n_params, len(y_val_fold))

                mse_scores.append(mse)
                bic_scores.append(bic)
                r2_scores.append(r2)

            except Exception as e:
                print(f"CV fold failed: {e}")
                mse_scores.append(1e6)
                bic_scores.append(1e6)
                r2_scores.append(-1e6)

        return {"mse": mse_scores, "bic": bic_scores, "r2": r2_scores}

    def _get_model_name(self) -> str:
        return "GP-KAN"

    def _print_best_parameters(self, best_params: Dict[str, Any]) -> None:
        hidden_sizes = []
        activation_types = []
        activation_params = []

        for i in range(best_params.get("num_hidden_layers", 0)):
            hidden_sizes.append(best_params[f"hidden_size_{i}"])
            activation_types.append(best_params[f"activation_type_{i}"])

            activation_param = {}
            if best_params[f"activation_type_{i}"] == "ReshapeGaussian":
                activation_param["new_shape"] = [best_params[f"hidden_size_{i}"], 1]
            elif best_params[f"activation_type_{i}"] == "ReduceSumGaussian":
                activation_param["dim"] = best_params.get(f"reducesum_dim_{i}", -1)
                activation_param["keep_dim"] = best_params.get(
                    f"reducesum_keepdim_{i}", False
                )

            activation_params.append(activation_param)

        print(f"Best architecture: {[self.n_features] + hidden_sizes + [1]}")
        print(f"Best activation types: {activation_types}")
        print(f"Best parameters: {best_params}")

    def _process_optimization_results(
        self, best_params: Dict[str, Any], best_value: float
    ) -> Dict[str, Any]:
        hidden_sizes = []
        activation_types = []
        activation_params = []

        for i in range(best_params.get("num_hidden_layers", 0)):
            hidden_sizes.append(best_params[f"hidden_size_{i}"])
            activation_types.append(best_params[f"activation_type_{i}"])

            activation_param = {}
            if best_params[f"activation_type_{i}"] == "ReshapeGaussian":
                activation_param["new_shape"] = [best_params[f"hidden_size_{i}"], 1]
            elif best_params[f"activation_type_{i}"] == "ReduceSumGaussian":
                activation_param["dim"] = best_params.get(f"reducesum_dim_{i}", -1)
                activation_param["keep_dim"] = best_params.get(
                    f"reducesum_keepdim_{i}", False
                )

            activation_params.append(activation_param)

        self._save_best_parameters(best_params, activation_types, activation_params)
        return {
            "best_params": best_params,
            "best_value": best_value,
            "hidden_sizes": hidden_sizes,
            "activation_types": activation_types,
            "activation_params": activation_params,
        }

    def _save_best_parameters(
        self,
        best_params: Dict[str, Any],
        activation_types: List[str],
        activation_params: List[Dict[str, Any]],
    ) -> None:

        hidden_sizes = []
        for i in range(best_params.get("num_hidden_layers", 0)):
            hidden_sizes.append(best_params[f"hidden_size_{i}"])

        # Update config with best parameters
        if "NETWORK" not in self.config:
            self.config["NETWORK"] = {}
        if "GP" not in self.config:
            self.config["GP"] = {}
        if "NORMALIZATION" not in self.config:
            self.config["NORMALIZATION"] = {}
        if "TRAINING" not in self.config:
            self.config["TRAINING"] = {}
        if "DEVICE" not in self.config:
            self.config["DEVICE"] = {}

        self.config["NETWORK"]["input_size"] = str(self.n_features)
        self.config["NETWORK"]["output_size"] = "1"

        self.config["GP"]["num_inducing_points"] = str(
            best_params["num_inducing_points"]
        )
        self.config["GP"]["z_init_low"] = str(best_params["z_init_low"])
        self.config["GP"]["z_init_high"] = str(best_params["z_init_high"])
        self.config["GP"]["h_init_low"] = str(best_params["h_init_low"])
        self.config["GP"]["h_init_high"] = str(best_params["h_init_high"])
        self.config["GP"]["global_length_scale"] = str(
            best_params["global_length_scale"]
        )
        self.config["GP"]["min_length_scale"] = str(best_params["min_length_scale"])
        self.config["GP"]["global_covariance_scale"] = str(
            best_params["global_covariance_scale"]
        )
        self.config["GP"]["min_covariance_scale"] = str(
            best_params["min_covariance_scale"]
        )
        self.config["GP"]["global_jitter"] = str(best_params["global_jitter"])
        self.config["GP"]["baseline_jitter"] = str(best_params["baseline_jitter"])

        self.config["NORMALIZATION"]["min_var"] = str(best_params["min_var"])

        self.config["TRAINING"]["seed"] = "42"
        self.config["TRAINING"]["learning_rate"] = str(best_params["learning_rate"])
        self.config["TRAINING"]["num_epochs"] = str(self.num_epochs)
        self.config["TRAINING"]["batch_size"] = str(best_params["batch_size"])
        self.config["TRAINING"]["pretrain_iters"] = str(self.pretrain_iters)

        self.config["DEVICE"]["use_gpu"] = str(self.use_gpu).lower()
        self.config["DEVICE"]["device"] = "gpu" if self.use_gpu else "cpu"
        self.config["DEVICE"]["precision"] = "float32"

        with open(self.config_path, "w") as f:
            self.config.write(f)

        print("Training final network with best hyperparameters...")
        final_network = GP_KAN(
            self.config,
            hidden_sizes=hidden_sizes,
            activation_types=activation_types,
            activation_params=activation_params,
        )

        learning_rate = float(best_params["learning_rate"])
        batch_size = int(best_params["batch_size"])
        pretrain_iters = int(self.pretrain_iters)

        final_network.train(
            jnp.array(self.X_train),
            jnp.array(self.y_train),
            learning_rate=learning_rate,
            num_epochs=self.num_epochs,
            batch_size=batch_size,
            pretrain_iters=pretrain_iters,
        )

        network_params = final_network.get_params()

        params_to_save = {
            "hidden_sizes": hidden_sizes,
            "activation_types": activation_types,
            "activation_params": activation_params,
            "best_params": best_params,
            "network_params": network_params,
        }

        with open(self.params_save_path, "wb") as f:
            pickle.dump(params_to_save, f)

        print("\nSaved hyperparameters and trained network:")
        print(f"Config file: {self.config_path}")
        print(f"Params file: {self.params_save_path}")
        print(f"Architecture: {[self.n_features] + hidden_sizes + [1]}")
        print(f"Activation types: {activation_types}")
        print(f"Num inducing points: {best_params['num_inducing_points']}")
        print(f"Length scale: {best_params['global_length_scale']}")
        print(f"Covariance scale: {best_params['global_covariance_scale']}")
        print(f"Jitter: {best_params['global_jitter']}")
        print(f"Min variance: {best_params['min_var']}")
        print(f"Learning rate: {best_params['learning_rate']}")
        print(f"Batch size: {best_params['batch_size']}")
        print(f"Pretrain iterations: {self.pretrain_iters}")
        print(f"Num epochs: {self.num_epochs}")

    def load_optimized_parameters(
        self,
    ) -> Tuple[ConfigParser, List[int], List[str], List[Dict[str, Any]]]:
        config = ConfigParser()
        config.read(self.config_path)

        if os.path.exists(self.params_save_path):
            with open(self.params_save_path, "rb") as f:
                params_data = pickle.load(f)
        else:
            raise FileNotFoundError(f"Params file not found: {self.params_save_path}")

        return (
            config,
            params_data["hidden_sizes"],
            params_data.get("activation_types", []),
            params_data.get("activation_params", []),
        )


def load_gpkan_params_from_file(file_path: str) -> Dict[str, Any]:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return data if isinstance(data, dict) else {"data": data}
    else:
        raise FileNotFoundError(f"Params file not found: {file_path}")


def create_optimized_network(config_path: str, params_path: str) -> GP_KAN:
    config = ConfigParser()
    config.read(config_path)

    params_data = load_gpkan_params_from_file(params_path)
    hidden_sizes = params_data["hidden_sizes"]
    activation_types = params_data.get("activation_types", [])
    activation_params = params_data.get("activation_params", [])

    network = GP_KAN(
        config,
        hidden_sizes=hidden_sizes,
        activation_types=activation_types,
        activation_params=activation_params,
    )

    if "network_params" in params_data:
        print("Loading saved network parameters...")
        network.set_params(params_data["network_params"])
    else:
        print("No saved network parameters found. Using initialized parameters.")

    return network
