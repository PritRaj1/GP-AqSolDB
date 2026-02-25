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
        self.params_save_path = params_save_path
        self.max_hidden_layers = max_hidden_layers
        self.max_hidden_size = max_hidden_size
        self.num_epochs = num_epochs
        self.pretrain_iters = pretrain_iters
        self.patience = patience

        self.available_acts = available_acts or [
            "NormaliseGaussian",
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
        if "DEVICE" in self.config:
            use_gpu = self.config["DEVICE"].get("use_gpu", "false").lower() == "true"
            precision = self.config["DEVICE"].get("precision", "float32")

        else:
            use_gpu = False
            precision = "float32"

        print(f"Device config: GPU={use_gpu}, precision={precision}")

        if use_gpu:
            try:
                gpu_devices = jax.devices("gpu")
                print(f"  GPU devices available: {len(gpu_devices)}")
            except RuntimeError:
                print("  GPU devices available: 0")

    def _create_default_config(self) -> ConfigParser:
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
        return self.config

    @staticmethod
    def _make_activation_param(
        act_type: str, hidden_size: int, dim: int = -1, keep_dim: bool = False
    ) -> Dict[str, Any]:
        if act_type == "ReshapeGaussian":
            return {"new_shape": [hidden_size, 1]}
        elif act_type == "ReduceSumGaussian":
            return {"dim": dim, "keep_dim": keep_dim}
        return {}

    def _extract_architecture(
        self, best_params: Dict[str, Any]
    ) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
        hidden_sizes = []
        activation_types = []
        activation_params = []

        for i in range(best_params.get("num_hidden_layers", 0)):
            hidden_sizes.append(best_params[f"hidden_size_{i}"])
            act_type = best_params[f"activation_type_{i}"]
            activation_types.append(act_type)
            activation_params.append(
                self._make_activation_param(
                    act_type,
                    best_params[f"hidden_size_{i}"],
                    dim=best_params.get(f"reducesum_dim_{i}", -1),
                    keep_dim=best_params.get(f"reducesum_keepdim_{i}", False),
                )
            )

        return hidden_sizes, activation_types, activation_params

    def _count_network_parameters(self, hidden_sizes: List[int]) -> int:
        layer_sizes = [self.n_features] + hidden_sizes + [1]
        total_params = 0

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            num_inducing = int(self.config["GP"].get("num_inducing_points", "10"))
            gp_params = input_size * output_size * (2 * num_inducing + 3)
            total_params += gp_params
            if i < len(layer_sizes) - 2:
                total_params += 2

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

                dim = -1
                keep_dim = False
                if activation_type == "ReduceSumGaussian":
                    dim = trial.suggest_int(f"reducesum_dim_{i}", -1, 0)
                    keep_dim = trial.suggest_categorical(
                        f"reducesum_keepdim_{i}", [True, False]
                    )
                activation_params.append(
                    self._make_activation_param(
                        activation_type, hidden_size, dim=dim, keep_dim=keep_dim
                    )
                )

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
            config["TRAINING"] = {
                "seed": "42",
                "learning_rate": str(learning_rate),
                "num_epochs": str(self.num_epochs),
                "batch_size": str(batch_size),
                "pretrain_iters": str(self.pretrain_iters),
            }
            config["DEVICE"] = {
                "use_gpu": str(self.use_gpu).lower(),
                "device": "gpu" if self.use_gpu else "cpu",
                "precision": "float32",
            }

            cv_results = self._cross_validate_gpkan(
                config, hidden_sizes, activation_types, activation_params, n_splits=3
            )

            if self.metric == "BIC":
                return float(-np.mean(cv_results["bic"]))

            elif self.metric == "R2":
                return float(-np.mean(cv_results["r2"]))

            else:
                return float(np.mean(cv_results["mse"]))

        except (ValueError, RuntimeError) as e:
            print(f"Trial failed: {e}")
            return float("inf")

    def _cross_validate_gpkan(
        self,
        config: ConfigParser,
        hidden_sizes: List[int],
        activation_types: List[str],
        activation_params: List[Dict[str, Any]],
        n_splits: int = 5,
    ) -> Dict[str, List[float]]:
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

            except (ValueError, RuntimeError) as e:
                print(f"CV fold failed: {e}")
                mse_scores.append(1e6)
                bic_scores.append(1e6)
                r2_scores.append(-1e6)

        return {"mse": mse_scores, "bic": bic_scores, "r2": r2_scores}

    def _get_model_name(self) -> str:
        return "GP-KAN"

    def _print_best_parameters(self, best_params: Dict[str, Any]) -> None:
        hidden_sizes, activation_types, _ = self._extract_architecture(best_params)
        print(f"Best architecture: {[self.n_features] + hidden_sizes + [1]}")
        print(f"Best activations: {activation_types}")

    def _process_optimization_results(
        self, best_params: Dict[str, Any], best_value: float
    ) -> Dict[str, Any]:
        hidden_sizes, activation_types, activation_params = self._extract_architecture(
            best_params
        )
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
        hidden_sizes, _, _ = self._extract_architecture(best_params)

        self.config["NETWORK"] = {
            "input_size": str(self.n_features),
            "output_size": "1",
        }
        self.config["GP"] = {
            "num_inducing_points": str(best_params["num_inducing_points"]),
            "z_init_low": str(best_params["z_init_low"]),
            "z_init_high": str(best_params["z_init_high"]),
            "h_init_low": str(best_params["h_init_low"]),
            "h_init_high": str(best_params["h_init_high"]),
            "global_length_scale": str(best_params["global_length_scale"]),
            "min_length_scale": str(best_params["min_length_scale"]),
            "global_covariance_scale": str(best_params["global_covariance_scale"]),
            "min_covariance_scale": str(best_params["min_covariance_scale"]),
            "global_jitter": str(best_params["global_jitter"]),
            "baseline_jitter": str(best_params["baseline_jitter"]),
        }
        self.config["NORMALIZATION"] = {"min_var": str(best_params["min_var"])}
        self.config["TRAINING"] = {
            "seed": "42",
            "learning_rate": str(best_params["learning_rate"]),
            "num_epochs": str(self.num_epochs),
            "batch_size": str(best_params["batch_size"]),
            "pretrain_iters": str(self.pretrain_iters),
        }
        self.config["DEVICE"] = {
            "use_gpu": str(self.use_gpu).lower(),
            "device": "gpu" if self.use_gpu else "cpu",
            "precision": "float32",
        }

        with open(self.config_path, "w") as f:
            self.config.write(f)

        print("Training final network with best hyperparameters...")
        final_network = GP_KAN(
            self.config,
            hidden_sizes=hidden_sizes,
            activation_types=activation_types,
            activation_params=activation_params,
        )

        final_network.train(
            jnp.array(self.X_train),
            jnp.array(self.y_train),
            learning_rate=float(best_params["learning_rate"]),
            num_epochs=self.num_epochs,
            batch_size=int(best_params["batch_size"]),
            pretrain_iters=self.pretrain_iters,
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

        print(f"Saved: config={self.config_path}, params={self.params_save_path}")

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
