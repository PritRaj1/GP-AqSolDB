import os
import pickle
from configparser import ConfigParser
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from ..core.models import GP
from .base_autotuner import BaseAutoTuner


class GPAutoTuner(BaseAutoTuner):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config_path: str = "../config/gp.ini",
        sigma_save_path: str = "../config/gp_sigmas.pkl",
        gp_mode: str = "both",
        metric: str = "BIC",
        use_gpu: bool = False,
        sampler: str = "bayesian",
        max_samples: Optional[int] = None,
    ) -> None:
        self.sigma_save_path = sigma_save_path
        self.gp_mode = gp_mode
        self.max_samples = max_samples

        if self.gp_mode not in ["dense", "sparse", "both"]:
            raise ValueError("gp_mode must be 'dense', 'sparse', or 'both'")

        self.X_train_full = X_train.copy()
        self.y_train_full = y_train.copy()

        if max_samples is not None and len(X_train) > max_samples:
            print(f"Subset: {max_samples}/{len(X_train)} samples for autotuning")
            np.random.seed(42)
            subset_idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_train = X_train[subset_idx]
            y_train = y_train[subset_idx]

        super().__init__(X_train, y_train, config_path, metric, use_gpu, sampler)

    def _create_default_config(self) -> ConfigParser:
        from ..utils.config_utils import create_default_config

        return create_default_config(
            n_features=self.n_features,
            use_gpu=self.use_gpu,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            if self.gp_mode == "dense":
                use_sparse = False

            elif self.gp_mode == "sparse":
                use_sparse = True

            else:
                use_sparse = trial.suggest_categorical("use_sparse", [True, False])

            kernel_type = trial.suggest_categorical(
                "kernel_type", ["RBF", "RQ", "MATERN", "TPS"]
            )
            lmbda = trial.suggest_float("lmbda", 1e-3, 0.1, log=True)

            if kernel_type == "RQ":
                alpha = trial.suggest_float("rq_alpha", 0.5, 5.0)

            elif kernel_type == "MATERN":
                alpha = trial.suggest_categorical("matern_nu", [0.5, 1.5, 2.5])

            else:
                alpha = 1.0

            sigmas = [
                trial.suggest_float(f"sigma_{i}", 0.2, 2.0)
                for i in range(self.n_features)
            ]

            if use_sparse:
                min_inducing = max(10, int(0.1 * self.n_samples))
                max_inducing = min(int(0.5 * self.n_samples), self.n_samples - 1)
                num_inducing = trial.suggest_int(
                    "num_inducing", min_inducing, max_inducing
                )
                inducing_method = trial.suggest_categorical(
                    "inducing_method",
                    [
                        "random",
                        "uniform",
                        "kmeans",
                        "kmeans_plus_plus",
                        "stratified",
                        "adaptive",
                        "furthest_point",
                    ],
                )

            else:
                num_inducing = 20
                inducing_method = "random"

            config = ConfigParser()
            config["KERNEL"] = {
                "type": kernel_type,
                "lmbda": str(lmbda),
                "alpha": str(alpha),
            }
            config["SPARSE"] = {
                "use_sparse": str(use_sparse).lower(),
                "num_inducing": str(num_inducing),
                "inducing_method": inducing_method,
            }

            cv_results = self._cross_validate_gp(config, sigmas, n_splits=3)

            if self.metric == "BIC":
                return float(-np.mean(cv_results["bic"]))

            elif self.metric == "R2":
                return float(-np.mean(cv_results["r2"]))

            else:
                return float(np.mean(cv_results["mse"]))

        except (RuntimeError, ValueError) as e:
            print(f"Trial failed: {e}")
            return float("inf")

    def _cross_validate_gp(
        self, config: ConfigParser, sigmas: List[float], n_splits: int = 3
    ) -> Dict[str, List[float]]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scores = []
        bic_scores = []
        r2_scores = []

        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]

            gp = GP(config, np.array(sigmas))
            gp.fit(X_train_fold, y_train_fold)

            n_params = gp.get_model_complexity()

            y_pred = gp.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, np.asarray(y_pred))
            r2 = r2_score(y_val_fold, np.asarray(y_pred))
            bic = self.calculate_bic(mse, n_params, len(y_val_fold))

            mse_scores.append(mse)
            bic_scores.append(bic)
            r2_scores.append(r2)

        return {"mse": mse_scores, "bic": bic_scores, "r2": r2_scores}

    def _get_model_name(self) -> str:
        return "GP"

    def _print_best_parameters(self, best_params: Dict[str, Any]) -> None:
        if self.gp_mode == "dense":
            model_type = "Dense GP"

        elif self.gp_mode == "sparse":
            model_type = "Sparse GP"

        else:
            model_type = (
                "Sparse GP" if best_params.get("use_sparse", False) else "Dense GP"
            )

        print(f"Best model type: {model_type}")
        print(f"Best kernel: {best_params['kernel_type']}")
        print(f"Best lambda: {best_params['lmbda']}")

    def optimize(
        self, n_trials: int = 100, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = super().optimize(n_trials, timeout)
        best_params: Dict[str, Any] = result["best_params"]
        self._save_best_parameters(best_params)

        if self.max_samples is not None and len(self.X_train_full) > self.max_samples:
            print(
                f"Training final model on full dataset "
                f"({len(self.X_train_full)} samples)..."
            )
            config, sigmas = self.load_optimized_parameters()
            gp = GP(config, sigmas)
            gp.fit(self.X_train_full, self.y_train_full)
            print("Final model training completed on full dataset.")

        return best_params

    def _save_best_parameters(self, best_params: Dict[str, Any]) -> None:
        if self.gp_mode == "dense":
            use_sparse = "false"

        elif self.gp_mode == "sparse":
            use_sparse = "true"

        else:
            use_sparse = str(best_params["use_sparse"]).lower()

        kernel_type = best_params["kernel_type"]
        lmbda = best_params["lmbda"]

        if kernel_type == "RQ":
            alpha = best_params.get("rq_alpha", best_params.get("alpha", 1.0))

        elif kernel_type == "MATERN":
            alpha = best_params.get("matern_alpha", best_params.get("alpha", 1.5))

        else:
            alpha = best_params.get("alpha", 1.0)

        if "KERNEL" not in self.config:
            self.config["KERNEL"] = {}

        if "SPARSE" not in self.config:
            self.config["SPARSE"] = {}

        self.config["KERNEL"]["type"] = kernel_type
        self.config["KERNEL"]["lmbda"] = str(lmbda)
        self.config["KERNEL"]["alpha"] = str(alpha)

        self.config["SPARSE"]["use_sparse"] = use_sparse
        if use_sparse == "true":
            num_inducing = best_params.get("num_inducing", 20)
            inducing_method = best_params.get("inducing_method", "random")
            self.config["SPARSE"]["num_inducing"] = str(num_inducing)
            self.config["SPARSE"]["inducing_method"] = inducing_method

        with open(self.config_path, "w") as f:
            self.config.write(f)

        sigmas = [best_params[f"sigma_{i}"] for i in range(self.n_features)]

        with open(self.sigma_save_path, "wb") as f:
            pickle.dump(sigmas, f)

        print(f"Saved: config={self.config_path}, sigmas={self.sigma_save_path}")

    def load_optimized_parameters(self) -> Tuple[ConfigParser, np.ndarray]:
        config = ConfigParser()
        config.read(self.config_path)

        if os.path.exists(self.sigma_save_path):
            with open(self.sigma_save_path, "rb") as f:
                sigmas = pickle.load(f)

        else:
            raise FileNotFoundError(f"Sigma file not found: {self.sigma_save_path}")

        return config, np.array(sigmas)
