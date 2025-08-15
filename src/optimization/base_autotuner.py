import os
import warnings
from abc import ABC, abstractmethod
from configparser import ConfigParser
from typing import Any, Dict, Optional

import numpy as np
import optuna

from ..utils.config_utils import create_default_config

warnings.filterwarnings("ignore")


class BaseAutoTuner(ABC):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config_path: str,
        metric: str = "BIC",
        n_jobs: int = 2,
        use_gpu: bool = False,
        sampler: str = "bayesian",
    ) -> None:
        """
        Common autotuner.

        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        config_path : str
            Path to save configuration file
        metric : str
            Optimization metric: 'BIC', 'MSE', or 'R2'
        n_jobs : int
            Number of jobs for parallelization
        use_gpu : bool
            Whether to use GPU
        sampler : str
            Optimization sampler: "bayesian", "tpe", "random", or "cmaes"
        """
        self.X_train = X_train
        self.y_train = y_train
        self.config_path = config_path
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
        self.metric = metric.upper()
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.sampler = sampler.lower()

        if self.metric not in ["BIC", "MSE", "R2"]:
            raise ValueError("metric must be 'BIC', 'MSE', or 'R2'")

        if self.sampler not in ["bayesian", "tpe", "random", "cmaes"]:
            raise ValueError("sampler must be 'bayesian', 'tpe', 'random', or 'cmaes'")

        self.config = ConfigParser()
        if os.path.exists(config_path):
            self.config.read(config_path)
        else:
            self.config = self._create_default_config()

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        if self.sampler == "bayesian":
            return optuna.samplers.GPSampler(seed=42)
        elif self.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=42)
        elif self.sampler == "random":
            return optuna.samplers.RandomSampler(seed=42)
        elif self.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

    def optimize(
        self, n_trials: int = 100, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        self._print_optimization_start(n_trials)

        study = optuna.create_study(
            direction="minimize",
            sampler=self._get_sampler(),
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(self._objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        best_value = study.best_value

        self._print_optimization_complete(best_value)
        self._print_best_parameters(best_params)

        return self._process_optimization_results(best_params, best_value)

    def _print_optimization_start(self, n_trials: int) -> None:
        print(
            f"Starting {self._get_model_name()} hyperparameter optimization with {n_trials} trials..."
        )
        print(f"Features: {self.n_features}")
        print(f"Samples: {self.n_samples}")
        print(f"Metric: {self.metric}")
        print(f"Sampler: {self.sampler}")
        if self.metric == "BIC":
            print("  BIC balances accuracy and model complexity")
        elif self.metric == "R2":
            print("  R² optimizes for prediction accuracy (higher is better)")
        else:
            print("  MSE optimizes for pure prediction accuracy")

    def _print_optimization_complete(self, best_value: float) -> None:
        print("\nOptimization completed!")
        if self.metric == "BIC":
            print(f"Best CV BIC: {-best_value:.2f}")
        elif self.metric == "R2":
            print(f"Best CV R²: {-best_value:.4f}")
        else:
            print(f"Best CV MSE: {best_value:.4f}")

    def _process_optimization_results(
        self, best_params: Dict[str, Any], best_value: float
    ) -> Dict[str, Any]:
        """Process and return optimization results."""
        return {"best_params": best_params, "best_value": best_value}

    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float:
        pass

    def _create_default_config(self) -> ConfigParser:
        kwargs = {
            "n_features": self.n_features,
            "n_jobs": self.n_jobs,
            "use_gpu": self.use_gpu,
        }

        if hasattr(self, "chunk_size"):
            kwargs["chunk_size"] = self.chunk_size
        if hasattr(self, "min_size_for_parallel"):
            kwargs["min_size_for_parallel"] = self.min_size_for_parallel

        if hasattr(self, "num_epochs"):
            kwargs["num_epochs"] = self.num_epochs
        if hasattr(self, "pretrain_iters"):
            kwargs["pretrain_iters"] = self.pretrain_iters

        return create_default_config(**kwargs)

    @abstractmethod
    def _get_model_name(self) -> str:
        pass

    @abstractmethod
    def _print_best_parameters(self, best_params: Dict[str, Any]) -> None:
        pass
