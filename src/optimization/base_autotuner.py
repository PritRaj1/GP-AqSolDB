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
        use_gpu: bool = False,
        sampler: str = "bayesian",
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.config_path = config_path
        self.n_features = X_train.shape[1]
        self.n_samples = X_train.shape[0]
        self.metric = metric.upper()
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
            return optuna.samplers.GPSampler(seed=42, warn_independent_sampling=False)

        elif self.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=42)

        elif self.sampler == "random":
            return optuna.samplers.RandomSampler(seed=42)

        elif self.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)

        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

    def calculate_bic(self, mse: float, n_params: int, n_samples: int) -> float:
        """BIC = n * log(MSE) + k * log(n). Lower is better."""
        return float(n_samples * np.log(mse) + n_params * np.log(n_samples))

    def optimize(
        self, n_trials: int = 100, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        print(
            f"Starting {self._get_model_name()} optimization: "
            f"{n_trials} trials, {self.n_features} features, "
            f"{self.n_samples} samples, metric={self.metric}, sampler={self.sampler}"
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=self._get_sampler(),
            pruner=optuna.pruners.MedianPruner(),
        )

        study.optimize(self._objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        best_value = study.best_value

        print(f"Optimization complete. Best {self.metric}: {best_value:.4f}")
        self._print_best_parameters(best_params)

        return {"best_params": best_params, "best_value": best_value}

    @abstractmethod
    def _objective(self, trial: optuna.Trial) -> float:
        pass

    def _create_default_config(self) -> ConfigParser:
        return create_default_config(
            n_features=self.n_features,
            use_gpu=self.use_gpu,
        )

    @abstractmethod
    def _get_model_name(self) -> str:
        pass

    @abstractmethod
    def _print_best_parameters(self, best_params: Dict[str, Any]) -> None:
        pass
