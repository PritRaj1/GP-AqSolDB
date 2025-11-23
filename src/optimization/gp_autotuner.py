import os
import pickle
from configparser import ConfigParser
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from ..core.kernels import get_cache_stats, get_parallel_info, load_parallel_conf
from ..core.models import GP
from .base_autotuner import BaseAutoTuner


class GPAutoTuner(BaseAutoTuner):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config_path: str = "../config/gp.ini",
        sigma_save_path: str = "../config/gp_sigmas.pkl",
        gp_mode: str = "both",  # "dense", "sparse", "both"
        metric: str = "BIC",
        n_jobs: int = 2,
        use_gpu: bool = False,
        chunk_size: int = 500,
        min_size_for_parallel: int = 1000,
        sampler: str = "bayesian",  # "bayesian", "tpe", "random", "cmaes"
        max_samples: Optional[int] = None,
    ) -> None:
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
        gp_mode : str
            GP optimization mode: "dense" (dense only), "sparse" (sparse only),
            "both" (optimize both types)
        metric : str
            Optimization metric: 'BIC' or 'MSE'
        n_jobs : int
            Number of jobs for kernel parallelization
        use_gpu : bool
            Whether to use GPU
        chunk_size : int
            Size of chunks for parallel processing
        min_size_for_parallel : int
            Minimum matrix size to trigger parallel processing
        sampler : str
            Optimization sampler: "bayesian" (GPSampler), "tpe" (TPESampler),
            "random" (RandomSampler), or "cmaes" (CmaEsSampler)
        max_samples : Optional[int]
            If provided and dataset is larger, a random subset will be used for tuning.
            Final model will still be trained on full dataset.
        """
        self.sigma_save_path = sigma_save_path
        self.gp_mode = gp_mode
        self.chunk_size = chunk_size
        self.min_size_for_parallel = min_size_for_parallel
        self.max_samples = max_samples

        if self.gp_mode not in ["dense", "sparse", "both"]:
            raise ValueError("gp_mode must be 'dense', 'sparse', or 'both'")

        self.X_train_full = X_train.copy()
        self.y_train_full = y_train.copy()

        if max_samples is not None and len(X_train) > max_samples:
            print(f"Using subset of {max_samples} samples for autotuning...")
            print(f"Full dataset size: {len(X_train)} samples")
            np.random.seed(42)
            subset_idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_train = X_train[subset_idx]
            y_train = y_train[subset_idx]

        super().__init__(
            X_train, y_train, config_path, metric, n_jobs, use_gpu, sampler
        )

        self._load_parallel_settings()

    def _load_parallel_settings(self) -> None:
        try:
            load_parallel_conf(self.config)
            parallel_info = get_parallel_info()

            print("Parallelization configuration:")
            print(f"  CPU cores available: {parallel_info['cpu_cores']}")
            print(f"  GPU available: {parallel_info['gpu_available']}")
            if parallel_info["gpu_available"]:
                if parallel_info["cupy_available"]:
                    print("  GPU backend: CuPy")

            settings = parallel_info["settings"]
            print(f"  Use parallel: {settings['use_parallel']}")
            print(f"  Jobs: {settings['n_jobs'] if settings['n_jobs'] else 'auto'}")
            print(f"  Use GPU: {settings['use_gpu']}")
            print(f"  Chunk size: {settings['chunk_size']}")
            print(f"  Min size for parallel: {settings['min_size_for_parallel']}")

        except Exception as e:
            print(f"Warning: Could not load parallel settings: {e}")

    def calculate_bic(self, mse: float, n_params: int, n_samples: int) -> float:
        """
        Calculate Bayesian Information Criterion (BIC)

        BIC = n * log(MSE) + k * log(n)
        where n = number of samples, k = number of parameters

        Lower BIC is better (penalizes complexity)
        """
        return float(n_samples * np.log(mse) + n_params * np.log(n_samples))

    def _objective(self, trial: optuna.Trial) -> float:
        try:
            if self.gp_mode == "dense":
                use_sparse = False
            elif self.gp_mode == "sparse":
                use_sparse = True
            else:
                use_sparse = trial.suggest_categorical("use_sparse", [True, False])

            kernel_type = trial.suggest_categorical(
                "kernel_type", ["RBF", "RQ", "MATERN"]
            )
            lmbda = trial.suggest_float("lmbda", 1e-3, 0.1, log=True)

            if kernel_type == "RQ":
                alpha = trial.suggest_float("rq_alpha", 0.5, 5.0)
            elif kernel_type == "MATERN":
                alpha = trial.suggest_categorical("matern_nu", [0.5, 1.5, 2.5])
            else:  # RBF
                alpha = 1.0

            sigmas = [
                trial.suggest_float(f"sigma_{i}", 0.2, 2.0)
                for i in range(self.n_features)
            ]

            # Sparse GP specific parameters - suggest number of inducing points
            # (between 10% and 50% of data size)
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
                "use_cache": "true",
                "cache_size": "100",
            }
            config["SPARSE"] = {
                "use_sparse": str(use_sparse).lower(),
                "num_inducing": str(num_inducing),
                "inducing_method": inducing_method,
            }

            if "PARALLEL" in self.config:
                config["PARALLEL"] = dict(self.config["PARALLEL"])

            cv_results = self._cross_validate_gp(config, sigmas, n_splits=3)

            if self.metric == "BIC":
                return float(-np.mean(cv_results["bic"]))  # Negative for minimization
            elif self.metric == "R2":
                return float(-np.mean(cv_results["r2"]))  # Negative for minimization
            else:
                return float(np.mean(cv_results["mse"]))  # Direct minimization

        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")  # Return large value for failed trials

    def _cross_validate_gp(
        self, config: ConfigParser, sigmas: List[float], n_splits: int = 3
    ) -> Dict[str, List[float]]:
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
            mse = mean_squared_error(y_val_fold, y_pred)
            r2 = r2_score(y_val_fold, y_pred)
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
        if best_params["kernel_type"] == "MATERN":
            print(f"Best alpha: {best_params.get('matern_alpha', 1.5)}")
        elif best_params["kernel_type"] == "RQ":
            print(f"Best alpha: {best_params.get('rq_alpha', 1.0)}")
        print(f"Best lambda: {best_params['lmbda']}")
        print(f"Best parameters: {best_params}")

        # Report cache stats
        try:
            cache_stats = get_cache_stats()
            total_requests = cache_stats["hits"] + cache_stats["misses"]
            if total_requests > 0:
                print("\nKernel cache statistics:")
                print(f"  Cache hits: {cache_stats['hits']}")
                print(f"  Cache misses: {cache_stats['misses']}")
                print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
                print(f"  Cache size: {cache_stats['cache_size']}")
        except Exception:
            pass

    def optimize(
        self, n_trials: int = 100, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = super().optimize(n_trials, timeout)
        best_params: Dict[str, Any] = result["best_params"]
        self._save_best_parameters(best_params)

        # Train final model on full dataset
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

        print("\nSaved hyperparameters:")
        print(f"Config file: {self.config_path}")
        print(f"Sigma file: {self.sigma_save_path}")
        print(f"Model type: {'Sparse' if use_sparse == 'true' else 'Dense'}")
        print(f"Kernel type: {kernel_type}")
        print(f"Lambda: {lmbda}")
        if kernel_type == "MATERN":
            print(f"Alpha: {alpha}")
        elif kernel_type == "RQ":
            print(f"Alpha: {alpha}")
        print(f"Sigmas: {sigmas}")
        if use_sparse == "true":
            print(f"Number of inducing points: {best_params.get('num_inducing', 20)}")
            print(f"Inducing method: {best_params.get('inducing_method', 'random')}")

        preserved_sections = [
            section
            for section in self.config.sections()
            if section not in ["KERNEL", "SPARSE"]
        ]
        if preserved_sections:
            print(f"Preserved sections: {preserved_sections}")

    def load_optimized_parameters(self) -> Tuple[ConfigParser, np.ndarray]:
        config = ConfigParser()
        config.read(self.config_path)

        if os.path.exists(self.sigma_save_path):
            with open(self.sigma_save_path, "rb") as f:
                sigmas = pickle.load(f)
        else:
            raise FileNotFoundError(f"Sigma file not found: {self.sigma_save_path}")

        return config, np.array(sigmas)


def load_sigmas_from_file(file_path: str) -> np.ndarray:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return np.array(pickle.load(f))
    else:
        raise FileNotFoundError(f"Sigma file not found: {file_path}")
