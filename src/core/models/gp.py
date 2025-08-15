from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .dense_gp import DenseGP
from .sparse_gp import FITCGP


class GP:
    def __init__(self, config: Any, sigma: np.ndarray) -> None:
        self.config = config
        self.sigma: np.ndarray = np.asarray(sigma)

        self.use_sparse = config.getboolean("SPARSE", "use_sparse", fallback=False)

        # Only import when needed
        if self.use_sparse:
            self.gp_impl: Union[DenseGP, FITCGP] = FITCGP(config, sigma)
            self.model_type = "sparse"
        else:
            self.gp_impl = DenseGP(config, sigma)
            self.model_type = "dense"

    def __repr__(self) -> str:
        return (
            f"GP(sparse: {self.use_sparse}), "
            f"kernel: {self.config.get('KERNEL', 'type')}, "
            f"gpu: {self.config.get('PARALLEL', 'use_gpu')}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GP":
        if self.use_sparse and isinstance(self.gp_impl, FITCGP):
            inducing_method = self.config.get(
                "SPARSE", "inducing_method", fallback="random"
            )
            self.gp_impl.fit(X, y, inducing_method=inducing_method)
        else:
            self.gp_impl.fit(X, y)

        return self

    def predict(
        self, X_test: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return self.gp_impl.predict(X_test, return_std=return_std)

    def eval_fit(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float, float]:
        if hasattr(self.gp_impl, "eval_fit"):
            return self.gp_impl.eval_fit(y_pred, y_true)
        else:
            # FITCGP doesn't have eval_fit, return default values
            return 0.0, 0.0, 0.0

    def get_cache_stats(self) -> Optional[Dict[str, Union[int, float]]]:
        return self.gp_impl.get_cache_stats()

    def clear_cache(self) -> None:
        return self.gp_impl.clear_cache()

    def get_model_info(self) -> Dict[str, Any]:
        info = {"model_type": self.model_type, "use_sparse": self.use_sparse}

        if self.use_sparse and hasattr(self.gp_impl, "get_sparse_info"):
            sparse_info = self.gp_impl.get_sparse_info()
            if sparse_info:
                info.update(sparse_info)
                info["inducing_method"] = self.config.get(
                    "SPARSE", "inducing_method", fallback="random"
                )

        return info

    def get_model_complexity(self) -> int:
        n_params: int
        if self.use_sparse:
            num_inducing = self.config.getint("SPARSE", "num_inducing", fallback=20)
            n_params = len(self.sigma) + 1  # sigmas + lambda
            if self.config.get("KERNEL", "type") == "RQ":
                n_params += 1  # alpha parameter
            n_params += num_inducing * len(self.sigma)
        else:
            n_params = len(self.sigma) + 1  # sigmas + lambda
            if self.config.get("KERNEL", "type") == "RQ":
                n_params += 1  # alpha parameter
        return n_params
