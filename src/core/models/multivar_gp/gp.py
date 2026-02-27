from typing import Any, Dict, Tuple, Union

import jax.numpy as jnp
import numpy as np

from .dense_gp import DenseGP
from .fitc_gp import FITCGP

ArrayLike = Union[np.ndarray, jnp.ndarray]


class GP:
    def __init__(self, config: Any, sigma: ArrayLike) -> None:
        self.config = config
        self.sigma: jnp.ndarray = jnp.asarray(sigma)
        self.use_sparse = config.getboolean("SPARSE", "use_sparse", fallback=False)

        sigma_jnp = jnp.asarray(sigma)
        if self.use_sparse:
            self.gp_impl: Union[DenseGP, FITCGP] = FITCGP(config, sigma_jnp)
            self.model_type = "sparse"
        else:
            self.gp_impl = DenseGP(config, sigma_jnp)
            self.model_type = "dense"

    def __repr__(self) -> str:
        return (
            f"GP(sparse: {self.use_sparse}), "
            f"kernel: {self.config.get('KERNEL', 'type')}"
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GP":
        X_jnp = jnp.asarray(X)
        y_jnp = jnp.asarray(y)
        if self.use_sparse and isinstance(self.gp_impl, FITCGP):
            inducing_method = self.config.get(
                "SPARSE", "inducing_method", fallback="random"
            )
            self.gp_impl.fit(X_jnp, y_jnp, inducing_method=inducing_method)

        else:
            self.gp_impl.fit(X_jnp, y_jnp)

        return self

    def predict(
        self, X_test: ArrayLike, return_std: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        return self.gp_impl.predict(jnp.asarray(X_test), return_std=return_std)

    def get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "model_type": self.model_type,
            "use_sparse": self.use_sparse,
        }
        if self.use_sparse and hasattr(self.gp_impl, "get_sparse_info"):
            sparse_info = self.gp_impl.get_sparse_info()
            if sparse_info:
                info.update(sparse_info)
                info["inducing_method"] = self.config.get(
                    "SPARSE", "inducing_method", fallback="random"
                )

        return info

    def get_model_complexity(self) -> int:
        return self.gp_impl.get_model_complexity()
