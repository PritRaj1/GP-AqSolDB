import functools
from typing import Optional

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Shared distance computation
# ---------------------------------------------------------------------------


@jax.jit
def _compute_sq_dist(
    X1: jnp.ndarray, X2: jnp.ndarray, sigma: jnp.ndarray
) -> jnp.ndarray:
    """Squared Mahalanobis distance: ||x-y||^2_S with diagonal S = diag(sigma^2)."""
    X1_norm = X1 / sigma
    X2_norm = X2 / sigma
    X1_sq = jnp.sum(X1_norm**2, axis=1, keepdims=True)
    X2_sq = jnp.sum(X2_norm**2, axis=1)
    inner_prod = X1_norm @ X2_norm.T
    return X1_sq + X2_sq - 2 * inner_prod


# ---------------------------------------------------------------------------
# Kernel formulas (pure functions: sq_dist -> kernel matrix)
# ---------------------------------------------------------------------------


@jax.jit
def _rbf_formula(sq_dist: jnp.ndarray, alpha: Optional[float]) -> jnp.ndarray:
    return jnp.exp(-0.5 * sq_dist)


@functools.partial(jax.jit, static_argnums=(1,))
def _rq_formula(sq_dist: jnp.ndarray, alpha: float) -> jnp.ndarray:
    return (1 + 0.5 * sq_dist / alpha) ** (-alpha)


@functools.partial(jax.jit, static_argnums=(1,))
def _matern_formula(sq_dist: jnp.ndarray, alpha: float) -> jnp.ndarray:
    dist = jnp.sqrt(jnp.maximum(sq_dist, 0))

    # alpha is static, so Python if/elif is fine — JAX recompiles per value
    if alpha == 0.5:
        return jnp.exp(-dist)

    elif alpha == 1.5:
        sqrt3_dist = jnp.sqrt(3.0) * dist
        return (1 + sqrt3_dist) * jnp.exp(-sqrt3_dist)

    elif alpha == 2.5:
        sqrt5_dist = jnp.sqrt(5.0) * dist
        return (1 + sqrt5_dist + 5 * dist**2 / 3) * jnp.exp(-sqrt5_dist)

    else:
        raise ValueError(f"Matern alpha={alpha} not supported. Use 0.5, 1.5, or 2.5.")


@jax.jit
def _tps_formula(sq_dist: jnp.ndarray, alpha: Optional[float]) -> jnp.ndarray:
    r = jnp.sqrt(jnp.maximum(sq_dist, 0))
    # r^2 * ln(r), with 0*ln(0) = 0 by convention
    return jnp.where(r > 0, r**2 * jnp.log(r), 0.0)


KERNEL_FORMULAS = {
    "RBF": _rbf_formula,
    "RQ": _rq_formula,
    "MATERN": _matern_formula,
    "TPS": _tps_formula,
}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def compute_kernel(
    kernel_type: str,
    X1: jnp.ndarray,
    X2: jnp.ndarray,
    sigma: jnp.ndarray,
    alpha: Optional[float] = None,
) -> jnp.ndarray:
    """
    Compute kernel matrix K[i,j] = k(X1[i], X2[j]).

    Parameters
    ----------
    kernel_type : str
        One of "RBF", "RQ", "MATERN", "TPS".
    X1 : array, shape (n1, d)
    X2 : array, shape (n2, d)
    sigma : array, shape (d,)
        Per-dimension length scales.
    alpha : float, optional
        Shape parameter for RQ, smoothness for MATERN.

    Returns
    -------
    K : jnp.ndarray, shape (n1, n2)
    """
    if kernel_type not in KERNEL_FORMULAS:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    X1 = jnp.asarray(X1)
    X2 = jnp.asarray(X2)
    sigma = jnp.asarray(sigma)

    sq_dist = _compute_sq_dist(X1, X2, sigma)
    formula = KERNEL_FORMULAS[kernel_type]
    return formula(sq_dist, alpha)
