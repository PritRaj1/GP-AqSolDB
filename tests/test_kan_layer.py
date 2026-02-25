import jax
import jax.numpy as jnp

from src.core.models.gp_kan import (
    DenseGPLayer,
    NormalDist,
    NormaliseGaussian,
    ReduceSumGaussian,
    ReshapeGaussian,
)
from src.utils.config_utils import create_default_config


def test_forward_pass_uncertainty_propagation():
    config = create_default_config()
    config["GP"]["num_inducing_points"] = "5"
    layer = DenseGPLayer(
        input_size=3, output_size=2, config=config, key=jax.random.PRNGKey(42)
    )

    layer.length_scale = jnp.log(jnp.ones_like(layer.length_scale))
    layer.s = jnp.log(jnp.ones_like(layer.s))
    layer.jitter = jnp.log(jnp.ones_like(layer.jitter))

    low_uncertainty = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]), jnp.array([[0.01, 0.01, 0.01]])
    )
    high_uncertainty = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]), jnp.array([[1.0, 1.0, 1.0]])
    )

    low_output = layer.forward(low_uncertainty)
    high_output = layer.forward(high_uncertainty)

    assert jnp.all(jnp.isfinite(low_output.var))
    assert jnp.all(jnp.isfinite(high_output.var))
    assert not jnp.allclose(low_output.var, high_output.var)


def test_activation_functions():
    config = create_default_config()
    config["NORMALIZATION"] = {"min_var": "0.2"}

    # NormaliseGaussian: output should be bounded and variance >= min_var
    norm_act = NormaliseGaussian(min_var=0.2, config=config)
    test_input = NormalDist(
        jnp.array([[1.0, -1.0, 0.5]]), jnp.array([[0.1, 0.2, 0.15]])
    )
    norm_output = norm_act(test_input)

    assert jnp.all(norm_output.var >= 0.2)
    assert jnp.all(norm_output.mean >= -1) and jnp.all(norm_output.mean <= 1)

    # ReshapeGaussian
    reshape_act = ReshapeGaussian(new_shape=[3], config=config)
    reshape_output = reshape_act(test_input)

    assert reshape_output.mean.shape == (3,)
    assert reshape_output.var.shape == (3,)

    # ReduceSumGaussian: sum should be mathematically correct
    reduce_act = ReduceSumGaussian(dim=1, keep_dim=True, config=config)
    reduce_output = reduce_act(test_input)

    assert reduce_output.mean.shape == (1, 1)
    assert jnp.allclose(
        reduce_output.mean, jnp.sum(test_input.mean, axis=1, keepdims=True)
    )
    assert jnp.allclose(
        reduce_output.var, jnp.sum(test_input.var, axis=1, keepdims=True)
    )
