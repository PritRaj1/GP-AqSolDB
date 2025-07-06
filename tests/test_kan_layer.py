import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gp_kan.dense_layer import DenseGPLayer, GPConfig
from src.gp_kan.normal_dist import NormalDist

def test_layer_initialization(sample_gp_config):
    layer = DenseGPLayer(input_size=2, output_size=3, config=sample_gp_config)
    
    assert layer.I == 2, "Input size should be 2"
    assert layer.O == 3, "Output size should be 3"
    assert layer.P == sample_gp_config.num_inducing_points, "Number of inducing points should match config"
    assert layer.num_neurons == 6, "Number of neurons should be input_size * output_size"
    
    params = layer.get_params()
    assert params['z'].shape == (2, 3, 5), "z should have shape (I, O, P)"
    assert params['h'].shape == (2, 3, 5), "h should have shape (I, O, P)"
    assert params['l'].shape == (2, 3), "l should have shape (I, O)"
    assert params['s'].shape == (2, 3), "s should have shape (I, O)"
    assert params['jitter'].shape == (2, 3), "jitter should have shape (I, O)"


def test_layer_parameter_transformations(sample_layer):
    jitter = sample_layer.get_jitter()
    s = sample_layer.get_s()
    l = sample_layer.get_l()
    z = sample_layer.get_z()
    
    assert jitter.shape == (3, 2), "jitter should have shape (I, O)"
    assert s.shape == (3, 2), "s should have shape (I, O)"
    assert l.shape == (3, 2), "l should have shape (I, O)"
    assert z.shape == (3, 2, 5), "z should have shape (I, O, P)"
    
    assert jnp.all(jitter > sample_layer.config.baseline_jitter), "jitter should be positive after exp"
    assert jnp.all(s > sample_layer.config.min_covariance_scale), "s should be positive after exp"
    assert jnp.all(l > sample_layer.config.min_length_scale), "l should be positive after exp"
    assert jnp.all(z >= -1) and jnp.all(z <= 1), "z should be in [-1, 1] after tanh"


def test_forward_pass_basic(sample_layer, sample_input_dist):
    output_dist = sample_layer.forward(sample_input_dist)
    
    assert isinstance(output_dist, NormalDist), "Output should be NormalDist"
    assert output_dist.mean.shape == (1, 2), "Output mean should have shape (batch_size, output_size)"
    assert output_dist.var.shape == (1, 2), "Output variance should have shape (batch_size, output_size)"
    assert jnp.all(jnp.isfinite(output_dist.mean)), "Output mean should be finite"
    assert jnp.all(jnp.isfinite(output_dist.var)), "Output variance should be finite"


@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_forward_pass_batch_sizes(sample_layer, batch_size):
    input_mean = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3))
    input_var = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3))) + 0.1
    input_dist = NormalDist(input_mean, input_var)
    
    output_dist = sample_layer.forward(input_dist)
    
    assert output_dist.mean.shape == (batch_size, 2), f"Output mean should have shape ({batch_size}, 2)"
    assert output_dist.var.shape == (batch_size, 2), f"Output variance should have shape ({batch_size}, 2)"


def test_forward_pass_uncertainty_propagation(sample_layer):

    # Low uncertainty input
    low_uncertainty = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[0.01, 0.01, 0.01]])
    )
    
    # High uncertainty input
    high_uncertainty = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[1.0, 1.0, 1.0]])
    )
    
    low_output = sample_layer.forward(low_uncertainty)
    high_output = sample_layer.forward(high_uncertainty)
    
    assert jnp.mean(high_output.var) > jnp.mean(low_output.var), "Higher input uncertainty should lead to higher output uncertainty"


def test_loglikelihood_computation(sample_layer):
    ll = sample_layer.loglikelihood()
    
    assert isinstance(ll, jax.Array), "Log-likelihood should be a JAX array"
    assert jnp.isfinite(ll), "Log-likelihood should be finite"
    assert ll.shape == (), "Log-likelihood should be a scalar"


def test_parameter_getting_setting(sample_layer):
    original_params = sample_layer.get_params()
    
    new_params = {}
    for key, value in original_params.items():
        new_params[key] = value + 0.1
    
    sample_layer.set_params(new_params)
    retrieved_params = sample_layer.get_params()
    
    for key in original_params.keys():
        assert jnp.allclose(new_params[key], retrieved_params[key]), f"Parameter {key} should be updated"


def test_reset_gp_hyp(sample_layer):
    original_params = sample_layer.get_params()
    
    sample_layer.l = sample_layer.l + 1.0
    sample_layer.s = sample_layer.s + 1.0
    sample_layer.jitter = sample_layer.jitter + 1.0
    
    sample_layer.reset_gp_hyp()
    
    new_params = sample_layer.get_params()
    for key in ['l', 's', 'jitter']:
        assert not jnp.allclose(original_params[key] + 1.0, new_params[key]), f"Parameter {key} should change after reset"


def test_invalid_input_shapes(sample_layer):
    wrong_dim_input = NormalDist(
        jnp.array([0.0, 1.0]),  
        jnp.array([0.1, 0.1])
    )
    
    with pytest.raises(AssertionError):
        sample_layer.forward(wrong_dim_input)
    
    # Wrong input size
    wrong_size_input = NormalDist(
        jnp.array([[0.0, 1.0, 0.5, 0.3]]),
        jnp.array([[0.1, 0.1, 0.1, 0.1]])
    )
    
    with pytest.raises(AssertionError):
        sample_layer.forward(wrong_size_input)


def test_different_configurations():
    config1 = GPConfig(num_inducing_points=3, global_length_scale=0.1)
    config2 = GPConfig(num_inducing_points=7, global_length_scale=0.8)
    
    layer1 = DenseGPLayer(input_size=2, output_size=2, config=config1)
    layer2 = DenseGPLayer(input_size=2, output_size=2, config=config2)
    
    input_dist = NormalDist(
        jnp.array([[0.0, 1.0]]),
        jnp.array([[0.1, 0.1]])
    )
    
    output1 = layer1.forward(input_dist)
    output2 = layer2.forward(input_dist)
    
    assert not jnp.allclose(output1.mean, output2.mean), "Different configs should produce different outputs"
    assert not jnp.allclose(output1.var, output2.var), "Different configs should produce different uncertainties"


def test_individual_gp_distribution(sample_layer):
    x = jnp.array([0.5])
    i_idx, o_idx = 0, 0
    
    gp_dist = sample_layer._DenseGPLayer__gp_dist(x, i_idx, o_idx)
    
    assert isinstance(gp_dist, NormalDist), "Should return NormalDist"
    assert gp_dist.mean.shape == (1,), "Mean should be scalar"
    assert gp_dist.var.shape == (1,), "Variance should be scalar"
    assert jnp.isfinite(gp_dist.mean), "Mean should be finite"
    assert jnp.isfinite(gp_dist.var), "Variance should be finite"


def test_plotting_functionality(sample_layer):
    output_path = "tests/figures/test_plot.png"
    sample_layer.save_fig(output_path, max_neurons_shown=3)
    
    assert os.path.exists(output_path), "Plot file should be created"


def test_layer_repr(sample_layer):
    layer_str = str(sample_layer)
    assert "DenseGPLayer" in layer_str, "String representation should contain class name"
    assert "in=3" in layer_str, "String representation should contain input size"
    assert "out=2" in layer_str, "String representation should contain output size"


def test_config_serialization(sample_gp_config):
    config_dict = sample_gp_config.to_dict()
    assert isinstance(config_dict, dict), "to_dict should return a dictionary"
    assert 'num_inducing_points' in config_dict, "Config should contain num_inducing_points"
    
    new_config = GPConfig.from_dict(config_dict)
    assert new_config.num_inducing_points == sample_gp_config.num_inducing_points, "Config should be reconstructed correctly"
    
    sample_gp_config.update(num_inducing_points=15)
    assert sample_gp_config.num_inducing_points == 15, "Update should modify the config"
    
    with pytest.raises(ValueError):
        sample_gp_config.update(invalid_param=1.0)


def test_random_seed_consistency():
    config = GPConfig(num_inducing_points=5)
    key = jax.random.PRNGKey(42)
    
    layer1 = DenseGPLayer(input_size=2, output_size=2, config=config, key=key)
    layer2 = DenseGPLayer(input_size=2, output_size=2, config=config, key=key)
    
    params1 = layer1.get_params()
    params2 = layer2.get_params()
    
    for key in params1.keys():
        assert jnp.allclose(params1[key], params2[key]), f"Parameter {key} should be identical with same seed"

if __name__ == "__main__":
    print("\nRunning KAN layer tests...")
    
    sample_gp_config = GPConfig(
        num_inducing_points=5,
        z_init_low=-2.0,
        z_init_high=2.0,
        h_init_low=-1.0,
        h_init_high=1.0,
        global_length_scale=0.4,
        min_length_scale=0.2,
        global_covariance_scale=1.0,
        min_covariance_scale=0.1,
        global_jitter=1e-3,
        baseline_jitter=1e-2
    )
    
    sample_input_dist = NormalDist(
        jnp.array([[0.0, 1.0, -0.5]]),
        jnp.array([[0.1, 0.2, 0.15]])
    )
    
    sample_layer = DenseGPLayer(
        input_size=3,
        output_size=2,
        config=sample_gp_config,
        key=jax.random.PRNGKey(42)
    )

    test_layer_initialization(sample_gp_config)
    test_layer_parameter_transformations(sample_layer)
    test_forward_pass_basic(sample_layer, sample_input_dist)
    test_forward_pass_batch_sizes(sample_layer, 3)
    test_forward_pass_uncertainty_propagation(sample_layer)
    test_loglikelihood_computation(sample_layer)
    test_parameter_getting_setting(sample_layer)
    test_reset_gp_hyp(sample_layer)
    test_invalid_input_shapes(sample_layer)
    test_different_configurations()
    test_individual_gp_distribution(sample_layer)
    test_plotting_functionality(sample_layer)
    test_layer_repr(sample_layer)
    test_config_serialization(sample_gp_config)
    test_random_seed_consistency()

    print("All KAN layer tests passed!")