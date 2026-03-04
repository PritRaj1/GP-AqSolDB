from configparser import ConfigParser
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import optax
from jax import grad, jit

from ....utils import load_kan_conf
from .dense_layer import DenseGPLayer
from .invariant_acts import (
    NormaliseGaussian,
    ReduceSumGaussian,
    ReshapeGaussian,
)
from .normal_dist import NormalDist, get_device_config, setup_jax_device


class GP_KAN:
    """Gaussian Process Kolmogorov-Arnold Network."""

    def __init__(
        self,
        config: ConfigParser,
        hidden_sizes: Optional[List[int]] = None,
        activation_types: Optional[List[str]] = None,
        activation_params: Optional[List[Dict[str, Any]]] = None,
        input_noise_log_init: float = -4.0,
    ) -> None:
        self.config = config
        self.layers: List[DenseGPLayer] = []
        self.activations: List[Optional[Any]] = []

        self.device_config = get_device_config(config)
        setup_jax_device(config)

        config_params = load_kan_conf(config)
        self.input_size = config_params["input_size"]
        self.output_size = config_params["output_size"]
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else []
        self.seed = config_params["seed"]

        self.activation_types = activation_types if activation_types is not None else []
        self.activation_params = (
            activation_params if activation_params is not None else []
        )

        if config.has_section("MODEL"):
            init_val = config["MODEL"].get("input_noise_log_init")
            if init_val is not None:
                input_noise_log_init = float(init_val)

        self.input_noise_log = jnp.array(input_noise_log_init, dtype=jnp.float32)

        self._build_network()

    def _build_network(self) -> None:
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            layer = DenseGPLayer(
                input_size=int(layer_sizes[i]),
                output_size=int(layer_sizes[i + 1]),
                config=self.config,
                key=jax.random.PRNGKey(self.seed + i),
            )
            self.layers.append(layer)

            if i < len(layer_sizes) - 2:
                activation = self._create_activation(i)
                self.activations.append(activation)

            else:
                self.activations.append(None)

    def _create_activation(self, layer_idx: int) -> Optional[Any]:
        if layer_idx >= len(self.activation_types):
            min_var = float(self.config["NORMALIZATION"].get("min_var", "0.2"))
            return NormaliseGaussian(
                min_var=min_var, config=self.config
            )  # Default normalizer

        activation_type = self.activation_types[layer_idx]
        activation_param = (
            self.activation_params[layer_idx]
            if layer_idx < len(self.activation_params)
            else {}
        )

        if activation_type == "NormaliseGaussian":
            min_var = activation_param.get(
                "min_var", float(self.config["NORMALIZATION"].get("min_var", "0.2"))
            )
            return NormaliseGaussian(min_var=min_var, config=self.config)

        elif activation_type == "ReshapeGaussian":
            new_shape = activation_param.get("new_shape", [])
            if not new_shape:
                return ReshapeGaussian(
                    new_shape=[-1], config=self.config
                )  # Default falttener
            return ReshapeGaussian(new_shape=new_shape, config=self.config)

        elif activation_type == "ReduceSumGaussian":
            dim = activation_param.get("dim", -1)
            keep_dim = activation_param.get("keep_dim", False)
            return ReduceSumGaussian(dim=dim, keep_dim=keep_dim, config=self.config)

        elif activation_type == "None" or activation_type is None:
            return None

        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def forward(self, x: NormalDist) -> NormalDist:
        current = x

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            current = layer.forward(current)

            if activation is not None:
                current = activation(current)

        return current

    def get_input_noise(self) -> jax.Array:
        return jnp.exp(self.input_noise_log)

    def _input_dist_from_tensor(self, x: jax.Array) -> NormalDist:
        noise_var = self.get_input_noise() ** 2
        input_var = jnp.full_like(x, noise_var)
        return NormalDist(x, input_var)

    def predict(self, x: jax.Array) -> NormalDist:
        return self.forward(self._input_dist_from_tensor(x))

    def get_params(self) -> Dict[str, Any]:
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}"] = layer.get_params()
        return params

    def set_params(self, params: Dict[str, Any]) -> None:
        for i, layer in enumerate(self.layers):
            if f"layer_{i}" in params:
                layer.set_params(params[f"layer_{i}"])

    def loglikelihood(self) -> jax.Array:
        """Expected log-likelihood on the inducing points."""
        total_loglik = jnp.array(0.0)
        count = 0
        for layer in self.layers:
            total_loglik += layer.loglikelihood()
            count += 1
        if count > 0:
            return total_loglik / count
        else:
            return jnp.array(0.0)

    def _condlikelihood(
        self, pred_mean: jax.Array, pred_var: jax.Array, true_val: jax.Array
    ) -> jax.Array:
        pred_var = jnp.maximum(pred_var, 1e-6)  # Positive var

        return jnp.mean(
            -0.5 * jnp.log(2 * jnp.pi * pred_var)
            - 0.5 * ((pred_mean - true_val) ** 2) / pred_var
        )

    def train(
        self,
        X_train: jax.Array,
        y_train: jax.Array,
        X_val: Optional[jax.Array] = None,
        y_val: Optional[jax.Array] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        patience: int = 10,
        pretrain_iters: Optional[int] = None,
    ) -> None:
        """
        Train the GP-KAN network using gradient descent.

        Parameters:
        -----------
        X_train : jax.Array
            Training features
        y_train : jax.Array
            Training targets
        X_val : jax.Array, optional
            Validation features
        y_val : jax.Array, optional
            Validation targets
        learning_rate : float, optional
            Learning rate for optimization (reads from config if None)
        num_epochs : int, optional
            Number of training epochs (reads from config if None)
        batch_size : int, optional
            Batch size for training (reads from config if None)
        patience : int
            Early stopping patience
        pretrain_iters : int, optional
            Number of pretraining iterations for GP hyperparameters
            (reads from config if None)
        """
        if learning_rate is None:
            learning_rate = float(self.config["TRAINING"].get("learning_rate", "0.001"))

        if num_epochs is None:
            num_epochs = int(self.config["TRAINING"].get("num_epochs", "30"))

        if batch_size is None:
            batch_size = int(self.config["TRAINING"].get("batch_size", "32"))

        if pretrain_iters is None:
            pretrain_iters = int(self.config["TRAINING"].get("pretrain_iters", "10"))

        X_train = jnp.array(X_train, dtype=jnp.float32)
        y_train = jnp.array(y_train, dtype=jnp.float32).reshape(-1, 1)

        if X_val is not None:
            X_val = jnp.array(X_val, dtype=jnp.float32)
            y_val = jnp.array(y_val, dtype=jnp.float32).reshape(-1, 1)

        # Pretrain hps
        print("Pretraining GP hyperparameters...")
        try:
            self._pretrain_gp_hyperparameters(X_train, y_train, pretrain_iters)
        except (RuntimeError, ValueError, FloatingPointError) as e:
            print(f"Warning: Pretraining failed: {e}")
            print("Continuing with default hyperparameters...")

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate, momentum=0.9),
        )

        params = self.get_params()
        opt_state = optimizer.init(params)

        key = jax.random.PRNGKey(self.seed)

        def loss_fn(
            params: Dict[str, Any], X_batch: jax.Array, y_batch: jax.Array
        ) -> jax.Array:
            self.set_params(params)

            output_dist = self.predict(X_batch)
            return -self._condlikelihood(output_dist.mean, output_dist.var, y_batch)

        grad_fn = jit(grad(loss_fn))
        loss_fn_jit = jit(loss_fn)

        best_val_loss = float("inf")
        patience_counter = 0
        best_params = None

        for epoch in range(num_epochs):
            try:
                key, subkey = jax.random.split(key)
                indices = jax.random.permutation(subkey, len(X_train))
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]

                total_loss = 0.0
                num_batches = 0

                for i in range(0, len(X_train), batch_size):
                    X_batch = X_train_shuffled[i : i + batch_size]
                    y_batch = y_train_shuffled[i : i + batch_size]

                    grads = grad_fn(params, X_batch, y_batch)
                    updates, opt_state = optimizer.update(grads, opt_state)
                    params = optax.apply_updates(params, updates)

                    loss = loss_fn_jit(params, X_batch, y_batch)
                    total_loss += loss
                    num_batches += 1

                avg_loss = total_loss / num_batches

                if X_val is not None:
                    val_loss = loss_fn_jit(params, X_val, y_val)

                    if epoch % 5 == 0:
                        print(
                            f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, "
                            f"Val Loss = {val_loss:.4f}"
                        )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_params = params.copy()

                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        if best_params is not None:
                            self.set_params(best_params)

                        break

                else:
                    if epoch % 5 == 0:
                        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

            except (RuntimeError, ValueError, FloatingPointError) as e:
                print(f"Training failed at epoch {epoch}: {e}")
                if best_params is not None:
                    self.set_params(best_params)
                break

        if best_params is not None:
            self.set_params(best_params)

        else:
            self.set_params(params)

    def _pretrain_gp_hyperparameters(
        self, X_train: jax.Array, y_train: jax.Array, num_iters: int
    ) -> None:
        """Pretrain GP hyperparameters by maximizing ll of inducing points via L-BFGS."""

        def pretrain_loss_fn(params: Dict[str, Any]) -> jax.Array:
            self.set_params(params)
            return -self.loglikelihood()

        value_and_grad_fn = jit(jax.value_and_grad(pretrain_loss_fn))

        optimizer = optax.lbfgs()
        params = self.get_params()
        opt_state = optimizer.init(params)

        for i in range(num_iters):
            try:
                val, grads = value_and_grad_fn(params)
                updates, opt_state = optimizer.update(
                    grads, opt_state, params,
                    value=val, grad=grads, value_fn=pretrain_loss_fn,
                )
                params = optax.apply_updates(params, updates)

                if i % 5 == 0:
                    print(f"  Pretrain {i}: Inducing point log-likelihood {-val:.4f}")

            except (RuntimeError, ValueError, FloatingPointError) as e:
                print(f"  Pretraining failed at iteration {i}: {e}")
                break

        self.set_params(params)

    def save_fig(self, path: str, max_neurons_per_layer: int = 3) -> None:
        """Save figures for each layer separately."""
        from ....plotting import save_layer_fig

        for layer_idx, layer in enumerate(self.layers):
            layer_path = path.replace(".png", f"_layer_{layer_idx}.png")
            save_layer_fig(layer, layer_path, max_neurons_shown=max_neurons_per_layer)

    def to_device(self, device: str) -> None:
        """Move the entire network to a specific device."""
        if device == "gpu":
            for layer in self.layers:
                layer._move_to_gpu()

            for activation in self.activations:
                if activation is not None and hasattr(activation, "device_config"):
                    activation.device_config["use_gpu"] = True
        else:
            cpu_device = jax.devices("cpu")[0]
            for layer in self.layers:
                layer.z = jax.device_put(layer.z, cpu_device)
                layer.h = jax.device_put(layer.h, cpu_device)
                layer.length_scale = jax.device_put(layer.length_scale, cpu_device)
                layer.s = jax.device_put(layer.s, cpu_device)
                layer.jitter = jax.device_put(layer.jitter, cpu_device)

    def __repr__(self) -> str:
        layer_info = [f"{layer.input_dim}→{layer.output_dim}" for layer in self.layers]
        device_info = f"GPU={self.device_config['use_gpu']}"
        return f"GP_KAN({' → '.join(layer_info)}, {device_info})"
