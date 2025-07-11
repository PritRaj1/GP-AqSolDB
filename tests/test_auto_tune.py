import os
import pickle
import tempfile
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.multivar_gp.auto_tune_gp import GPAutoTuner, load_sigmas_from_file
from src.multivar_gp.dense_gp import DenseGP

sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("paper", font_scale=1.2)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    }
)


def test_auto_tuner_initialization():
    np.random.seed(42)
    for n_features in [1, 2, 3]:
        X_train = np.random.uniform(0, 5, (30, n_features))
        y_train = np.random.normal(0, 1, 30)
        tuner = GPAutoTuner(X_train, y_train)
        assert tuner.X_train.shape == (30, n_features)
        assert tuner.y_train.shape == (30,)
        assert tuner.n_features == n_features
        assert tuner.config is not None
        assert "KERNEL" in tuner.config


def test_auto_tuner_default_config():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (20, 2))
    y_train = np.random.normal(0, 1, 20)

    # Use a non-existent config path to trigger default config creation
    non_existent_config = "non_existent_config.ini"

    try:
        tuner = GPAutoTuner(X_train, y_train, config_path=non_existent_config)
        assert "KERNEL" in tuner.config
        assert tuner.config["KERNEL"]["type"] == "RBF"
        assert tuner.config["KERNEL"]["lmbda"] == "0.1"
        assert tuner.config["KERNEL"]["alpha"] == "1.0"
        assert "SPARSE" in tuner.config
        assert tuner.config["SPARSE"]["use_sparse"] == "false"
    finally:
        if os.path.exists(non_existent_config):
            os.unlink(non_existent_config)


def test_auto_tuner_preserves_existing_config():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (20, 2))
    y_train = np.random.normal(0, 1, 20)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        f.write(
            """
                [KERNEL]
                type="RBF"
                lmbda=0.1
                alpha=1.0

                [TUNING]
                n_trials=50
                timeout=300
                """
        )
        temp_path = f.name
    try:
        tuner = GPAutoTuner(X_train, y_train, config_path=temp_path)
        assert "TUNING" in tuner.config
        assert tuner.config["TUNING"]["n_trials"] == "50"
        assert tuner.config["TUNING"]["timeout"] == "300"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_objective_function():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    for kernel_type in ["RBF", "RQ"]:
        for use_sparse in [True, False]:
            tuner = GPAutoTuner(X_train, y_train)

            class MockTrial:
                def suggest_categorical(self, name, choices):
                    if name == "use_sparse":
                        return use_sparse
                    if name == "kernel_type":
                        return kernel_type
                    return choices[0]

                def suggest_float(self, name, low, high, log=False):
                    if name == "lmbda":
                        return 0.1
                    elif name == "alpha":
                        return 2.0
                    else:
                        return 1.0

                def suggest_int(self, name, low, high):
                    return max(low, min(20, high))

            trial = MockTrial()
            score = tuner._objective(trial)
            assert isinstance(score, float)
            assert not np.isnan(score)
            assert not np.isinf(score)


def test_cross_validation_gp():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    tuner = GPAutoTuner(X_train, y_train)
    config = ConfigParser()
    config["KERNEL"] = {"type": "RBF", "lmbda": "0.1", "alpha": "1.0"}
    sigmas = [1.0, 1.0]
    cv_scores = tuner._cross_validate_gp(config, sigmas, n_splits=3)
    assert len(cv_scores["bic"]) == 3
    assert all(isinstance(score, float) for score in cv_scores["bic"])
    assert all(score >= 0 or score < 0 for score in cv_scores["bic"])
    assert not any(np.isnan(score) for score in cv_scores["bic"])


def test_optimization_small_scale():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            """
                        [KERNEL]
                        type="RBF"
                        lmbda=0.1
                        alpha=1.0

                        [TUNING]
                        n_trials=50
                        timeout=300
                            """
        )
        temp_config_path = temp_config.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        best_params = tuner.optimize(n_trials=5)
        assert isinstance(best_params, dict)
        assert "kernel_type" in best_params
        assert "lmbda" in best_params
        assert "sigma_0" in best_params
        assert "sigma_1" in best_params
        assert "use_sparse" in best_params
        if best_params["use_sparse"]:
            assert "num_inducing" in best_params
            assert "inducing_method" in best_params
        assert os.path.exists(temp_config_path)
        assert os.path.exists(temp_sigma_path)
        config = ConfigParser()
        config.read(temp_config_path)
        assert "TUNING" in config
        assert config["TUNING"]["n_trials"] == "50"
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


def test_save_best_parameters():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            """
                        [KERNEL]
                        type="RBF"
                        lmbda=0.1
                        alpha=1.0

                        [TUNING]
                        n_trials=50
                        timeout=300
                            """
        )
        temp_config_path = temp_config.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        # Test both dense and sparse
        for use_sparse in [True, False]:
            best_params = {
                "kernel_type": "RQ",
                "lmbda": 0.05,
                "alpha": 2.5,
                "sigma_0": 1.2,
                "sigma_1": 0.8,
                "use_sparse": use_sparse,
            }
            if use_sparse:
                best_params["num_inducing"] = 20
                best_params["inducing_method"] = "random"
            tuner._save_best_parameters(best_params)
            config = ConfigParser()
            config.read(temp_config_path)
            assert config["KERNEL"]["type"] == "RQ"
            assert config["KERNEL"]["lmbda"] == "0.05"
            assert config["KERNEL"]["alpha"] == "2.5"
            assert "TUNING" in config
            with open(temp_sigma_path, "rb") as f:
                sigmas = pickle.load(f)
            assert len(sigmas) == 2
            assert sigmas[0] == 1.2
            assert sigmas[1] == 0.8
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


def test_load_optimized_parameters():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            """
                        [KERNEL]
                        type="RBF"
                        lmbda=0.1
                        alpha=1.0

                        [TUNING]
                        n_trials=100
                        timeout=3600
                            """
        )
        temp_config_path = temp_config.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        config = ConfigParser()
        config["KERNEL"] = {"type": "RBF", "lmbda": "0.1", "alpha": "1.0"}
        config["TUNING"] = {"n_trials": "100", "timeout": "3600"}
        with open(temp_config_path, "w") as f:
            config.write(f)
        test_sigmas = [1.5, 0.7]
        with open(temp_sigma_path, "wb") as f:
            pickle.dump(test_sigmas, f)
        loaded_config, loaded_sigmas = tuner.load_optimized_parameters()
        assert loaded_config["KERNEL"]["type"] == "RBF"
        assert loaded_config["KERNEL"]["lmbda"] == "0.1"
        assert loaded_config["TUNING"]["n_trials"] == "100"
        assert np.array_equal(loaded_sigmas, test_sigmas)
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


def test_load_sigmas_from_file():
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    try:
        test_sigmas = [1.2, 0.8, 1.5]
        with open(temp_sigma_path, "wb") as f:
            pickle.dump(test_sigmas, f)
        loaded_sigmas = load_sigmas_from_file(temp_sigma_path)
        assert isinstance(loaded_sigmas, np.ndarray)
        assert np.array_equal(loaded_sigmas, test_sigmas)
    finally:
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


def test_load_sigmas_from_file_not_found():
    try:
        load_sigmas_from_file("non_existent_file.pkl")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_sigma_parameter_count():
    np.random.seed(42)
    for n_features in [1, 2, 3, 4]:
        X_train = np.random.uniform(0, 5, (30, n_features))
        y_train = np.random.normal(0, 1, 30)
        tuner = GPAutoTuner(X_train, y_train)

        class MockTrial:
            def suggest_categorical(self, name, choices):
                if name == "use_sparse":
                    return True
                if name == "kernel_type":
                    return "RBF"
                return choices[0]

            def suggest_float(self, name, low, high, log=False):
                return 1.0

            def suggest_int(self, name, low, high):
                return max(low, min(20, high))

        assert tuner.n_features == n_features


def test_gp_with_optimized_parameters():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            """
                        [KERNEL]
                        type="RBF"
                        lmbda=0.1
                        alpha=1.0

                        [TUNING]
                        n_trials=50
                        timeout=300
                            """
        )
        temp_config_path = temp_config.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        for use_sparse in [True, False]:
            best_params = {
                "kernel_type": "RBF",
                "lmbda": 0.1,
                "alpha": 1.0,
                "sigma_0": 1.2,
                "sigma_1": 0.8,
                "use_sparse": use_sparse,
            }
            if use_sparse:
                best_params["num_inducing"] = 20
                best_params["inducing_method"] = "random"
            tuner._save_best_parameters(best_params)
            config, sigmas = tuner.load_optimized_parameters()
            gp = DenseGP(config, sigmas)
            gp.fit(X_train, y_train)
            X_test = np.random.uniform(0, 5, (10, 2))
            y_pred, y_std = gp.predict(X_test, return_std=True)
            assert len(y_pred) == 10
            assert len(y_std) == 10
            assert np.all(y_std >= 0)
            assert not np.any(np.isnan(y_pred))
            assert not np.any(np.isnan(y_std))
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


def visualize_auto_tune_results():
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config:
        temp_config_path = temp_config.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    try:
        tuner = GPAutoTuner(
            X_train,
            y_train,
            config_path=temp_config_path,
            sigma_save_path=temp_sigma_path,
        )
        tuner.optimize(n_trials=20)
        config, sigmas = tuner.load_optimized_parameters()
        gp = DenseGP(config, sigmas)
        gp.fit(X_train, y_train)
        x1 = np.linspace(0, 5, 50)
        x2 = np.linspace(0, 5, 50)
        X1, X2 = np.meshgrid(x1, x2)
        X_test = np.column_stack([X1.ravel(), X2.ravel()])
        y_pred, y_std = gp.predict(X_test, return_std=True)
        y_pred_grid = y_pred.reshape(X1.shape)
        y_std_grid = y_std.reshape(X1.shape)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axes[0].contourf(
            X1, X2, y_pred_grid, levels=25, cmap="viridis", alpha=0.8
        )
        axes[0].scatter(
            X_train[:, 0],
            X_train[:, 1],
            c="red",
            s=30,
            marker="x",
            label="Training Data",
        )
        axes[0].set_title(
            "Optimized GP Prediction\nKernel: "
            + config["KERNEL"]["type"]
            + r", $\lambda$: "
            + config["KERNEL"]["lmbda"]
        )
        axes[0].set_xlabel(r"$x_1$")
        axes[0].set_ylabel(r"$x_2$")
        axes[0].legend()
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        im2 = axes[1].contourf(X1, X2, y_std_grid, levels=25, cmap="plasma", alpha=0.8)
        axes[1].scatter(
            X_train[:, 0],
            X_train[:, 1],
            c="lime",
            s=30,
            marker="x",
            label="Training Data",
        )
        axes[1].set_title("Prediction Uncertainty\n" + r"$\sigma$: " + str(sigmas))
        axes[1].set_xlabel(r"$x_1$")
        axes[1].set_ylabel(r"$x_2$")
        axes[1].legend()
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        plt.tight_layout()
        plt.savefig(
            "tests/figures/auto_tune_results.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)


if __name__ == "__main__":
    print("\nRunning auto-tuning tests...")
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) * np.exp(X_train[:, 1] / 5) + np.random.normal(
        0, 0.1, 50
    )
    sample_data_2d = (X_train, y_train)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ini", delete=False
    ) as temp_config:
        temp_config.write(
            """
                        [KERNEL]
                        type="RBF"
                        lmbda=0.1
                        alpha=1.0

                        [TUNING]
                        n_trials=50
                        timeout=300
                            """
        )
        temp_config_path = temp_config.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_sigma:
        temp_sigma_path = temp_sigma.name
    try:
        test_auto_tuner_initialization()
        test_objective_function()
        test_cross_validation_gp()
        test_auto_tuner_default_config()
        test_auto_tuner_preserves_existing_config()
        test_save_best_parameters()
        test_load_optimized_parameters()
        test_load_sigmas_from_file()
        test_load_sigmas_from_file_not_found()
        test_sigma_parameter_count()
        test_gp_with_optimized_parameters()
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        if os.path.exists(temp_sigma_path):
            os.unlink(temp_sigma_path)
    print("All tests passed!")
    print("Creating visualizations...")
    visualize_auto_tune_results()
    print("Visualizations saved to tests/figures/auto_tune_results.png")
