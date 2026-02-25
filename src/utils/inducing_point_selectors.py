from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit


class InducingPointSelector:
    """Base class for inducing point selection methods."""

    def __init__(self, num_inducing: int, random_state: int = 42):
        self.num_inducing = num_inducing
        self.random_state = random_state
        np.random.seed(random_state)

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select inducing points from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,), optional
            Target values for adaptive selection methods

        Returns
        -------
        np.ndarray, shape (num_inducing, n_features)
            Selected inducing points
        """
        raise NotImplementedError("Subclasses must implement select method")


class RandomSelector(InducingPointSelector):
    """Multinomial sampling of inducing points."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        N = X.shape[0]
        if self.num_inducing >= N:
            return X

        idx = np.random.choice(N, self.num_inducing, replace=False)
        return X[idx]


class UniformSelector(InducingPointSelector):
    """Uniform sampling of inducing points."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        N = X.shape[0]
        if self.num_inducing >= N:
            return X

        step = N // self.num_inducing
        idx = np.arange(0, N, step)[: self.num_inducing]
        return X[idx]


class KMeansSelector(InducingPointSelector):
    """K-means clustering for inducing points."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        kmeans = KMeans(
            n_clusters=self.num_inducing, random_state=self.random_state, n_init=10
        )
        kmeans.fit(X)
        return np.asarray(kmeans.cluster_centers_)


class KMeansPlusPlusSelector(InducingPointSelector):
    """K-means++ clustering for inducing point selection."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        kmeans = KMeans(
            n_clusters=self.num_inducing,
            init="k-means++",
            random_state=self.random_state,
            n_init=10,
        )
        kmeans.fit(X)
        return np.asarray(kmeans.cluster_centers_)


class StratifiedSelector(InducingPointSelector):
    """Stratified sampling based on target values."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if y is None:
            raise ValueError("Target values required for stratified selection")

        n_bins = min(10, self.num_inducing // 2)  # Divide into strata
        y_binned = pd.cut(y, bins=n_bins, labels=False)

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.num_inducing, random_state=self.random_state
        )

        for train_idx, test_idx in splitter.split(X, y_binned):
            return np.asarray(X[test_idx])

        return X[: self.num_inducing]


class AdaptiveSelector(InducingPointSelector):
    """Adaptive selection based on target values and gradients."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if y is None:
            raise ValueError("Target values required for adaptive selection")

        N = X.shape[0]

        # Normalize features for distance calculations
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        y_std = y.std()
        if y_std < 1e-10:
            # Constant targets - use uniform weights
            y_weights = np.ones(N) / N

        else:
            y_centered = y - y.mean()
            y_weights = np.abs(y_centered) / y_std
            y_weights = y_weights / y_weights.sum()

        # Use subsets if lots of samples
        if N > 1000:
            sample_size = min(1000, N // 2)
            sample_idx = np.random.choice(N, sample_size, replace=False)
            X_sample = X_norm[sample_idx]
            distances = cdist(X_norm, X_sample)
            local_density = np.mean(distances, axis=1)

        else:
            distances = cdist(X_norm, X_norm)
            local_density = np.mean(distances, axis=1)

        density_weights = local_density / (local_density.sum() + 1e-8)

        # Combine weights with adaptive weighting
        target_weight = min(0.8, 0.5 + 0.3 * (1000 / max(N, 1000)))
        density_weight = 1.0 - target_weight

        combined_weights = target_weight * y_weights + density_weight * density_weights
        combined_weights = combined_weights / combined_weights.sum()

        # Sample from weighted distribution
        idx = np.random.choice(
            N, size=self.num_inducing, replace=False, p=combined_weights
        )

        return X[idx]


class FurthestPointSelector(InducingPointSelector):
    """Furthest point sampling."""

    def select(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        N = X.shape[0]

        # Normalize features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # For large datasets, just use k-means++
        if N > 1000:
            kmeans = KMeans(
                n_clusters=self.num_inducing,
                init="k-means++",
                random_state=self.random_state,
                n_init=1,
            )
            kmeans.fit(X_norm)
            return np.asarray(kmeans.cluster_centers_)

        # Else use exact furthest point sampling
        first = np.random.randint(0, N)
        selected_idx = [first]
        remaining = set(range(N))
        remaining.discard(first)

        for _ in range(self.num_inducing - 1):
            if not remaining:
                break

            remaining_list = sorted(remaining)
            distances = cdist(X_norm[remaining_list], X_norm[selected_idx])
            min_distances = np.min(distances, axis=1)

            furthest_idx = remaining_list[np.argmax(min_distances)]
            selected_idx.append(furthest_idx)
            remaining.discard(furthest_idx)

        return X[selected_idx]


def get_inducing_selector(
    method: str, num_inducing: int, random_state: int = 42
) -> InducingPointSelector:
    selectors = {
        "random": RandomSelector,
        "uniform": UniformSelector,
        "kmeans": KMeansSelector,
        "kmeans_plus_plus": KMeansPlusPlusSelector,
        "stratified": StratifiedSelector,
        "adaptive": AdaptiveSelector,
        "furthest_point": FurthestPointSelector,
    }

    if method not in selectors:
        raise ValueError(f"Unknown inducing point selection method: {method}")

    return selectors[method](num_inducing, random_state)
