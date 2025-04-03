"""
This module contains the implementation of a Random Forest Regressor.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- runRandomForest: A class that runs the Random Forest algorithm.
"""

# Importing the required libraries
import multiprocessing
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from .treeRegressor import RegressorTree


def _fit_tree(X, y, max_depth):
    """
    Helper function for parallel tree fitting. Fits a single tree on a bootstrapped sample.

    Args:
        X (array-like): The input features.
        y (array-like): The target labels.
        max_depth (int): The maximum depth of the tree.

    Returns:
        RegressorTree: A fitted tree object.
    """
    # Create bootstrapped sample
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_sample = X[indices]
    y_sample = y[indices]

    # Fit tree on bootstrapped sample
    tree = RegressorTree(max_depth=max_depth)
    return tree.learn(X_sample, y_sample)


def _predict_oob(X, trees, bootstraps):
    """
    Helper function for parallel out-of-bag predictions. Predicts using out-of-bag samples.

    Args:
        X (array-like): The input features.
        trees (list): The list of fitted trees.
        bootstraps (list): The list of bootstrapped indices for each tree.

    Returns:
        list: The list of out-of-bag predictions.
    """
    all_predictions = []

    for i, record in enumerate(X):
        predictions = []
        for _j, (tree, bootstrap) in enumerate(zip(trees, bootstraps, strict=False)):
            # Check if record is out-of-bag for this tree
            if i not in bootstrap:
                predictions.append(RegressorTree.evaluate_tree(tree, record))

        if predictions:
            all_predictions.append(np.mean(predictions))
        else:
            # If not out-of-bag for any tree, use all trees
            all_predictions.append(
                np.mean([RegressorTree.evaluate_tree(tree, record) for tree in trees])
            )

    return all_predictions


class RandomForestRegressor:
    """
    A class representing a Random Forest model for regression.

    Atributes:
        forest_size (int): The number of trees in the forest.
        max_depth (int): The maximum depth of each tree.
        n_jobs (int): The number of jobs to run in parallel.
        random_seed (int): Seed for random number generation.
        X (array-like): The input features.
        y (array-like): The target labels.

    Methods:
        fit(X=None, y=None, verbose=False): Fits the random forest to the data.
        calculate_metrics(y_true, y_pred): Calculates the evaluation metrics.
        predict(X): Predicts the target values for the input features.
        get_stats(verbose=False): Returns the evaluation metrics.
    """

    def __init__(
        self, forest_size=100, max_depth=10, n_jobs=-1, random_seed=None, X=None, y=None
    ):
        """Initialize the Random Forest Regressor with optimized parameters."""
        self.n_estimators = forest_size
        self.max_depth = max_depth
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, multiprocessing.cpu_count())
        self.random_state = random_seed
        self.trees = []
        self.bootstraps = []

        self.X = X
        self.y = y

    def fit(self, X=None, y=None, verbose=False):
        """Fit the random forest with parallel processing."""
        if X is None and self.X is None:
            raise ValueError(
                "X must be provided either during initialization or fitting."
            )
        if y is None and self.y is None:
            raise ValueError(
                "y must be provided either during initialization or fitting."
            )

        start_time = datetime.now()

        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Convert inputs to numpy arrays
        X = np.asarray(X if X is not None else self.X)
        y = np.asarray(y if y is not None else self.y)

        #  If X or y are empty, raise an error
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty.")

        if verbose:
            print("Fitting trees in parallel...")

        # Fit trees in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(X, y, self.max_depth) for _ in range(self.n_estimators)
        )

        # Generate bootstrapped indices for OOB scoring
        self.bootstraps = [
            np.random.choice(len(X), size=len(X), replace=True)
            for _ in range(self.n_estimators)
        ]

        # Compute OOB predictions
        if verbose:
            print("Computing OOB predictions...")

        y_pred = _predict_oob(X, self.trees, self.bootstraps)

        # Calculate evaluation metrics
        self.calculate_metrics(y, y_pred)

        if verbose:
            print(f"Execution time: {datetime.now() - start_time}")
            print(f"MSE:  {self.mse:.4f}")
            print(f"R^2:  {self.r2:.4f}")
            print(f"MAPE: {self.mape:.4f}%")
            print(f"MAE:  {self.mae:.4f}")
            print(f"RMSE: {self.rmse:.4f}")

        return self

    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        self.mse = np.mean((y_true - y_pred) ** 2)
        self.ssr = np.sum((y_true - y_pred) ** 2)
        self.sst = np.sum((y_true - np.mean(y_true)) ** 2)
        self.r2 = 1 - (self.ssr / self.sst) if self.sst != 0 else 0

        # Handle zero values in y_true for MAPE
        mask = y_true != 0
        if np.any(mask):
            self.mape = (
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            )
        else:
            self.mape = np.nan

        self.mae = np.mean(np.abs(y_true - y_pred))
        self.rmse = np.sqrt(self.mse)

    def predict(self, X):
        """Predict using the trained random forest."""
        X = np.asarray(X)

        if X.size == 0:
            return np.array([])

        # Validate input dimensions
        if self.X is not None and X.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Input data must have {self.X.shape[1]} features, but got {X.shape[1]}."
            )

        # Make predictions for each tree
        predictions = []
        for tree in self.trees:
            tree_predictions = [
                RegressorTree.evaluate_tree(tree, record) for record in X
            ]
            predictions.append(tree_predictions)

        # Average predictions across trees
        return np.mean(predictions, axis=0)

    def get_stats(self, verbose=False):
        """Return the evaluation metrics."""
        stats = {
            "MSE": self.mse,
            "R^2": self.r2,
            "MAPE": self.mape,
            "MAE": self.mae,
            "RMSE": self.rmse,
        }

        if verbose:
            for metric, value in stats.items():
                print(f"{metric}: {value:.4f}")

        return stats
