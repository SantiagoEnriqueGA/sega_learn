# HIGH LEVEL STRUCTURE OF IMPUTATION CLASSES
# --------------------------------------------------------------------------------------------
# BaseImputer: a base class for common interface and utility functions
# fit, fit_transform, transform, ...
# StatisticalImputer: a class for statistical imputation (mean, median, mode)
# DirectionalImputer: a class for directional imputation (forward, backward)
# InterpolationImputer: a class for interpolation imputation (linear, polynomial, etc.)
# KNNImputer: a class for KNN imputation
# CustomImputer: a class for custom imputation, take estimator as input and use it for imputation

import numpy as np
from scipy.stats import mode

from sega_learn.nearest_neighbors.knn_classifier import KNeighborsClassifier
from sega_learn.nearest_neighbors.knn_regressor import KNeighborsRegressor


class BaseImputer:
    """Base class for imputers providing a common interface."""

    def fit(self, X, y=None):
        """Fit the imputer on the data."""
        raise NotImplementedError("The fit method must be implemented by subclasses.")

    def transform(self, X):
        """Transform the data using the fitted imputer."""
        raise NotImplementedError(
            "The transform method must be implemented by subclasses."
        )

    def fit_transform(self, X, y=None):
        """Fit the imputer and transform the data."""
        self.fit(X, y)
        return self.transform(X)


class StatisticalImputer(BaseImputer):
    """Statistical imputer for handling missing values using mean, median, or mode."""

    def __init__(self, strategy="mean", nan_policy="omit"):
        """Initialize the StatisticalImputer with a specified strategy.

        The strategy can be "mean", "median", or "mode".
        The nan_policy can be "omit", "propagate", or "raise".
        Nan policy determines how to handle NaN values in the data.
            - "omit": Ignore NaN values when calculating the statistic.
            - "propagate": Keep NaN values in the data. Treat NaN values as 0.
            - "raise": Raise an error if NaN values are found in the data.

        Args:
            strategy (str): The imputation strategy ("mean", "median", or "mode").
            nan_policy (str): Policy for handling NaN values ("omit", "propagate", or "raise").
        """
        if strategy not in ["mean", "median", "mode"]:
            raise ValueError("Strategy must be one of 'mean', 'median', or 'mode'.")
        if nan_policy not in ["omit", "propagate", "raise"]:
            raise ValueError(
                "nan_policy must be one of 'omit', 'propagate', or 'raise'."
            )

        self.strategy = strategy
        self.nan_policy = nan_policy
        self.statistic_ = None

    def fit(self, X, y=None):
        """Compute the statistic to be used for imputation.

        Args:
            X (array-like): The input data with missing values.
            y (ignored): Not used, present for compatibility.
        """
        # Mean Strategy
        if self.strategy == "mean":
            if self.nan_policy == "omit":
                self.statistic_ = np.nanmean(X, axis=0)
            elif self.nan_policy == "propagate":
                self.statistic_ = np.nanmean(np.nan_to_num(X), axis=0)
            elif self.nan_policy == "raise":
                if np.isnan(X).any():
                    raise ValueError(
                        "Input contains NaN values. Use nan_policy='omit' to ignore them."
                    )
                self.statistic_ = np.mean(X, axis=0)

        # Median Strategy
        elif self.strategy == "median":
            if self.nan_policy == "omit":
                self.statistic_ = np.nanmedian(X, axis=0)
            elif self.nan_policy == "propagate":
                self.statistic_ = np.nanmedian(np.nan_to_num(X), axis=0)
            elif self.nan_policy == "raise":
                if np.isnan(X).any():
                    raise ValueError(
                        "Input contains NaN values. Use nan_policy='omit' to ignore them."
                    )
                self.statistic_ = np.median(X, axis=0)

        # Mode Strategy
        # TODO: refactor to remove scipy.stats mode?
        elif self.strategy == "mode":
            self.statistic_ = mode(X, nan_policy=self.nan_policy).mode.squeeze()

        return self

    def transform(self, X):
        """Impute missing values in X.

        Using the statistic computed during fit.
        This method replaces NaN values in X with the corresponding statistic.

        Args:
            X (array-like): The input data with missing values.
        """
        if self.statistic_ is None:
            raise ValueError(
                "The imputer has not been fitted yet. Call fit() before transform()."
            )

        X = np.array(X, copy=True)
        mask = np.isnan(X)
        X[mask] = np.take(self.statistic_, np.where(mask)[1])
        return X


class DirectionalImputer(BaseImputer):
    """Directional imputer for handling missing values using forward or backward fill."""

    def __init__(self, direction="forward"):
        """Initialize the DirectionalImputer with a specified direction.

        The direction can be "forward" or "backward".

        Args:
            direction (str): The imputation direction ("forward" or "backward").
        """
        if direction not in ["forward", "backward"]:
            raise ValueError("Direction must be either 'forward' or 'backward'.")
        self.direction = direction

    def fit(self, X=None, y=None):
        """Fit the imputer on the data. No operation needed for directional imputer."""
        return self

    def transform(self, X):
        """Impute missing values in X using the specified direction.

        Args:
            X (array-like): The input data with missing values.
        """
        X = np.array(X, copy=True)
        if self.direction == "forward":
            for i in range(1, X.shape[0]):
                # For each row, replace NaN with the previous value
                # If the previous value is also NaN, it will remain NaN
                X[i] = np.where(np.isnan(X[i]), X[i - 1], X[i])
        elif self.direction == "backward":
            # For backward fill, iterate from the end to the beginning (reverse order)
            for i in range(X.shape[0] - 2, -1, -1):
                # For each row, replace NaN with the next value
                # If the next value is also NaN, it will remain NaN
                X[i] = np.where(np.isnan(X[i]), X[i + 1], X[i])
        return X


class InterpolationImputer(BaseImputer):
    """Interpolation imputer for handling missing values using linear interpolation."""

    def __init__(self, method="linear", degree=1):
        """Initialize the InterpolationImputer with a specified method.

        The method can be "linear" or "polynomial"

        Args:
            method (str): The interpolation method ("linear", "polynomial").
            degree (int): The degree of the polynomial for polynomial interpolation.
        """
        self.method = method
        self.degree = degree

    def fit(self, X=None, y=None):
        """Fit the imputer on the data. No operation needed for interpolation imputer."""
        return self

    def transform(self, X):
        """Impute missing values in X using interpolation.

        Args:
            X (array-like): The input data with missing values.
        """
        X = np.array(X, copy=True)

        # Linear Interpolation
        if self.method == "linear":
            for i in range(X.shape[1]):
                X[:, i] = np.interp(
                    np.arange(X.shape[0]),
                    np.flatnonzero(~np.isnan(X[:, i])),
                    X[~np.isnan(X[:, i]), i],
                )

        # Polynomial Interpolation
        elif self.method == "polynomial":
            for i in range(X.shape[1]):
                X[:, i] = np.polyval(
                    np.polyfit(
                        np.flatnonzero(~np.isnan(X[:, i])),
                        X[~np.isnan(X[:, i]), i],
                        self.degree,
                    ),
                    np.arange(X.shape[0]),
                )
        return X


class KNNImputer(BaseImputer):
    """K-Nearest Neighbors imputer for handling missing values using KNN."""

    def __init__(self, n_neighbors=5, distance_metric="euclidean"):
        """Initialize the KNNImputer with a specified number of neighbors.

        Args:
            n_neighbors (int): The number of neighbors to use for imputation.
            distance_metric (str): Distance metric for calculating distances ('euclidean', 'manhattan', 'minkowski').
        """
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        if distance_metric not in ["euclidean", "manhattan", "minkowski"]:
            raise ValueError(
                "distance_metric must be one of 'euclidean', 'manhattan', or 'minkowski'."
            )

    def fit(self, X=None, y=None):
        """Fit the imputer on the data. No operation needed for KNN imputer."""
        return self

    def transform(self, X):
        """Impute missing values in X using KNN.

        Args:
            X (array-like): The input data with missing values.
        """
        # TODO: Update to correctly handle categorical features
        X = np.array(X, copy=True)

        # Identify numerical features
        num_features = np.array(
            [np.issubdtype(X[:, i].dtype, np.number) for i in range(X.shape[1])]
        )

        # Iterate over each feature
        for i in range(X.shape[1]):
            missing_mask = np.isnan(X[:, i])

            if not np.any(missing_mask):
                # Skip if no missing values in the feature
                continue

            if num_features[i]:
                # Use KNeighborsRegressor for numerical features
                knn = KNeighborsRegressor(
                    n_neighbors=self.n_neighbors, distance_metric=self.distance_metric
                )
            else:
                # Use KNeighborsClassifier for categorical features
                knn = KNeighborsClassifier(
                    n_neighbors=self.n_neighbors, distance_metric=self.distance_metric
                )

            # Fit the KNN model on non-missing data
            knn.fit(X[~missing_mask, :], X[~missing_mask, i])

            # Predict missing values
            X[missing_mask, i] = knn.predict(X[missing_mask, :])

        return X
