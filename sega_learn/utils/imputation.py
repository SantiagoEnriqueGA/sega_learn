# HIGH LEVEL STRUCTURE OF IMPUTATION CLASSES
# --------------------------------------------------------------------------------------------
# BaseImputer: a base class for common interface and utility functions
# fit, fit_transform, transform, ...
# StatisticalImputer: a class for statistical imputation (mean, median, mode)
# DirectionalImputer: a class for directional imputation (forward, backward)
# InterpolationImputer: a class for interpolation imputation (linear, polynomial, etc.)
# KNNImputer: a class for KNN imputation
# CustomImputer: a class for custom imputation
# Can take estimator as input and use it for imputation

import numpy as np
from scipy.stats import mode


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
