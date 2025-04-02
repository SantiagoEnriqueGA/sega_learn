# Importing the required libraries
import numpy as np


class PolynomialTransform:
    """
    This class implements Polynomial Feature Transformation.
    Polynomial feature transformation is a technique used to create new features from the existing features by raising them to a power or by creating interaction terms.

    Parameters:
    - degree: int, default=2
        The degree of the polynomial features.

    Attributes:
    - n_samples: int
        The number of samples.
    - n_features: int
        The number of features.
    - n_output_features: int
        The number of output features.
    - combinations: list of tuples of shape (n_features,)
        The combinations of features(X) of degree n.
    - bias: bool, default=True
        Whether to include a bias term in the output features.
    """

    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X):
        """
        Fit the model to the data.
        Uses itertools.combinations_with_replacement to generate all possible combinations of features(X) of degree n.
        """
        from itertools import combinations_with_replacement

        self.n_samples, self.n_features = X.shape

        # Generate all possible combinations of features(X) of degree n
        self.combinations = []
        for d in range(1, self.degree + 1):
            self.combinations.extend(
                combinations_with_replacement(range(self.n_features), d)
            )

        self.n_output_features = len(self.combinations) + 1  # +1 for the bias term

    def transform(self, X):
        """
        Transform the data into polynomial features by computing the product of the features for each combination of features.
        """
        n_samples = X.shape[0]

        # Initialize the polynomial features with the bias term
        X_poly = np.empty((n_samples, self.n_output_features))
        X_poly[:, 0] = 1  # Bias term

        # For each combination of features, compute the product of the features
        for i, comb in enumerate(self.combinations, start=1):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)

        # Return the polynomial features
        return X_poly

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        self.fit(X)
        return self.transform(X)
