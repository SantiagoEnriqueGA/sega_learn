import numpy as np

from .treeRegressor import RegressorTree, RegressorTreeUtility


class GradientBoostedRegressor:
    """A class to represent a Gradient Boosted Decision Tree Regressor.

    Attributes:
        random_seed (int): The random seed for the random number generator.
        num_trees (int): The number of decision trees in the ensemble.
        max_depth (int): The maximum depth of each decision tree.
        display (bool): A flag to display the decision tree.
        X (list): A list of input data features.
        y (list): A list of target values.
        XX (list): A list of input data features and target values.
        numerical_cols (set): A set of indices of numeric attributes (columns).

    Methods:
        __init__(file_loc, num_trees=5, random_seed=0, max_depth=10): Initializes the GBDT object.
        reset(): Resets the GBDT object.
        fit(): Fits the GBDT model to the training data.
        predict(): Predicts the target values for the input data.
        get_stats(y_predicted): Calculates various evaluation metrics for the predicted target values.
    """

    def __init__(
        self,
        X=None,
        y=None,
        num_trees: int = 10,
        max_depth: int = 10,
        random_seed: int = 0,
    ):
        """Initializes the Gradient Boosted Decision Tree Regressor.

        Args:
            X: (np.ndarray), optional - Input feature data (default is None).
            y: (np.ndarray), optional - Target data (default is None).
            num_trees: (int), optional - Number of trees in the ensemble (default is 10).
            max_depth: (int), optional - Maximum depth of each tree (default is 10).
            random_seed: (int), optional - Random seed for reproducibility (default is 0).

        Returns:
            None
        """
        self.random_seed = random_seed  # Set the random seed for reproducibility
        self.num_trees = num_trees  # Set the number of trees in the ensemble
        self.max_depth = max_depth  # Set the maximum depth of each tree

        self.X = []  # Initialize the list of input data features
        self.y = []  # Initialize the list of target values

        self.numerical_cols = (
            0  # Initialize the set of indices of numeric attributes (columns)
        )
        self.mean_absolute_residuals = []  # Initialize the list of Mean Absolute Residuals for each tree

        self.utility = RegressorTreeUtility()  # Initialize the Utility object
        self.trees = [
            RegressorTree(self.max_depth) for i in range(self.num_trees)
        ]  # Initialize the list of decision trees, each with the specified maximum depth
        self.numerical_cols = (
            set()
        )  # Initialize the set of indices of numeric attributes (columns)

        if X is not None:
            self.X = X.tolist()  # Convert ndarray to list
        if y is not None:
            self.y = y.tolist()  # Convert ndarray to list
        if X is not None and y is not None:
            self.XX = [
                list(x) + [y] for x, y in zip(X, y, strict=False)
            ]  # Combine X and y

    def reset(self):
        """Resets the GBDT object to its initial state."""
        # Reset the GBDT object
        self.random_seed = 0
        self.num_trees = 10
        self.max_depth = 10
        self.X = []
        self.y = []
        self.numerical_cols = 0
        self.mean_absolute_residuals = []

    def fit(self, X=None, y=None, stats=False):
        """Fits the gradient boosted decision tree regressor to the training data.

        This method trains the ensemble of decision trees by iteratively fitting each tree to the residuals
        of the previous iteration. The residuals are updated after each iteration by subtracting the predictions
        made by the current tree from the :target values.

        Args:
            X: (numpy.ndarray) - An array of input :data features. Default is None.
            y: (numpy.ndarray) - An array of target values. Default is None.
            stats: (bool) - A flag to decide whether to return stats or not. Default is False.

        Returns:
            None
        """
        if X is not None:
            self.X = X.tolist()
        if y is not None:
            self.y = y.tolist()
        if not self.X or not self.y:  # If the input data X or target values y are empty
            raise ValueError("Input data X and target values y cannot be empty.")

        # Ensure residuals are float
        residuals = np.array(self.y).astype(float)

        for i, tree in enumerate(self.trees):  # Loop over the number of trees
            # Fit the tree to the residuals
            self.trees[i] = tree.learn(self.X, residuals)

            # Predict residuals using the current tree
            predictions = np.array(
                [
                    RegressorTree.evaluate_tree(self.trees[i], record)
                    for record in self.X
                ],
                dtype=float,
            )  # Ensure predictions are float

            # Update residuals
            residuals -= predictions

            # Calculate mean absolute residuals
            mean_absolute_residual = np.mean(np.abs(residuals))
            self.mean_absolute_residuals.append(mean_absolute_residual)

            if stats:  # Print stats if required
                print(
                    f"Tree {i + 1} trained. Mean Absolute Residuals: {mean_absolute_residual}"
                )

    def predict(self, X=None):
        """Predicts the target values for the input data using the gradient boosted decision tree regressor.

        Args:
            X: (numpy.ndarray) - An array of input data features. Default is None.

        Returns:
            predictions: (numpy.ndarray) - An array of predicted target values for the input data.
        """
        if X is not None:
            self.X = X.tolist()

        predictions = np.zeros(
            len(self.X)
        )  # Initialize an array of zeros for the predictions

        for i in range(self.num_trees):  # Loop over the number of trees in the ensemble
            oneTree_predictions = np.zeros(
                len(self.X)
            )  # Initialize an array of zeros for the predictions of the current tree

            for j in range(len(self.X)):  # Loop over the indices of the input data
                oneTree_predictions[j] += RegressorTree.evaluate_tree(
                    self.trees[i], self.X[j]
                )  # Predict the target value for the current input data

            predictions += oneTree_predictions  # Add the predictions of the current tree to the overall predictions

        return predictions

    def get_stats(self, y_predicted):
        """Calculates various evaluation metrics for the predicted target values.

        Args:
            y_predicted (numpy.ndarray): An array of predicted target values.

        Returns:
            dict: A dictionary containing the evaluation metrics.
                - MSE (float): Mean Squared Error
                - R^2 (float): R-squared Score
                - MAPE (float): Mean Absolute Percentage Error
                - MAE (float): Mean Absolute Error
                - RMSE (float): Root Mean Squared Error
        """
        mse = np.mean(
            (np.array(y_predicted) - np.array(self.y)) ** 2
        )  # Mean Squared Error (MSE): (y - y')^2

        ssr = np.sum(
            (np.array(y_predicted) - np.array(self.y)) ** 2
        )  # Sum of Squared Residuals (SSR): (y - y')^2
        sst = np.sum(
            (np.array(self.y) - np.mean(self.y)) ** 2
        )  # Total Sum of Squares (SST): (y - mean(y))^2
        r2 = 1 - (ssr / sst)  # R-squared Score (R^2): 1 - (SSR / SST)

        epsilon = 1e-10  # Small value to prevent division by zero
        mape = (
            np.mean(
                np.abs(
                    (np.array(self.y) - np.array(y_predicted))
                    / (np.array(self.y) + epsilon)
                )
            )
            * 100
        )  # Mean Absolute Percentage Error (MAPE): (|y - y'| / y) * 100

        mae = np.mean(
            np.abs(np.array(self.y) - np.array(y_predicted))
        )  # Mean Absolute Error (MAE): |y - y'|

        rmse = np.sqrt(
            np.mean((np.array(y_predicted) - np.array(self.y)) ** 2)
        )  # Root Mean Squared Error (RMSE): sqrt((y - y')^2)

        # Return the evaluation metrics
        return {
            "MSE": mse,
            "R^2": r2,
            "MAPE": mape,
            "MAE": mae,
            "RMSE": rmse,
            # "Mean_Absolute_Residuals": self.mean_absolute_residuals
        }


# PROFILE REPORT
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#     87                                               @profile
#     88                                               def fit(self, X=None, y=None, stats=False):
#     89                                                   """Fits the gradient boosted decision tree regressor to the training data.
#     90
#     91                                                   This method trains the ensemble of decision trees by iteratively fitting each tree to the residuals
#     92                                                   of the previous iteration. The residuals are updated after each iteration by subtracting the predictions
#     93                                                   made by the current tree from the :target values.
#     94
#     95                                                   Args:
#     96                                                       X: (numpy.ndarray) - An array of input :data features. Default is None.
#     97                                                       y: (numpy.ndarray) - An array of target values. Default is None.
#     98                                                       stats: (bool) - A flag to decide whether to return stats or not. Default is False.
#     99
#    100                                                   Returns:
#    101                                                       None
#    102                                                   """
#    103         1          0.6      0.6      0.0          if X is not None:
#    104                                                       self.X = X.tolist()
#    105         1          0.4      0.4      0.0          if y is not None:
#    106                                                       self.y = y.tolist()
#    107         1          1.3      1.3      0.0          if not self.X or not self.y:  # If the input data X or target values y are empty
#    108                                                       raise ValueError("Input data X and target values y cannot be empty.")
#    109
#    110                                                   # Ensure residuals are float
#    111         1         42.3     42.3      0.0          residuals = np.array(self.y).astype(float)
#    112
#    113        11         35.7      3.2      0.0          for i, tree in enumerate(self.trees):  # Loop over the number of trees
#    114                                                       # Fit the tree to the residuals
#    115        10     341363.7  34136.4     80.1              self.trees[i] = tree.learn(self.X, residuals)
#    116
#    117                                                       # Predict residuals using the current tree
#    118        20        269.6     13.5      0.1              predictions = np.array(
#    119        20      82328.3   4116.4     19.3                  [
#    120                                                               RegressorTree.evaluate_tree(self.trees[i], record)
#    121        10          6.8      0.7      0.0                      for record in self.X
#    122                                                           ],
#    123        10          5.5      0.6      0.0                  dtype=float,
#    124                                                       )  # Ensure predictions are float
#    125
#    126                                                       # Update residuals
#    127        10         43.9      4.4      0.0              residuals -= predictions
#    128
#    129                                                       # Calculate mean absolute residuals
#    130        10        311.9     31.2      0.1              mean_absolute_residual = np.mean(np.abs(residuals))
#    131        10         19.1      1.9      0.0              self.mean_absolute_residuals.append(mean_absolute_residual)
#    132
#    133        10          3.0      0.3      0.0              if stats:  # Print stats if required
#    134        20       1406.0     70.3      0.3                  print(
#    135        10         78.6      7.9      0.0                      f"Tree {i + 1} trained. Mean Absolute Residuals: {mean_absolute_residual}"
#    136                                                           )
