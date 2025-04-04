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
        self.XX = []  # Initialize the list of input data features and target values

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
        self.XX = []
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
        if X is not None and y is not None:
            self.XX = [
                list(x) + [y] for x, y in zip(X, y, strict=False)
            ]  # Combine X and y

        if not self.X or not self.y:  # If the input data X or target values y are empty
            raise ValueError("Input data X and target values y cannot be empty.")

        residuals = np.array(self.y)  # Initialize the residuals with the target values

        for i in range(self.num_trees):  # Loop over the number of trees in the ensemble
            tree = self.trees[i]  # Get the current tree
            self.trees[i] = tree.learn(
                self.X, residuals
            )  # Fit the tree to the residuals

            predictions = np.array(
                [
                    RegressorTree.evaluate_tree(self.trees[i], record)
                    for record in self.X
                ]
            )  # Predict the target values using the current tree

            residuals = (
                residuals - predictions
            )  # Update the residuals by subtracting the predictions from the target values

            mean_absolute_residual = np.mean(
                np.abs(residuals)
            )  # Calculate the mean absolute residuals
            self.mean_absolute_residuals.append(
                mean_absolute_residual
            )  # Append the mean absolute residuals to the list

            if stats:  # If stats is True, print the mean absolute residuals
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
