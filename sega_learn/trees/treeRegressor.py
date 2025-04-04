# Importing the required libraries

import numpy as np


class RegressorTreeUtility:
    """Utility class for computing variance, partitioning classes, and calculating information gain."""

    def calculate_variance(self, y):
        """Calculate the variance of a dataset.

        Variance is used as the measure of impurity in the case of regression.
        """
        if len(y) == 0:
            return 0
        return np.var(y)

    def partition_classes(self, X, y, split_attribute, split_val):
        """Partitions the dataset into two subsets based on a given split attribute and value.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target labels.
            split_attribute: (int) - The index of the attribute to split on.
            split_val: (float) - The value to split the attribute on.

        Returns:
            X_left: (array-like) - The subset of input features where the split attribute is less than or equal to the split value.
            X_right: (array-like) - The subset of input features where the split attribute is greater than the split value.
            y_left: (array-like) - The subset of target labels corresponding to X_left.
            y_right: (array-like) - The subset of target labels corresponding to X_right.
        """
        # Check type of X and y
        if not isinstance(X, list | np.ndarray) or not isinstance(y, list | np.ndarray):
            raise TypeError("X and y must be lists or NumPy arrays.")

        # Convert X and y to NumPy arrays if they are not already
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        if X.ndim == 1:  # If X is a 1D array
            X = X.reshape(-1, 1)  # Convert to a 2D array with one column

        # Use NumPy boolean indexing for partitioning
        # X_left  contains rows where the split attribute is less than or equal to the split value
        # X_right contains rows where the split attribute is greater than the split value
        # y_left  contains target labels corresponding to X_left
        # y_right contains target labels corresponding to X_right
        mask = X[:, split_attribute] <= split_val
        X_left = X[mask]
        X_right = X[~mask]
        y_left = y[mask]
        y_right = y[~mask]

        return X_left, X_right, y_left, y_right

    def information_gain(self, previous_y, current_y):
        """Calculate the information gain from a split by subtracting the variance of child nodes from the variance of the parent node."""
        if len(previous_y) == 0:  # If the parent node is empty
            return 0  # Return 0 information gain

        # Compute parent variance
        parent_variance = self.calculate_variance(previous_y)

        # Compute weighted child variance
        total_len = len(previous_y)
        child_variance = (
            sum(len(y) * self.calculate_variance(y) for y in current_y) / total_len
        )

        return parent_variance - child_variance

    def best_split(self, X, y):
        """Finds the best attribute and value to split the data based on information gain.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target variable.

        Returns:
            dict: A dictionary containing the best split attribute, split value, left and right subsets of X and y,
                and the information gain achieved by the split.
        """
        # Check type of X and y
        if not isinstance(X, list | np.ndarray) or not isinstance(y, list | np.ndarray):
            raise TypeError("X and y must be lists or NumPy arrays.")

        # Convert X and y to NumPy arrays if they are not already
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        if X.size == 0:  # If X is empty
            return {
                "split_attribute": None,
                "split_val": None,
                "X_left": np.empty((0, 1)),
                "X_right": np.empty((0, 1)),
                "y_left": np.empty((0,)),
                "y_right": np.empty((0,)),
                "info_gain": 0,
            }

        if X.shape[0] == 1:  # If X has a single value
            return {
                "split_attribute": None,
                "split_val": None,
                "X_left": np.empty((0, X.shape[1])),
                "X_right": np.empty((0, X.shape[1])),
                "y_left": np.empty((0,)),
                "y_right": np.empty((0,)),
                "info_gain": 0,
            }

        # Randomly select a subset of attributes for splitting
        num_features = int(np.sqrt(X.shape[1]))  # Square root of total attributes
        selected_attributes = np.random.choice(
            X.shape[1], size=num_features, replace=False
        )  # Randomly select attributes

        # Initialize the best information gain to negative infinity, others to None
        best_info_gain = float("-inf")
        best_split = None

        # Use numpy's percentile function to reduce split points
        for split_attribute in selected_attributes:
            # Instead of trying all values, sample a subset of potential split points
            feature_values = X[:, split_attribute]

            # Use percentiles to get a representative sample of split points
            percentiles = np.percentile(feature_values, [25, 50, 75])

            for split_val in percentiles:
                X_left, X_right, y_left, y_right = self.partition_classes(
                    X, y, split_attribute, split_val
                )

                # Skip if split doesn't divide the data
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                info_gain = self.information_gain(y, [y_left, y_right])

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        "split_attribute": split_attribute,
                        "split_val": split_val,
                        "X_left": X_left,
                        "X_right": X_right,
                        "y_left": y_left,
                        "y_right": y_right,
                        "info_gain": info_gain,
                    }

        if best_split is None:
            # If no good split found, return a default split
            return {
                "split_attribute": None,
                "split_val": None,
                "X_left": np.empty((0, X.shape[1])),
                "X_right": np.empty((0, X.shape[1])),
                "y_left": np.empty(0),
                "y_right": np.empty(0),
                "info_gain": 0,
            }

        return best_split


class RegressorTree:
    """A class representing a decision tree for regression.

    Args:
        max_depth: (int) - The maximum depth of the decision tree.

    Methods:
        learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
        classify(record): Predicts the target value for a record using the decision tree.
    """

    def __init__(self, max_depth=5):
        """Initialize the decision tree."""
        self.tree = {}  # Initialize an empty dictionary to represent the decision tree
        self.max_depth = max_depth  # Set the maximum depth of the tree

    def fit(self, X, y):
        """Fit the decision tree to the training data.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target labels.

        Returns:
            dict: The learned decision tree.
        """
        self.tree = self.learn(X, y)
        return self.tree

    def learn(self, X, y, par_node=None, depth=0):
        """Builds the decision tree based on the given training data.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target labels.
            par_node: (dict) - The parent node of the current subtree (default: {}).
            depth: (int) - The current depth of the subtree (default: 0).

        Returns:
            dict: The learned decision tree.
        """
        # Check type of X and y
        if not isinstance(X, list | np.ndarray) or not isinstance(y, list | np.ndarray):
            raise TypeError("X and y must be lists or NumPy arrays.")

        y = (
            y.tolist() if isinstance(y, np.ndarray) else y
        )  # Convert y to a list if it is a NumPy array

        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y, dtype=float)

        # If X and y are empty, return an empty dictionary
        if X.size == 0 or y.size == 0:
            return {}

        # If X is empty, return mean of y
        if X.size == 0:
            return {"value": np.mean(y)}

        # Base cases
        if len(set(y)) == 1:  # If the node is pure (all labels are the same)
            return {"value": y[0]}  # Return the label as the value of the leaf node

        if depth >= self.max_depth:  # If the maximum depth is reached
            return {
                "value": np.mean(y)
            }  # Return the mean of the target values as the value of the leaf node

        # Find best split
        utility = RegressorTreeUtility()
        best_split = utility.best_split(X, y)

        if best_split["split_attribute"] is None or best_split["info_gain"] <= 0:
            return {"value": np.mean(y)}

        # Build subtrees
        left_tree = self.learn(
            best_split["X_left"], best_split["y_left"], depth=depth + 1
        )
        right_tree = self.learn(
            best_split["X_right"], best_split["y_right"], depth=depth + 1
        )

        return {
            "split_attribute": best_split["split_attribute"],
            "split_val": best_split["split_val"],
            "left": left_tree,
            "right": right_tree,
        }

    @staticmethod
    def evaluate_tree(tree, record):
        """Make a prediction using the decision tree."""
        # If tree is empty, return None
        if tree is None or tree == {}:
            return None

        if "value" in tree:
            return tree["value"]

        if record[tree["split_attribute"]] <= tree["split_val"]:
            return RegressorTree.evaluate_tree(tree["left"], record)
        else:
            return RegressorTree.evaluate_tree(tree["right"], record)

    def predict(self, X):
        """Predict the target value for a record using the decision tree.

        Args:
            X: (array-like) - The input features.

        Returns:
            float: The predicted target value.
        """
        if not isinstance(X, list | np.ndarray):
            raise TypeError("X must be a list or NumPy array.")

        if isinstance(X, np.ndarray):
            X = X.tolist()

        predictions = [self.evaluate_tree(self.tree, record) for record in X]
        return np.array(predictions) if len(predictions) > 1 else predictions[0]
