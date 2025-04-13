# Importing the required libraries

import numpy as np


class RegressorTreeUtility:
    """Utility class containing helper functions for building the Regressor Tree.

    Handles variance calculation, leaf value calculation, and finding the best split.
    """

    def __init__(self, X, y, min_samples_split, n_features):
        """Initialize the utility class with references to data and parameters.

        Args:
            X (np.ndarray): Reference to the feature data.
            y (np.ndarray): Reference to the target data.
            min_samples_split (int): Minimum number of samples required to split a node.
            n_features (int): Total number of features in X.
        """
        self._X = X
        self._y = y
        self.min_samples_split = min_samples_split
        self._n_features = n_features

    def calculate_variance(self, indices):
        """Calculate variance for the subset defined by indices."""
        if len(indices) == 0:
            return 0.0
        # Directly index the original y array stored in the utility instance
        return np.var(self._y[indices])

    def calculate_leaf_value(self, indices):
        """Calculate the value for a leaf node (mean)."""
        if len(indices) == 0:
            return np.nan  # Or handle appropriately
        # Directly index the original y array stored in the utility instance
        return np.mean(self._y[indices])

    def best_split(self, indices):
        """Finds the best split for the data subset defined by indices."""
        n_samples_node = len(indices)
        if n_samples_node < self.min_samples_split:
            return None  # Not enough samples to split

        # Calculate variance of the current node using the utility's method
        parent_variance = self.calculate_variance(indices)
        if parent_variance == 0:  # Pure node already
            return None

        best_gain = -np.inf
        best_split_info = None

        # Consider a subset of features
        num_features_total = self._n_features
        if num_features_total <= 0:
            return None
        num_features_to_consider = max(1, int(np.sqrt(num_features_total)))
        if num_features_to_consider > num_features_total:
            num_features_to_consider = num_features_total

        selected_feature_indices = np.random.choice(
            num_features_total, size=num_features_to_consider, replace=False
        )

        # Use data corresponding to current indices (referencing X stored in utility)
        X_node = self._X[indices]
        # y_node is implicitly used via self.calculate_variance

        for feature_idx in selected_feature_indices:
            feature_values = X_node[:, feature_idx]

            # --- Optimization Point ---
            # Original used fixed percentiles.
            # A more robust (but potentially slower if not optimized) way is
            # unique sorted values. Let's stick to percentiles for *this* optimization
            # round focusing on index passing.
            # If this is still too slow, the *next* step is the incremental variance
            # update over sorted unique values.
            # Using percentiles remains a fast heuristic. Ensure unique values.
            potential_split_values = np.unique(
                np.percentile(feature_values, [25, 50, 75])
            )

            for split_val in potential_split_values:
                # Partition INDICES, not data
                # mask applies to the *subset* X_node
                mask_left = feature_values <= split_val
                indices_left = indices[mask_left]
                indices_right = indices[~mask_left]

                n_left, n_right = len(indices_left), len(indices_right)

                # Ensure split creates two non-empty children
                # (Could add min_samples_leaf check here too)
                if n_left == 0 or n_right == 0:
                    continue

                # Calculate gain based on children's variance using the utility's method
                var_left = self.calculate_variance(indices_left)
                var_right = self.calculate_variance(indices_right)

                # Weighted variance of children
                child_variance = (
                    n_left * var_left + n_right * var_right
                ) / n_samples_node

                # Information gain (variance reduction)
                info_gain = parent_variance - child_variance

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_split_info = {
                        "feature_idx": feature_idx,
                        "threshold": split_val,
                        "indices_left": indices_left,
                        "indices_right": indices_right,
                        "info_gain": info_gain,
                    }

        # If no split improves variance (info_gain <= 0), best_split_info remains None
        # Or if initial checks failed (e.g., pure node, not enough samples)
        if best_gain <= 0:
            return None

        return best_split_info


class RegressorTree:
    """A class representing a decision tree for regression.

    Args:
        max_depth: (int) - The maximum depth of the decision tree.

    Methods:
        learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
        classify(record): Predicts the target value for a record using the decision tree.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        """Initialize the decision tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split  # Minimum samples required to split
        self.tree = {}
        self._X = None  # Store reference to original X
        self._y = None  # Store reference to original y
        self._n_features = None

    def fit(self, X, y):
        """Fit the decision tree to the training data.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target labels.

        Returns:
            dict: The learned decision tree.
        """
        # Convert only once and store references
        self._X = np.asarray(X)
        self._y = np.asarray(y)
        self._n_features = self._X.shape[1]

        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # Initialize the utility class here, passing data references and params
        self.utility = RegressorTreeUtility(
            self._X, self._y, self.min_samples_split, self._n_features
        )

        initial_indices = np.arange(self._X.shape[0])
        self.tree = self._learn_recursive(initial_indices, depth=0)
        return self.tree

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

        X = np.asarray(X)
        predictions = [self._traverse_tree(x, self.tree) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        """Traverse the tree for a single sample x."""
        # Check if it's a leaf node
        if "value" in node:
            return node["value"]

        # Check if node is valid (basic check)
        if "feature_idx" not in node or "threshold" not in node:
            # This might happen if the tree is malformed or empty
            # Consider returning a default value or raising an error
            # Let's return NaN, assuming values are floats
            # print("Warning: Malformed node encountered during prediction.")
            return np.nan  # Or handle appropriately

        # Decide which branch to follow
        if x[node["feature_idx"]] <= node["threshold"]:
            # Potential issue: If left node is empty or malformed
            if isinstance(node.get("left"), dict):
                return self._traverse_tree(x, node["left"])
            else:
                # Handle cases where subtree might not be a dict (e.g., None if pruning happened badly)
                # print("Warning: Left node missing/malformed.")
                return np.nan  # Or a default value based on parent? Hard to say without more context.
        else:
            if isinstance(node.get("right"), dict):
                return self._traverse_tree(x, node["right"])
            else:
                # print("Warning: Right node missing/malformed.")
                return np.nan

    def _learn_recursive(self, indices, depth):
        """Recursive helper function for learning."""
        # Check termination conditions
        # 1. Max depth reached
        # 2. Node is pure (variance is 0) - checked implicitly by best_split gain > 0
        # 3. Not enough samples to split - checked by best_split returning None
        # 4. No split improves variance - checked by best_split gain > 0

        # Calculate leaf value first (will be used if termination condition met)
        leaf_value = self.utility.calculate_leaf_value(indices)

        if depth >= self.max_depth:
            # print(f"Depth limit reached at depth {depth}. Leaf value: {leaf_value}")
            return {"value": leaf_value}

        if len(indices) < self.min_samples_split:
            # print(f"Min samples limit reached ({len(indices)} < {self.min_samples_split}). Leaf value: {leaf_value}")
            return {"value": leaf_value}

        # Find the best split for the current indices
        split_info = self.utility.best_split(indices)

        # If no good split found (includes pure nodes, min_samples, no gain)
        if split_info is None:
            # print(f"No good split found at depth {depth}. Leaf value: {leaf_value}")
            return {"value": leaf_value}

        # print(f"Split at depth {depth}: Feature {split_info['feature_idx']} <= {split_info['threshold']:.2f}, Gain: {split_info['info_gain']:.4f}")

        # Recursively build left and right subtrees
        left_subtree = self._learn_recursive(split_info["indices_left"], depth + 1)
        right_subtree = self._learn_recursive(split_info["indices_right"], depth + 1)

        # Return internal node structure
        return {
            "feature_idx": split_info["feature_idx"],
            "threshold": split_info["threshold"],
            "left": left_subtree,
            "right": right_subtree,
            # Optional: store gain, samples etc. for inspection
            # "n_samples": len(indices),
            # "info_gain": split_info['info_gain']
        }
