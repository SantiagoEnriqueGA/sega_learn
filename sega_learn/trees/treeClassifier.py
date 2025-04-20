# Importing the required libraries

import numpy as np


class ClassifierTreeUtility:
    """Utility class for computing entropy, partitioning classes, and calculating information gain."""

    def entropy(self, class_y, sample_weight=None):
        """Computes the entropy for a given class.

        Args:
            class_y: (array-like) - The class labels.
            sample_weight: (array-like) - The sample weights (default: None).

        Returns:
            float: The entropy value.
        """
        n_samples = len(class_y)
        if n_samples == 0:
            return 0.0

        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            # Normalize weights if they don't sum to 1 (or close)
            total_weight = np.sum(sample_weight)
            if total_weight <= 0:
                return 0.0  # Avoid division by zero if all weights are zero
            if not np.isclose(total_weight, 1.0):
                sample_weight = sample_weight / total_weight

        # Use weighted counts
        unique_classes = np.unique(class_y)
        entropy_val = 0.0
        for cls in unique_classes:
            # Sum weights for samples belonging to the class
            weight_cls = np.sum(sample_weight[class_y == cls])
            if weight_cls > 0:
                # Use weight_cls directly as probability estimate within this subset
                entropy_val -= weight_cls * np.log2(weight_cls)

        return entropy_val

    def partition_classes(self, X, y, split_attribute, split_val, sample_weight=None):
        """Partitions the dataset into two subsets based on a given split attribute and value.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target labels.
            split_attribute: (int) - The index of the attribute to split on.
            split_val: (float) - The value to split the attribute on.
            sample_weight: (array-like) - The sample weights (default: None).

        Returns:
            X_left:  (array-like) - The subset of input features where the split attribute is less than or equal to the split value.
            X_right: (array-like) - The subset of input features where the split attribute is greater than the split value.
            y_left:  (array-like) - The subset of target labels corresponding to X_left.
            y_right: (array-like) - The subset of target labels corresponding to X_right.
        """
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:  # If X has only one feature
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

        # Partition weights
        if sample_weight is not None:
            sample_weight_left = sample_weight[mask]
            sample_weight_right = sample_weight[~mask]
            return (
                X_left,
                X_right,
                y_left,
                y_right,
                sample_weight_left,
                sample_weight_right,
            )

        else:
            return X_left, X_right, y_left, y_right, None, None

    def information_gain(
        self, previous_y, current_y, sample_weight_prev=None, sample_weight_current=None
    ):
        """Calculates the information gain between the previous and current values of y.

        Args:
            previous_y: (array-like) - The previous values of y.
            current_y: (array-like) - The current values of y.
            sample_weight_prev: (array-like) - The sample weights for the previous y values (default: None).
            sample_weight_current: (array-like) - The sample weights for the current y values (default: None).

        Returns:
            float: The information gain between the previous and current values of y.
        """
        n_samples_prev = len(previous_y)
        if n_samples_prev == 0:
            return 0.0

        if sample_weight_prev is None:
            sample_weight_prev = np.ones(n_samples_prev) / n_samples_prev
        else:
            total_weight_prev = np.sum(sample_weight_prev)
            if total_weight_prev <= 0:
                return 0.0
            if not np.isclose(total_weight_prev, 1.0):
                sample_weight_prev = sample_weight_prev / total_weight_prev

        entropy_prev = self.entropy(previous_y, sample_weight_prev)

        # Ensure sample_weight_current is a list of weights corresponding to current_y subsets
        if sample_weight_current is None:
            # Create dummy weights if none provided (should not happen if best_split passes them)
            sample_weight_current = [
                np.ones(len(subset)) / len(subset) if len(subset) > 0 else np.array([])
                for subset in current_y
            ]

        weighted_entropy_current = 0.0
        total_weight_prev_sum = np.sum(
            sample_weight_prev
        )  # Use sum for weighting factor

        for i, subset_y in enumerate(current_y):
            if len(subset_y) > 0:
                subset_weights = sample_weight_current[i]
                subset_total_weight = np.sum(subset_weights)
                if total_weight_prev_sum > 0:
                    weighting_factor = subset_total_weight / total_weight_prev_sum
                    weighted_entropy_current += weighting_factor * self.entropy(
                        subset_y, subset_weights
                    )

        info_gain = entropy_prev - weighted_entropy_current
        return info_gain

    def best_split(self, X, y, sample_weight=None):
        """Finds the best attribute and value to split the data based on information gain.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target variable.
            sample_weight: (array-like) - The sample weights (default: None).

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

        n_samples = X.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)  # Ensure it's an array
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight must have the same length as X and y.")

        # Normalize weights for this node's calculation
        current_total_weight = np.sum(sample_weight)
        if current_total_weight <= 0:
            return None  # Cannot split if total weight is zero
        current_sample_weight = sample_weight / current_total_weight

        # Base cases where splitting is impossible or unnecessary
        if n_samples < 2 or current_total_weight <= 0:
            # Calculate weighted majority even if no split
            unique_classes, class_indices = np.unique(y, return_inverse=True)
            # Handle potential zero weights carefully in bincount
            if current_total_weight > 0:
                weighted_counts = np.bincount(class_indices, weights=sample_weight)
                leaf_label = unique_classes[np.argmax(weighted_counts)]
            elif len(y) > 0:  # Fallback if weights are zero but samples exist
                counts = np.bincount(y)
                leaf_label = np.argmax(counts)
            else:
                # Cannot determine label for empty node, should ideally not happen if called correctly
                # Returning None or a default might be needed depending on broader logic
                # For now, let's signal this is effectively a leaf that shouldn't have been called
                return None  # Indicate no valid split possible

            return {"label": leaf_label}  # Return leaf info

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
                X_left, X_right, y_left, y_right, sw_left, sw_right = (
                    self.partition_classes(
                        X, y, split_attribute, split_val, current_sample_weight
                    )
                )

                # Skip if split doesn't divide the data
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                info_gain = self.information_gain(
                    y, [y_left, y_right], current_sample_weight, [sw_left, sw_right]
                )

                if info_gain > best_info_gain:
                    # Store original data and weights, not normalized ones
                    (
                        X_left_orig,
                        X_right_orig,
                        y_left_orig,
                        y_right_orig,
                        sw_left_orig,
                        sw_right_orig,
                    ) = self.partition_classes(
                        X, y, split_attribute, split_val, sample_weight
                    )
                    best_split = {
                        "split_attribute": split_attribute,
                        "split_val": split_val,
                        "X_left": X_left_orig,
                        "X_right": X_right_orig,
                        "y_left": y_left_orig,
                        "y_right": y_right_orig,
                        "sample_weight_left": sw_left_orig,
                        "sample_weight_right": sw_right_orig,  # Store original weights
                        "info_gain": info_gain,
                    }

        # If no split provided positive gain, return leaf node
        if (
            best_split is None or best_info_gain <= 1e-9
        ):  # Use small tolerance for float comparison
            unique_classes, class_indices = np.unique(y, return_inverse=True)
            if current_total_weight > 0:
                weighted_counts = np.bincount(class_indices, weights=sample_weight)
                leaf_label = unique_classes[np.argmax(weighted_counts)]
            elif len(y) > 0:
                counts = np.bincount(y)
                leaf_label = np.argmax(counts)
            else:
                # This case should be handled by the initial checks
                return None  # Or appropriate default/error

            return {"label": leaf_label}

        return best_split


class ClassifierTree:
    """A class representing a decision tree.

    Args:
        max_depth: (int) - The maximum depth of the decision tree.

    Methods:
        learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
        classify(record): Classifies a record using the decision tree.
    """

    def __init__(self, max_depth=5):
        """Initializes the ClassifierTree with a maximum depth."""
        self.tree = {}  # Initialize the tree as an empty dictionary
        self.max_depth = max_depth  # Set the maximum depth of the tree
        self.info_gain = []  # Initialize the information gain list

    def fit(self, X, y, sample_weight=None):
        """Fits the decision tree to the training data.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target labels.
            sample_weight: (array-like) - The sample weights (default: None).
        """
        # Ensure y is integer type for bincount
        y = np.asarray(y).astype(int)
        self.tree = self.learn(X, y, sample_weight=sample_weight)
        return self.tree

    def learn(self, X, y, par_node=None, depth=0, sample_weight=None):
        """Builds the decision tree based on the given training data.

        Args:
            X: (array-like) - The input features.
            y: (array-like) - The target labels.
            par_node: (dict) - The parent node of the current subtree (default: {}).
            depth: (int) - The current depth of the subtree (default: 0).
            sample_weight: (array-like) - The sample weights (default: None).

        Returns:
            dict: The learned decision tree.
        """
        # Check type of X and y
        if not isinstance(X, list | np.ndarray) or not isinstance(y, list | np.ndarray):
            raise TypeError("X and y must be lists or NumPy arrays.")

        y = y.tolist() if isinstance(y, np.ndarray) else y  # Convert y to a Python list

        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y, dtype=int)

        # Handle sample_weight
        n_samples = len(y)
        if sample_weight is None or sample_weight is False:
            sample_weight = np.ones(n_samples)  # Equal weights if none provided
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight length mismatch.")

        # If X and Y are empty, return an empty dictionary
        if X.size == 0 and y.size == 0:
            return {}

        # If X is empty, return the most common label in y
        if X.size == 0:
            return {"label": max(set(y), key=list(y).count)}

        # Normalize weights for calculations within this node/subtree
        current_total_weight = np.sum(sample_weight)
        # Base case: Zero total weight in node
        if current_total_weight <= 0:
            # All samples have zero weight, cannot determine majority based on weight.
            # Fallback to unweighted majority or handle as error/default.
            if n_samples > 0:
                counts = np.bincount(y)
                majority_label = np.argmax(counts)
                return {"label": majority_label}
            else:
                return {}  # Empty node

        # Calculate weighted majority class for potential leaf node
        unique_classes, class_indices = np.unique(y, return_inverse=True)
        weighted_counts = np.bincount(class_indices, weights=sample_weight)
        majority_label = unique_classes[np.argmax(weighted_counts)]

        # Base cases
        if len(unique_classes) == 1:
            return {"label": y[0]}
        if depth >= self.max_depth:
            return {"label": majority_label}
        # Add base case for min_samples_split (can be weighted if needed)
        if n_samples < 2:  # Or some other threshold like self.min_samples_split
            return {"label": majority_label}

        # Find best split
        utility = ClassifierTreeUtility()
        best_split = utility.best_split(X, y, sample_weight)

        # If no beneficial split found or criteria met, return leaf
        if "label" in best_split:  # Check if best_split returned a leaf node structure
            return best_split
        if best_split is None or best_split["info_gain"] <= 0:
            return {"label": majority_label}

        # Build subtrees
        left_tree = self.learn(
            best_split["X_left"],
            best_split["y_left"],
            depth=depth + 1,
            sample_weight=best_split["sample_weight_left"],
        )
        right_tree = self.learn(
            best_split["X_right"],
            best_split["y_right"],
            depth=depth + 1,
            sample_weight=best_split["sample_weight_right"],
        )

        return {
            "split_attribute": best_split["split_attribute"],
            "split_val": best_split["split_val"],
            "left": left_tree,
            "right": right_tree,
        }

    @staticmethod
    def classify(tree, record):
        """Classifies a given record using the decision tree.

        Args:
            tree: (dict) - The decision tree.
            record: (dict) - A dictionary representing the record to be classified.

        Returns:
            The label assigned to the record based on the decision tree.
        """
        # If tree is empty return None
        if tree is None or tree == {}:
            return None

        if "label" in tree:
            return tree["label"]

        if record[tree["split_attribute"]] <= tree["split_val"]:
            return ClassifierTree.classify(tree["left"], record)
        else:
            return ClassifierTree.classify(tree["right"], record)

    def predict(self, X):
        """Predicts the labels for a given set of records using the decision tree.

        Args:
            X: (array-like) - The input features.

        Returns:
            list: A list of predicted labels for each record.
        """
        if self.tree is None or self.tree == {}:
            return None

        predictions = []

        for record in X:
            prediction = ClassifierTree.classify(self.tree, record)
            predictions.append(prediction)

        return predictions
