# Importing the required libraries
import csv
import numpy as np
import ast
from datetime import datetime
from math import log, floor, ceil
import random

class ClassifierTreeUtility(object):
    """
    Utility class for computing entropy, partitioning classes, and calculating information gain.
    """

    def entropy(self, class_y):
        """
        Computes the entropy for a given class.

        Parameters:
        - class_y (array-like): The class labels.

        Returns:
        - float: The entropy value.
        """
        # Handle empty arrays
        if len(class_y) == 0:
            return 0.0
        
        # Use NumPy's bincount to count occurrences of each class
        counts = np.bincount(class_y, minlength=np.max(class_y) + 1)
        total = len(class_y)

        # Avoid division by zero and skip zero probabilities
        probabilities = counts / total
        non_zero_probs = probabilities[probabilities > 0]

        # Use NumPy's dot product for efficient computation of entropy
        return -np.dot(non_zero_probs, np.log2(non_zero_probs))

    def partition_classes(self, X, y, split_attribute, split_val):
        """
        Partitions the dataset into two subsets based on a given split attribute and value.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target labels.
        - split_attribute (int): The index of the attribute to split on.
        - split_val (float): The value to split the attribute on.

        Returns:
        - X_left  (array-like): The subset of input features where the split attribute is less than or equal to the split value.
        - X_right (array-like): The subset of input features where the split attribute is greater than the split value.
        - y_left  (array-like): The subset of target labels corresponding to X_left.
        - y_right (array-like): The subset of target labels corresponding to X_right.
        """
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:             # If X has only one feature
            X = X.reshape(-1, 1)    # Convert to a 2D array with one column

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
        
        return X_left, X_right, y_left, y_right     # Return the partitioned subsets


    def information_gain(self, previous_y, current_y):
        """
        Calculates the information gain between the previous and current values of y.

        Parameters:
        - previous_y (array-like): The previous values of y.
        - current_y (array-like): The current values of y.

        Returns:
        - float: The information gain between the previous and current values of y.
        """
        entropy_prev = self.entropy(previous_y) # Compute the entropy of the previous y values
        total_count = len(previous_y)           # Get the total count of previous y values
        
        # Compute the weighted entropy of the current y values
        # For each subset in current_y, calculate its entropy and multiply it by the proportion of the subset in the total count
        entropy_current = np.sum([
            (len(subset) / total_count) * self.entropy(subset) for subset in current_y if len(subset) > 0
        ])
        
        # Information gain is the difference between the entropy of the previous y values and the weighted entropy of the current y values
        info_gain = entropy_prev - entropy_current
        
        return info_gain

    def best_split(self, X, y):
        """
        Finds the best attribute and value to split the data based on information gain.
    
        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target variable.
    
        Returns:
        - dict: A dictionary containing the best split attribute, split value, left and right subsets of X and y,
                and the information gain achieved by the split.
        """
        # Check type of X and y
        if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
            raise TypeError("X and y must be lists or NumPy arrays.")      
        
        # Convert X and y to NumPy arrays if they are not already
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
    
        if X.size == 0:  # If X is empty
            return {
                'split_attribute': None,
                'split_val': None,
                'X_left': np.empty((0, 1)),
                'X_right': np.empty((0, 1)),
                'y_left': np.empty((0,)),
                'y_right': np.empty((0,)),
                'info_gain': 0
            }
    
        if X.shape[0] == 1:  # If X has a single value
            return {
                'split_attribute': None,
                'split_val': None,
                'X_left': np.empty((0, X.shape[1])),
                'X_right': np.empty((0, X.shape[1])),
                'y_left': np.empty((0,)),
                'y_right': np.empty((0,)),
                'info_gain': 0
            }
    
        # Randomly select a subset of attributes for splitting
        num_features = int(np.sqrt(X.shape[1]))                                                 # Square root of total attributes
        selected_attributes = np.random.choice(X.shape[1], size=num_features, replace=False)    # Randomly select attributes
    
        # Initialize the best information gain to negative infinity, others to None
        best_info_gain = float('-inf')
        best_split = None
    
        # Use numpy's percentile function to reduce split points
        for split_attribute in selected_attributes:
            # Instead of trying all values, sample a subset of potential split points
            feature_values = X[:, split_attribute]
    
            # Use percentiles to get a representative sample of split points
            percentiles = np.percentile(feature_values, [25, 50, 75])
    
            for split_val in percentiles:
                X_left, X_right, y_left, y_right = self.partition_classes(X, y, split_attribute, split_val)
    
                # Skip if split doesn't divide the data
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
    
                info_gain = self.information_gain(y, [y_left, y_right])
    
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        'split_attribute': split_attribute,
                        'split_val': split_val,
                        'X_left': X_left,
                        'X_right': X_right,
                        'y_left': y_left,
                        'y_right': y_right,
                        'info_gain': info_gain
                    }
    
        if best_split is None:
            # If no good split found, return a default split
            return {
                'split_attribute': None,
                'split_val': None,
                'X_left': np.empty((0, X.shape[1])),
                'X_right': np.empty((0, X.shape[1])),
                'y_left': np.empty(0),
                'y_right': np.empty(0),
                'info_gain': 0
            }
    
        return best_split

class ClassifierTree(object):
    """
    A class representing a decision tree.

    Parameters:
    - max_depth (int): The maximum depth of the decision tree.

    Methods:
    - learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
    - classify(record): Classifies a record using the decision tree.

    """

    def __init__(self, max_depth=5):
        self.tree = {}              # Initialize the tree as an empty dictionary
        self.max_depth = max_depth  # Set the maximum depth of the tree
        self.info_gain = []             # Initialize the information gain list

    def learn(self, X, y, par_node={}, depth=0):
        """
        Builds the decision tree based on the given training data.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target labels.
        - par_node (dict): The parent node of the current subtree (default: {}).
        - depth (int): The current depth of the subtree (default: 0).

        Returns:
        - dict: The learned decision tree.
        """
        # Check type of X and y
        if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
            raise TypeError("X and y must be lists or NumPy arrays.")  
        
        y = y.tolist() if isinstance(y, np.ndarray) else y  # Convert y to a Python list
        
        # Convert X and y to NumPy arrays for faster computation
        X = np.array(X)
        y = np.array(y, dtype=int)
        
        # If X and Y are empty, return an empty dictionary
        if X.size == 0 and y.size == 0:
            return {}
        
        # If X is empty, return the most common label in y
        if X.size == 0:
            return {'label': max(set(y), key=list(y).count)}
        
        # Base cases
        if len(set(y)) == 1:        # If the node is pure (all labels are the same)
            return {'label': y[0]}  # Return the label as the value of the leaf node

        if depth >= self.max_depth:                     # If maximum depth is reached
            return {'label': np.argmax(np.bincount(y))} # Return the most common label

        # Find best split
        utility = ClassifierTreeUtility()
        best_split = utility.best_split(X, y)
        
        if best_split['split_attribute'] is None or best_split['info_gain'] <= 0:
            return {'label': max(set(y), key=list(y).count)}  # Return the most common label

        # Build subtrees
        left_tree = self.learn(best_split['X_left'], best_split['y_left'], depth=depth+1)
        right_tree = self.learn(best_split['X_right'], best_split['y_right'], depth=depth+1)
        
        return {
            'split_attribute': best_split['split_attribute'],
            'split_val': best_split['split_val'],
            'left': left_tree,
            'right': right_tree
        }

    @staticmethod
    def classify(tree, record):
        """
        Classifies a given record using the decision tree.

        Parameters:
        - tree (dict): The decision tree.
        - record (dict): A dictionary representing the record to be classified.

        Returns:
        - The label assigned to the record based on the decision tree.
        """
        # If tree is empty return None
        if tree is None or tree == {}:
            return None
        
        if 'label' in tree:
            return tree['label']
        
        if record[tree['split_attribute']] <= tree['split_val']:
            return ClassifierTree.classify(tree['left'], record)
        else:
            return ClassifierTree.classify(tree['right'], record)
