"""
This module contains the implementation of a Random Forest Regressor.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- runRandomForest: A class that runs the Random Forest algorithm.
"""

# Importing the required libraries
import csv
import numpy as np
import ast
from datetime import datetime
from math import log, floor, ceil
import random

from .treeRegressor import RegressorTreeUtility, RegressorTree

class RandomForestRegressor(object):
    """
    A class representing a Random Forest model.

    Attributes:
        num_trees (int): The number of decision trees in the random forest.
        decision_trees (list): A list of decision trees in the random forest.
        bootstraps_datasets (list): A list of bootstrapped datasets for each tree.
        bootstraps_labels (list): A list of corresponding labels for each bootstrapped dataset.
        max_depth (int): The maximum depth of each decision tree.

    Methods:
        __init__(num_trees, max_depth): Initializes the RandomForest object.
        _bootstrapping(XX, n): Performs bootstrapping on the dataset.
        bootstrapping(XX): Initializes the bootstrapped datasets for each tree.
        fitting(): Fits the decision trees to the bootstrapped datasets.
        voting(X): Performs voting to predict the target values for the input records.
        user(): Returns the user's GTUsername.
    """
    # Initialize class variables
    num_trees = 0               # Number of decision trees in the random forest
    decision_trees = []         # List of decision trees in the random forest
    bootstraps_datasets = []    # List of bootstrapped datasets for each tree
    bootstraps_labels = []      # List of true class labels corresponding to records in the bootstrapped datasets
    max_depth = 10              # Maximum depth of each decision tree

    random_seed = 0     # Random seed for reproducibility
    forest_size = 10    # Number of trees in the random forest
    max_depth = 10      # Maximum depth of each decision tree
    display = False     # Flag to display additional information about info gain
    
    X = list()          # Data features
    y = list()          # Data labels
    XX = list()         # Contains both data features and data labels
    numerical_cols = 0  # Number of numeric attributes (columns)

    def __init__(self, X=None, y=None, forest_size=10, random_seed=0, max_depth=10):
        """
        Initializes the RandomForest object.

        Args:
            X (ndarray): The input data features.
            y (ndarray): The target values.
            forest_size (int): The number of decision trees in the random forest.
            random_seed (int): The random seed for reproducibility.
            max_depth (int): The maximum depth of each decision tree.
        """
        self.reset()    # Reset the random forest object

        self.random_seed = random_seed  # Set the random seed for reproducibility
        np.random.seed(random_seed)     # Set the random seed for NumPy

        self.forest_size = forest_size  # Set the number of trees in the random forest
        self.max_depth = max_depth      # Set the maximum depth of each decision tree
            
        if X is not None: self.X = X.tolist()             # Convert ndarray to list
        if y is not None: self.y = y.tolist()             # Convert ndarray to list
        if X is not None and y is not None: self.XX = [list(x) + [y] for x, y in zip(X, y)]  # Combine X and y

        self.num_trees = forest_size    # Set the number of trees
        self.max_depth = max_depth      # Set the maximum depth

        self.decision_trees = [RegressorTree(max_depth) for i in range(forest_size)]  # Initialize the decision trees
        
        self.bootstraps_datasets = []   # Initialize the list of bootstrapped datasets
        self.bootstraps_labels = []     # Initialize the list of corresponding labels

    def reset(self):
        """
        Resets the random forest object.
        """
        # Reset the random forest object
        self.random_seed = 0
        self.forest_size = 10
        self.max_depth = 10
        self.display = False
        self.X = list()
        self.y = list()
        self.XX = list()
        self.numerical_cols = 0
    
    def _bootstrapping(self, XX, n):
        """
        Performs bootstrapping on the dataset.

        Args:
            XX (list): The dataset.
            n (int): The number of samples to be selected.

        Returns:
            tuple: A tuple containing the bootstrapped dataset and the corresponding labels.
        """
        sample_indices = np.random.choice(len(XX), size=n, replace=True)    # Randomly select indices with replacement
        
        sample = [XX[i][:-1] for i in sample_indices]   # Get the features of the selected samples
        labels = [XX[i][-1] for i in sample_indices]    # Get the labels of the selected samples
        
        return (sample, labels)

    def bootstrapping(self, XX):
        """
        Initializes the bootstrapped datasets for each tree.

        Args:
            XX (list): The dataset.
        """
        if not isinstance(XX, list):
            raise TypeError("XX must be a list")  # Raise an error if XX is not a list
        
        for i in range(self.num_trees):                                 # For each tree
            data_sample, data_label = self._bootstrapping(XX, len(XX))  # Perform bootstrapping on the dataset
            self.bootstraps_datasets.append(data_sample)                # Add the bootstrapped dataset to the list
            self.bootstraps_labels.append(data_label)                   # Add the corresponding labels to the list

    def fitting(self):
        """
        Fits the decision trees to the bootstrapped datasets.
        """
        for i in range(self.num_trees):             # For each tree
            tree = self.decision_trees[i]           # Get the current tree
            dataset = self.bootstraps_datasets[i]   # Get the bootstrapped dataset
            labels = self.bootstraps_labels[i]      # Get the corresponding labels
            
            self.decision_trees[i] = tree.learn(dataset, labels)    # Fit the tree to the bootstrapped dataset

    def voting(self, X):
        """
        Performs voting to predict the target values for the input records.

        Args:
            X (list): The input records.

        Returns:
            list: The predicted target values for the input records.
        """
        y = []
        for record in X:        # For each record
            predictions = []
            for i, dataset in enumerate(self.bootstraps_datasets):  # For each bootstrapped dataset
                
                # Records not in the dataset are considered out-of-bag (OOB) records, which can be used for voting
                if record not in dataset:               # If the record is not in the dataset
                    OOB_tree = self.decision_trees[i]   # Get the decision tree corresponding to the dataset
                    prediction = RegressorTree.predict(OOB_tree, record)    # Predict the target value for the record
                    predictions.append(prediction)      # Add the prediction to the votes list

            # Calculate the mean prediction for the record
            if len(predictions) > 0:                    # If there are predictions
                mean_prediction = np.mean(predictions)  # Calculate the mean prediction
                y.append(mean_prediction)               # Add the mean prediction to the list
            
            else:   # If the record is not out-of-bag (OOB), use all trees for prediction
                for i in range(self.num_trees):     # For each tree
                    tree = self.decision_trees[i]   # Get the current tree
                    predictions.append(RegressorTree.predict(tree, record)) # Predict the target value for the record
                    
                y.append(np.mean(predictions))      # Add the mean prediction to the list

        return y

    def fit(self, X=None, y=None, verbose=False):
        """
        Runs the random forest algorithm.
        """
        if X is not None: self.X = X.tolist()
        if y is not None: self.y = y.tolist()
        if X is not None and y is not None: self.XX = [list(x) + [y] for x, y in zip(X, y)]  # Combine X and y
        
        start = datetime.now()  # Start time
 
        if verbose: print("creating the bootstrap datasets")
        self.bootstrapping(self.XX)         # Create the bootstrapped datasets

        if verbose: print("fitting the forest")
        self.fitting()                      # Fit the decision trees to the bootstrapped datasets
        y_predicted = self.voting(self.X)   # Predict the target values for the input records
        
        if verbose: print("Execution time: " + str(datetime.now() - start))

        # Calculate evaluation metrics
        self.mse = np.mean((np.array(y_predicted) - np.array(self.y)) ** 2)  # Calculate the mean squared error (MSE): mean((y_true - y_pred)^2)
        self.ssr = np.sum((np.array(y_predicted) - np.array(self.y)) ** 2)   # Calculate the sum of squared residuals (SSR): sum((y_true - y_pred)^2)
        self.sst = np.sum((np.array(self.y) - np.mean(self.y)) ** 2)         # Calculate the total sum of squares (SST): sum((y_true - mean(y_true))^2)
        self.r2 = 1 - (self.ssr / self.sst)                                            # Calculate the R^2 score: 1 - (SSR / SST)
        self.mape = np.mean(np.abs((np.array(self.y) - 
                               np.array(y_predicted)) / np.array(self.y))) * 100    # Calculate the mean absolute percentage error (MAPE): mean(abs((y_true - y_pred) / y_true)) * 100
        self.mae = np.mean(np.abs(np.array(self.y) - np.array(y_predicted)))             # Mean Absolute Error (MAE): mean(abs(y_true - y_pred))
        self.rmse = np.sqrt(np.mean((np.array(y_predicted) - np.array(self.y)) ** 2))    # Root Mean Squared Error (RMSE): sqrt(mean((y_true - y_pred)^2))

        # Print the evaluation metrics
        if verbose: print("MSE:  %.4f" % self.mse)
        if verbose: print("R^2:  %.4f" % self.r2)
        if verbose: print("MAPE: %.4f%%" % self.mape)
        if verbose: print("MAE:  %.4f" % self.mae)
        if verbose: print("RMSE: %.4f" % self.rmse)

    def get_stats(self, verbose=True):
        """
        Returns the evaluation metrics.
        """
        return {"MSE": self.mse,"R^2": self.r2,"MAPE": self.mape,"MAE": self.mae,"RMSE": self.rmse}
    
    def predict(self, X=None):
        """
        Predicts the target values for the input data.
        """
        if X is not None: self.X = X.tolist()
        return self.voting(self.X)