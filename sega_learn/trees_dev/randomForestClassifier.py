"""
This module contains the implementation of a Random Forest Classifier.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- RandomForestWithInfoGain: A class representing a Random Forest model that returns information gain for vis.
- runRandomForest: A class that runs the Random Forest algorithm.
"""

# Importing the required libraries
import csv
import numpy as np
import ast
from datetime import datetime
from math import log, floor, ceil
import random
import multiprocessing
from joblib import Parallel, delayed

from .treeClassifier import ClassifierTreeUtility, ClassifierTree
from ..utils.metrics import Metrics

def _fit_tree(X, y, max_depth):
    """
    Helper function for parallel tree fitting. Fits a single tree on a bootstrapped sample.
    
    Args:
        X (array-like): The input features.
        y (array-like): The target labels.
        max_depth (int): The maximum depth of the tree.
    
    Returns:
        ClassifierTree: A fitted tree object.
    """
    # Create bootstrapped sample
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # Fit tree on bootstrapped sample
    tree = ClassifierTree(max_depth=max_depth)
    return tree.learn(X_sample, y_sample)

def _classify_oob(X, trees, bootstraps):
    """
    Helper function for parallel out-of-bag predictions. Classifies using out-of-bag samples.
    
    Args:
        X (array-like): The input features.
        trees (list): The list of fitted trees.
        bootstraps (list): The list of bootstrapped indices for each tree.

    Returns:
        list: The list of out-of-bag predictions. 
    """
    all_classifications = []
    
    for i, record in enumerate(X):
        classifications = []
        for j, (tree, bootstrap) in enumerate(zip(trees, bootstraps)):
            # Check if record is out-of-bag for this tree
            if i not in bootstrap:
                classifications.append(ClassifierTree.classify(tree, record))
        # Determine the majority vote
        if len(classifications) > 0:
            counts = np.bincount(classifications)
            majority_class = np.argmax(counts)
            all_classifications.append(majority_class)
        else:
            all_classifications.append(np.random.choice([0, 1]))
            
    return all_classifications

class RandomForestClassifier(object):
    """
    """   
    def __init__(self, forest_size=100, max_depth=10, n_jobs=-1, random_seed=None, X=None, y=None):
        """
        Initializes the RandomForest object.
        """
        self.n_estimators = forest_size
        self.max_depth = max_depth
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, multiprocessing.cpu_count())
        self.random_state = random_seed
        self.trees = []
        self.bootstraps = []
        
        self.X = X
        self.y = y
    
    def fit(self, X=None, y=None, verbose=False):
        """Fit the random forest with parallel processing."""
        if X is None and self.X is None:
            raise ValueError("X must be provided either during initialization or fitting.")
        if y is None and self.y is None:
            raise ValueError("y must be provided either during initialization or fitting.")        
        
        start_time = datetime.now()
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Convert inputs to numpy arrays
        X = np.asarray((X if X is not None else self.X))
        y = np.asarray((y if y is not None else self.y))
        
        if verbose:
            print("Fitting trees in parallel...")
        
        # Fit trees in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(X, y, self.max_depth) 
            for _ in range(self.n_estimators)
        )
        
        # Generate bootstrapped indices for OOB scoring
        self.bootstraps = [
            np.random.choice(len(X), size=len(X), replace=True)
            for _ in range(self.n_estimators)
        ]
        
        # Compute OOB predictions
        if verbose:
            print("Computing OOB predictions...")
                    
        y_pred = _classify_oob(X, self.trees, self.bootstraps)
        
        # Calculate evaluation metrics
        self.calculate_metrics(y, y_pred)
        
        if verbose:
            print(f"Execution time: {datetime.now() - start_time}")
            print(f"Accuracy:  {self.accuracy:.4f}")
            print(f"Precision: {self.precision:.4f}")
            print(f"Recall:    {self.recall:.4f}")
            print(f"F1 Score:  {self.f1_score:.4f}")
            print(f"Log Loss:  {self.log_loss:.4f}")
            
        return self
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics for classification."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        self.accuracy = Metrics.accuracy(y_true, y_pred)
        self.precision = Metrics.precision(y_true, y_pred)
        self.recall = Metrics.recall(y_true, y_pred)
        self.f1_score = Metrics.f1_score(y_true, y_pred)
        self.log_loss = Metrics.log_loss(y_true, y_pred)
        
    def predict(self, X):
        """Predict class labels for the provided data."""
        X = np.asarray(X)
        predictions = []

        for record in X:
            classifications = []
            for tree in self.trees:
                classifications.append(ClassifierTree.classify(tree, record))
            # Determine the majority vote
            counts = np.bincount(classifications)
            majority_class = np.argmax(counts)
            predictions.append(majority_class)

        return predictions

    def get_stats(self, verbose=False):
        """Return the evaluation metrics"""
        stats = {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 Score": self.f1_score,
            "Log Loss": self.log_loss
        }
        
        if verbose:
            for metric, value in stats.items():
                print(f"{metric}: {value:.4f}")
        
        return stats
    