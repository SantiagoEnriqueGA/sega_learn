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
import multiprocessing
from joblib import Parallel, delayed

from .treeRegressor import RegressorTreeUtility, RegressorTree

def _fit_tree(X, y, max_depth):
    """Helper function for parallel tree fitting."""
    # Create bootstrapped sample
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_sample = X[indices]
    y_sample = y[indices]
    
    # Fit tree on bootstrapped sample
    tree = RegressorTree(max_depth=max_depth)
    return tree.learn(X_sample, y_sample)

def _predict_oob(X, trees, bootstraps):
    """Helper function for parallel out-of-bag predictions."""
    all_predictions = []
    
    for i, record in enumerate(X):
        predictions = []
        for j, (tree, bootstrap) in enumerate(zip(trees, bootstraps)):
            # Check if record is out-of-bag for this tree
            if i not in bootstrap:
                predictions.append(RegressorTree.predict(tree, record))
        
        if predictions:
            all_predictions.append(np.mean(predictions))
        else:
            # If not out-of-bag for any tree, use all trees
            all_predictions.append(np.mean([RegressorTree.predict(tree, record) for tree in trees]))
    
    return all_predictions

class RandomForestRegressor(object):
    """Optimized Random Forest Regressor class."""
    
    def __init__(self, n_estimators=100, max_depth=10, n_jobs=-1, random_state=None):
        """Initialize the Random Forest Regressor with optimized parameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, multiprocessing.cpu_count())
        self.random_state = random_state
        self.trees = []
        self.bootstraps = []
        
    def fit(self, X, y, verbose=False):
        """Fit the random forest with parallel processing."""
        start_time = datetime.now()
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
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
        
        y_pred = _predict_oob(X, self.trees, self.bootstraps)
        
        # Calculate evaluation metrics
        self.calculate_metrics(y, y_pred)
        
        if verbose:
            print(f"Execution time: {datetime.now() - start_time}")
            print(f"MSE:  {self.mse:.4f}")
            print(f"R^2:  {self.r2:.4f}")
            print(f"MAPE: {self.mape:.4f}%")
            print(f"MAE:  {self.mae:.4f}")
            print(f"RMSE: {self.rmse:.4f}")
        
        return self
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        self.mse = np.mean((y_true - y_pred) ** 2)
        self.ssr = np.sum((y_true - y_pred) ** 2)
        self.sst = np.sum((y_true - np.mean(y_true)) ** 2)
        self.r2 = 1 - (self.ssr / self.sst) if self.sst != 0 else 0
        
        # Handle zero values in y_true for MAPE
        mask = y_true != 0
        if np.any(mask):
            self.mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            self.mape = np.nan
            
        self.mae = np.mean(np.abs(y_true - y_pred))
        self.rmse = np.sqrt(self.mse)
    
    def predict(self, X):
        """Predict using the trained random forest."""
        X = np.asarray(X)
        
        # Make predictions for each tree
        predictions = []
        for tree in self.trees:
            tree_predictions = [RegressorTree.predict(tree, record) for record in X]
            predictions.append(tree_predictions)
        
        # Average predictions across trees
        return np.mean(predictions, axis=0)
    
    def get_stats(self, verbose=True):
        """Return the evaluation metrics."""
        stats = {
            "MSE": self.mse,
            "R^2": self.r2,
            "MAPE": self.mape,
            "MAE": self.mae,
            "RMSE": self.rmse
        }
        
        if verbose:
            for metric, value in stats.items():
                print(f"{metric}: {value:.4f}")
                
        return stats