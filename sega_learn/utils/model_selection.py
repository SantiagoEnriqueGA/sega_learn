import numpy as np
from itertools import product

from .dataPrep import DataPrep
from .metrics import Metrics


class GridSearchCV(object):
    """
    Implements a grid search cross-validation for hyperparameter tuning.
    """
    def __init__(self, model, param_grid, cv=5, metric='mse', direction='minimize'):
        """
        Initializes the GridSearchCV object.
        
        Parameters:
        - model: The model Object to be tuned.
        - param_grid (list): A list of dictionaries containing hyperparameters to be tuned.
        - cv (int): The number of folds for cross-validation. Default is 5.
        - metric (str): The metric to be used for evaluation. Default is 'mse'.
            - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
            - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
        - direction (str): The direction to optimize the metric. Default is 'minimize'.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.metric = metric
        self.direction = direction
        
        # Generate all possible hyperparameter combinations
        self.param_combinations = self._get_param_combinations()
    
    def _get_param_combinations(self):
        """
        Generates all possible combinations of hyperparameters.
        
        Returns:
        - param_combinations (list): A list of dictionaries containing hyperparameter combinations.
        """
        all_params = {}
        for grid in self.param_grid:
            all_params.update(grid)
        
        keys, values = zip(*all_params.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        return param_combinations
    
    def fit(self, X, y, verbose=False):
        """
        Fits the model to the data for all hyperparameter combinations.
        
        Parameters:
        - X (numpy.ndarray): The feature columns.
        - y (numpy.ndarray): The label column.
        - verbose (bool): A flag to display the training progress. Default is True.
        """
        if self.direction == 'minimize': self.best_score_ = np.inf
        if self.direction == 'maximize': self.best_score_ = -np.inf
        self.best_params_ = None
        
        for params in self.param_combinations:
            scores = []
            
            if verbose: print(f"Training Model with Params: {params}")
            for i in range(self.cv):
                self.active_model = self.model(**params)
                
                
                X_folds, y_folds = DataPrep.k_split(X, y, k=self.cv)
                X_train, y_train = np.concatenate(X_folds[:i] + X_folds[i+1:]), np.concatenate(y_folds[:i] + y_folds[i+1:])
                X_test, y_test = X_folds[i], y_folds[i]
                
                self.active_model.fit(X_train, y_train)
                y_pred = self.active_model.predict(X_test)
                
                # Regression Metrics
                if self.metric in ['mse', 'mean_squared_error']: s = Metrics.mean_squared_error(y_test, y_pred)
                elif self.metric in ['r2', 'r_squared']: s = Metrics.r_squared(y_test, y_pred)
                elif self.metric in ['mae', 'mean_absolute_error']: s = Metrics.mean_absolute_error(y_test, y_pred)
                elif self.metric in ['rmse', 'root_mean_squared_error']: s = Metrics.root_mean_squared_error(y_test, y_pred)
                elif self.metric in ['mape', 'mean_absolute_percentage_error']: s = Metrics.mean_absolute_percentage_error(y_test, y_pred)
                elif self.metric in ['mpe', 'mean_percentage_error']: s = Metrics.mean_percentage_error(y_test, y_pred)
                
                # Classification Metrics
                elif self.metric == 'accuracy': s = Metrics.accuracy(y_test, y_pred)
                elif self.metric == 'precision': s = Metrics.precision(y_test, y_pred)
                elif self.metric == 'recall': s = Metrics.recall(y_test, y_pred)
                elif self.metric == 'f1': s = Metrics.f1_score(y_test, y_pred)
                elif self.metric == 'log_loss': s = Metrics.log_loss(y_test, y_pred)
                
                scores.append(s)
                    
                if verbose: print(f"\tCV Fold {i+1}: - {self.metric}: {s:.2f}")
                
            
            mean_score = np.mean(scores)
            if verbose: print(f"\t-Mean Score: {mean_score:.2f}")
            
            if self.direction == 'minimize' and mean_score < self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_model = self.active_model
            if self.direction == 'maximize' and mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_model = self.active_model
        
        return self.best_model


# TODO: Implement RandomSearchCV
# TODO: Implement BayesSearchCV