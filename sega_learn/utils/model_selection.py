import numpy as np
from itertools import product

from .dataPrep import DataPrep
from .metrics import Metrics


class Utility:
    @staticmethod
    def get_param_combinations(param_grid):
        """
        Generates all possible combinations of hyperparameters.
        
        Returns:
        - param_combinations (list): A list of dictionaries containing hyperparameter combinations.
        """
        all_params = {}
        for grid in param_grid:
            all_params.update(grid)
        
        keys, values = zip(*all_params.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        return param_combinations


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
        
        assert self.param_grid, "param_grid cannot be empty."
        
        # Generate all possible hyperparameter combinations
        self.param_combinations = Utility.get_param_combinations(self.param_grid)
    
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

class RandomSearchCV(object):
    """
    Implements a random search cross-validation for hyperparameter tuning.
    """
    def __init__(self, model, param_grid, iter=10, cv=5, metric='mse', direction='minimize'):
        """
        Initializes the RandomSearchCV object.
        
        Parameters:
        - model: The model Object to be tuned.
        - param_grid (list): A list of dictionaries containing hyperparameters to be tuned.
        - iter (int): The number of iterations for random search. Default is 10.
        - cv (int): The number of folds for cross-validation. Default is 5.
        - metric (str): The metric to be used for evaluation. Default is 'mse'.
            - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
            - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
        - direction (str): The direction to optimize the metric. Default is 'minimize'.
        """
        self.model = model
        self.param_grid = param_grid
        self.iter = iter
        self.cv = cv
        self.metric = metric
        self.direction = direction
        
        assert self.iter > 0, "iter must be greater than 0."
        assert self.param_grid, "param_grid cannot be empty."
        
        # Generate all possible hyperparameter combinations
        self.param_combinations = Utility.get_param_combinations(self.param_grid)
        
    def fit(self, X, y, verbose=False):
        """
        Fits the model to the data for iter random hyperparameter combinations.
        
        Parameters:
        - X (numpy.ndarray): The feature columns.
        - y (numpy.ndarray): The label column.
        - verbose (bool): A flag to display the training progress. Default is True.
        """
        if self.direction == 'minimize': self.best_score_ = np.inf
        if self.direction == 'maximize': self.best_score_ = -np.inf
        self.best_params_ = None
        
        # Store tried combinations
        self.tried_params = []
        
        for i in range(self.iter):
            # Check if all parameter combinations have been tried
            if len(self.tried_params) >= len(self.param_combinations):
                print("All parameter combinations have been tried.")
                break
            
            # Randomly select hyperparameters
            params = np.random.choice(self.param_combinations)
            while params in self.tried_params:
                params = np.random.choice(self.param_combinations)
            
            # Store tried combinations
            self.tried_params.append(params)
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
            

class segaSearchCV(object):
    """
    Implements a custom search cross-validation for hyperparameter tuning.
    --DO NOT USE, WORK IN PROGRESS--
    """
    def __init__(self, model, param_space, iter=10, cv=5, metric='mse', direction='minimize'):
        """
        Initializes the segaSearchCV object.
        
        Parameters:
        - model: The model Object to be tuned.
        - param_space (list): A list of dictionaries containing hyperparameters to be tuned. 
            Should be in the format: [{'param': [type, min, max]}, ...]
        - iter (int): The number of iterations for random search. Default is 10.
        - cv (int): The number of folds for cross-validation. Default is 5.
        - metric (str): The metric to be used for evaluation. Default is 'mse'.
            - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
            - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
        - direction (str): The direction to optimize the metric. Default is 'minimize'.
        """
        self.model = model
        self.param_space = param_space
        self.iter = iter
        self.cv = cv
        self.metric = metric
        self.direction = direction
        
        assert self.iter > 0, "iter must be greater than 0."
        assert self.param_space, "param_grid cannot be empty."
        
        self.param_lims = {}
        for param in self.param_space:
            key = list(param.keys())[0]
            self.param_lims[key] = param[key][1:]        
        
    def fit(self, X, y, verbose=False):
        """
        Fits the model to the data for iter random hyperparameter combinations.
        
        Parameters:
        - X (numpy.ndarray): The feature columns.
        - y (numpy.ndarray): The label column.
        - verbose (bool): A flag to display the training progress. Default is True.
        """
        if self.direction == 'minimize': self.best_score_ = np.inf
        if self.direction == 'maximize': self.best_score_ = -np.inf
        self.best_params_ = None
        
        params = {}
        directions = {}
        self.tried_params = []
        
        for i in range(self.iter):
            # If params is empty, set to midpoint of param_space
            if not params:
                for param in self.param_space:
                    key = list(param.keys())[0]
                    if param[key][0] == int:
                        params[key] = (param[key][1] + param[key][2]) // 2
                        step_size = (param[key][2] - param[key][1]) // 4
                        directions[key] = [1, step_size]    # [direction, step_size]
                    elif param[key][0] == float:
                        params[key] = (param[key][1] + param[key][2]) / 2
                        step_size = (param[key][2] - param[key][1]) / 4
                        directions[key] = [1, step_size]    # [direction, step_size]
            
            # If params were already tried, find closest untried params
            while params in self.tried_params:
                for param in self.param_space:
                    key = list(param.keys())[0]
                    step_size = directions[key][1] // 2
                    
                    if directions[key][0] == 1:
                        params[key] = min(params[key] + step_size, self.param_lims[key][1])
                    else:
                        params[key] = max(params[key] - step_size, self.param_lims[key][0])
                        
            
            # If params are out of bounds break
            if any(params[key] < self.param_lims[key][0] or params[key] > self.param_lims[key][1] for key in params):
                break
            
            self.tried_params.append(params.copy())
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
                self.best_params_ = params.copy()
                self.best_model = self.active_model
            
                # If the score improved, continue in the same direction
                for key in params:
                    step_size = directions[key][1] // 2
                    if directions[key][0] == 1:
                        params[key] = min(params[key] + step_size, self.param_lims[key][1])
                    else:
                        params[key] = max(params[key] - step_size, self.param_lims[key][0])
            elif self.direction == 'maximize' and mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params.copy()
                self.best_model = self.active_model
                
                # If the score improved, continue in the same direction
                for key in params:
                    step_size = directions[key][1] // 2
                    if directions[key][0] == 1:
                        params[key] = min(params[key] + step_size, self.param_lims[key][1])
                    else:
                        params[key] = max(params[key] - step_size, self.param_lims[key][0])
            else:
                # If the score did not improve, change direction
                for key in params:
                    directions[key][0] *= -1
                    step_size = directions[key][1] // 2
                    if directions[key][0] == 1:
                        params[key] = min(params[key] + step_size, self.param_lims[key][1])
                    else:
                        params[key] = max(params[key] - step_size, self.param_lims[key][0])
        
        return self.best_model