from ..svm import *
from ..trees import *
from ..neural_networks import *
from ..nearest_neighbors import *
from ..linear_models.linearModels import *

from ..utils.metrics import Metrics
r_squared = Metrics.r_squared
root_mean_squared_error = Metrics.root_mean_squared_error
mean_absolute_percentage_error = Metrics.mean_absolute_percentage_error

import time
import numpy as np

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except:
    TQDM_AVAILABLE = False

class AutoRegressor:
    """
    A class to automatically select and evaluate the best regression model for a given dataset.
    It uses various regression models and compares their performance using metrics such as R-squared, RMSE, and MAPE.
    """

    def __init__(self):
        """
        Initializes the AutoRegressor with a set of predefined regression models.
        """
        # Each model should have a fit and predict method
        self.models = {
            # Linear Models
            "OrdinaryLeastSquares": OrdinaryLeastSquares(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Bayesian": Bayesian(),
            "RANSAC": RANSAC(),
            "PassiveAggressive": PassiveAggressiveRegressor(),
            
            # SVM
            "LinearSVR": LinearSVR(),
            "GeneralizedSVR": GeneralizedSVR(),
            
            # Nearest Neighbors
            "KNeighborsRegressor": KNeighborsRegressor(),
            
            # Trees
            "RegressorTree": RegressorTree(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostedRegressor": GradientBoostedRegressor(),
            
            # Neural Networks - Not yet implemented
            # "NeuralNetworkRegressor": NeuralNetworkRegressor(),
        }
        # Add model classes for better categorization in the summary
        self.model_classes = {
            # Linear Models
            "OrdinaryLeastSquares": "Linear",
            "Ridge": "Linear",
            "Lasso": "Linear",
            "Bayesian": "Linear",
            "RANSAC": "Linear",
            "PassiveAggressive": "Linear",
            
            # SVM
            "LinearSVR": "SVM",
            "GeneralizedSVR": "SVM",
            
            # Nearest Neighbors
            "KNeighborsRegressor": "Nearest Neighbors",
            
            # Trees
            "RegressorTree": "Trees",
            "RandomForestRegressor": "Trees",
            "GradientBoostedRegressor": "Trees",
        }
        self.predictions = {}
        self.results = []

    def fit(self, X_train, y_train, X_test=None, y_test=None, custom_metrics=None, verbose=False):
        """
        Fits the regression models to the training data and evaluates their performance.

        Args:
            - X_train (np.ndarray) - Training feature data.
            - y_train (np.ndarray) - Training target data.
            - X_test (np.ndarray), optional - Testing feature data (default is None).
            - y_test (np.ndarray), optional - Testing target data (default is None).
            - custom_metrics (dict: str -> callable), optional - Custom metrics for evaluation (default is None).
            - verbose (bool), optional - If True, prints progress (default is False).

        Returns:
            - results (list) - A list of dictionaries containing model performance metrics.
            - predictions (dict) - A dictionary of predictions for each model.
        """       
        progress_bar = tqdm(self.models.items(), desc="Fitting Models", disable=not verbose or not TQDM_AVAILABLE) if TQDM_AVAILABLE else self.models.items()

        for name, model in progress_bar:
            if TQDM_AVAILABLE: progress_bar.set_description(f"Fitting {name}")
            start_time = time.time()

            if X_test is not None and y_test is not None:
                try:
                    # Attempt to fit using both training and testing data if the model supports it
                    model.fit(X_train, y_train, X_test, y_test)
                except TypeError:
                    # If the model does not support separate test data, combine train and test
                    X_combined = np.vstack((X_train, X_test))
                    y_combined = np.hstack((y_train, y_test))
                    model.fit(X_combined, y_combined)
            else:
                # Fit using only training data
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test if X_test is not None else X_train)
            elapsed_time = time.time() - start_time

            metrics = {}
            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    metrics[metric_name] = metric_func(y_test if y_test is not None else y_train, y_pred)
            else:
                metrics["R-Squared"] = r_squared(y_test if y_test is not None else y_train, y_pred)
                metrics["RMSE"] = root_mean_squared_error(y_test if y_test is not None else y_train, y_pred)
                metrics["MAPE"] = mean_absolute_percentage_error(y_test if y_test is not None else y_train, y_pred)

            self.predictions[name] = y_pred
            self.results.append({
                "Model": name,
                **metrics,
                "Time Taken": elapsed_time
            })
            
        
        if TQDM_AVAILABLE: progress_bar.set_description("Fitting Completed")
        return self.results, self.predictions

    def predict(self, X, model=None):
        """
        Generates predictions for the given input data using all fitted models.

        Parameters:
            - X (np.ndarray) - Input feature data.
            - model (str), optional - Specific model name to use for predictions (default is None).

        Returns:
            - predictions (dict) - A dictionary of predictions for each model.
        """
        if model:
            if model not in self.models:
                raise ValueError(f"Model '{model}' not found.")
            return self.models[model].predict(X)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions

    def evaluate(self, y_true, custom_metrics=None, model=None):
        """
        Evaluates the performance of the fitted models using the provided true values.

        Args:
            - y_true (np.ndarray): True target values.
            - custom_metrics (dict: str -> callable), optional: Custom metrics for evaluation (default is None).
            - model (str), optional: Specific model name to evaluate (default is None).

        Returns:
            - evaluation_results (dict): A dictionary containing evaluation metrics for the specified model(s).
        """
        evaluation_results = {}

        # Select specific model or all models
        models_to_evaluate = {model: self.models[model]} if model else self.models

        for name, model_instance in models_to_evaluate.items():
            if name not in self.predictions:
                raise ValueError(f"Model '{name}' has not been fitted yet.")

            y_pred = self.predictions[name]
            metrics = {}

            # Calculate metrics
            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    metrics[metric_name] = metric_func(y_true, y_pred)
            else:
                metrics["R-Squared"] = r_squared(y_true, y_pred)
                metrics["RMSE"] = root_mean_squared_error(y_true, y_pred)
                metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)

            evaluation_results[name] = metrics

        return evaluation_results
    
    def get_model(self, model_name):
        """
        Returns the model instance for the specified model name.

        Args:
            - model_name (str): The name of the model.

        Returns:
            - model_instance: The model instance.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        return self.models[model_name]

    def summary(self):
        """
        Prints a summary of the performance of all fitted models, sorted by R-squared, RMSE, and time taken.
        Dynamically adjusts to display the metrics used during evaluation.
        """
        if not self.results:
            print("No models have been fitted yet.")
            return

        # Extract all metric keys dynamically from the results
        metric_keys = [key for key in self.results[0].keys() if key not in ["Model", "Model Class", "Time Taken"]]

        # Sort results by R-Squared (if available), RMSE (if available), and Time Taken
        sorted_results = sorted(
            self.results,
            key=lambda x: (
                -x.get("R-Squared", float('-inf')),  # Descending R-Squared
                x.get("RMSE", float('inf')),         # Ascending RMSE
                x["Time Taken"]                      # Ascending Time Taken
            )
        )

        # Add model class to each result
        for result in sorted_results:
            result["Model Class"] = self.model_classes.get(result["Model"], "Unknown")

        try:
            from tabulate import tabulate
            headers = ["Model Class", "Model"] + metric_keys + ["Time Taken"]
            table_data = []
            for result in sorted_results:
                row = [result["Model Class"], result["Model"]]
                row += [result[key] for key in metric_keys]
                row.append(result["Time Taken"])
                table_data.append(row)
            print(tabulate(table_data, headers=headers, tablefmt="rounded_outline"))
        except Exception as e:
            # For environments where tabulate is not available, fallback to basic printing
            # Print header dynamically based on metrics
            header = "| Model Class       | Model                 | " + " | ".join(metric_keys) + " | Time Taken |"
            separator = "|:------------------|:----------------------|" + "|".join([":-----------" for _ in metric_keys]) + "|:------------|"
            print(header)
            print(separator)

            # Print each result dynamically
            for result in sorted_results:
                metrics_values = " | ".join(f"{result[key]:<9.6f}" for key in metric_keys)
                print(f"| {result['Model Class']:<16} | {result['Model']:<22} | {metrics_values} | {result['Time Taken']:<10.6f} |")


