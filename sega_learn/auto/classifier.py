from ..svm import *
from ..trees import *
from ..neural_networks import *
from ..nearest_neighbors import *
from ..linear_models.linearModels import *

from sega_learn.utils.metrics import Metrics
accuracy = Metrics.accuracy
precision = Metrics.precision
recall = Metrics.recall
f1 = Metrics.f1_score

import time
import numpy as np

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except:
    TQDM_AVAILABLE = False

class AutoClassifier:
    """
    A class to automatically select and evaluate the best classification model for a given dataset.
    It uses various classification models and compares their performance using metrics such as accuracy, precision, recall, F1 score, and ROC AUC.
    """

    def __init__(self):
        """
        Initializes the AutoClassifier with a set of predefined classification models.
        """
        self.models = {
            # Linear Models - Not yet implemented
            # "LogisticRegression": LogisticRegression(),
            # "SGDClassifier": SGDClassifier(),
            
            # SVM
            "LinearSVC": LinearSVC(),
            "GeneralizedSVC": GeneralizedSVC(),
            "OneClassSVM": OneClassSVM(),
            
            # Nearest Neighbors
            "KNeighborsClassifier": KNeighborsClassifier(),
            
            # Trees
            "ClassifierTree": ClassifierTree(),
            "RandomForestClassifier": RandomForestClassifier(),
            # GradientBoostingClassifier: GradientBoostingClassifier(), <- Not implemented yet
            
            # Neural Networks
            # Cannot be initialized here as it requires layer size (input/output size)
            # We can initialize it in the fit method and add it to the models dictionary
            "BaseBackendNeuralNetwork": None,  # Placeholder
        }
        self.models_classes = {
            # Linear Models - Not yet implemented
            "LogisticRegression": "Linear",
            "SGDClassifier": "Linear",
            
            # SVM
            "LinearSVC": "SVM",
            "GeneralizedSVC": "SVM",
            "OneClassSVM": "SVM",
            
            # Nearest Neighbors
            "KNeighborsClassifier": "Nearest Neighbors",
            
            # Trees
            "ClassifierTree": "Trees",
            "RandomForestClassifier": "Trees",            
            
            # Neural Networks
            "BaseBackendNeuralNetwork": "Neural Networks",
        }
        self.predictions = {}
        self.results = []

    def fit(self, X_train, y_train, X_test=None, y_test=None, custom_metrics=None, verbose=False):
        """
        Fits the classification models to the training data and evaluates their performance.

        Args:
            - X_train (np.ndarray): Training feature data.
            - y_train (np.ndarray): Training target data.
            - X_test (np.ndarray), optional: Testing feature data (default is None).
            - y_test (np.ndarray), optional: Testing target data (default is None).
            - custom_metrics (dict: str -> callable), optional: Custom metrics for evaluation (default is None).
            - verbose (bool), optional: If True, prints progress (default is False).

        Returns:
            - results (list): A list of dictionaries containing model performance metrics.
            - predictions (dict): A dictionary of predictions for each model.
        """
        # Initialize neural network if not already set
        if self.models["BaseBackendNeuralNetwork"] is None:
            input_size = X_train.shape[1]
            output_size = len(np.unique(y_train))
            layers = [128, 64, 32]    # Default hidden layers
            dropout_rate = 0.1
            reg_lambda = 0.0
            activations = ['relu'] * len(layers) + ['softmax']
            self.models["BaseBackendNeuralNetwork"] = BaseBackendNeuralNetwork(
                [input_size] + layers + [output_size],
                dropout_rate=dropout_rate,
                reg_lambda=reg_lambda,
                activations=activations
            )
            
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
                metrics["Accuracy"] = accuracy(y_test if y_test is not None else y_train, y_pred)
                metrics["Precision"] = precision(y_test if y_test is not None else y_train, y_pred)
                metrics["Recall"] = recall(y_test if y_test is not None else y_train, y_pred)
                metrics["F1 Score"] = f1(y_test if y_test is not None else y_train, y_pred)
                
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

        Args:
            - X (np.ndarray): Input feature data.
            - model (str), optional: Specific model name to use for predictions (default is None).

        Returns:
            - predictions (dict): A dictionary of predictions for each model.
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

        models_to_evaluate = {model: self.models[model]} if model else self.models

        for name, model_instance in models_to_evaluate.items():
            if name not in self.predictions:
                raise ValueError(f"Model '{name}' has not been fitted yet.")

            y_pred = self.predictions[name]
            metrics = {}

            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    metrics[metric_name] = metric_func(y_true, y_pred)
            else:
                metrics["Accuracy"] = accuracy(y_true, y_pred)
                metrics["Precision"] = precision(y_true, y_pred)
                metrics["Recall"] = recall(y_true, y_pred)
                metrics["F1 Score"] = f1(y_true, y_pred)

            evaluation_results[name] = metrics

        return evaluation_results

    def summary(self):
        """
        Prints a summary of the performance of all fitted models, sorted by Accuracy, F1 Score, and time taken.
        Dynamically adjusts to display the metrics used during evaluation.
        """
        if not self.results:
            print("No models have been fitted yet.")
            return

        # Extract all metric keys dynamically from the results
        metric_keys = [key for key in self.results[0].keys() if key not in ["Model", "Model Class", "Time Taken"]]

        # Sort results by Accuracy (if available), F1 Score (if available), and Time Taken
        sorted_results = sorted(
            self.results,
            key=lambda x: (
                -x.get("Accuracy", float('-inf')),  # Descending Accuracy
                -x.get("F1 Score", float('-inf')),  # Descending F1 Score
                x["Time Taken"]                     # Ascending Time Taken
            )
        )

        # Add model class to each result
        for result in sorted_results:
            result["Model Class"] = self.models_classes.get(result["Model"], "Unknown")

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