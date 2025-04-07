import time

import numpy as np

from ..linear_models.linearModels import *
from ..nearest_neighbors import *
from ..neural_networks import *
from ..svm import *
from ..trees import *
from ..utils import Scaler
from ..utils.metrics import Metrics

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except Exception as _e:
    TQDM_AVAILABLE = False

r_squared = Metrics.r_squared
root_mean_squared_error = Metrics.root_mean_squared_error
mean_absolute_percentage_error = Metrics.mean_absolute_percentage_error


class AutoRegressor:
    """A class to automatically select and evaluate the best regression model for a given dataset.

    Uses various regression models and compares their performance using metrics such as R-squared, RMSE, and MAPE.
    """

    def __init__(self, all_kernels=False):
        """Initializes the AutoRegressor with a set of predefined regression models.

        Args:
            all_kernels: (bool) - If True, include all kernels in the model list. Default is False.
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
            "GeneralizedSVR - Linear": GeneralizedSVR(kernel="linear"),
            # Nearest Neighbors
            "KNeighborsRegressor": KNeighborsRegressor(),
            # Trees
            "RegressorTree": RegressorTree(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostedRegressor": GradientBoostedRegressor(),
            # Neural Networks
            #   Cannot be initialized here as it requires layer size (input/output size)
            #   We can initialize it in the fit method and add it to the models dictionary
            "BaseBackendNeuralNetwork": None,
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
            "GeneralizedSVR - Linear": "SVM",
            "GeneralizedSVR - RBF": "SVM",
            "GeneralizedSVR - Polynomial": "SVM",
            # Nearest Neighbors
            "KNeighborsRegressor": "Nearest Neighbors",
            # Trees
            "RegressorTree": "Trees",
            "RandomForestRegressor": "Trees",
            "GradientBoostedRegressor": "Trees",
            # Neural Networks
            "BaseBackendNeuralNetwork": "Neural Networks",
        }

        # Add kernels if needed
        if all_kernels:
            self.models["GeneralizedSVR - RBF"] = GeneralizedSVR(kernel="rbf")
            self.models["GeneralizedSVR - Polynomial"] = GeneralizedSVR(kernel="poly")
            self.all_kernels = True
        else:
            self.all_kernels = False

        self.predictions = {}
        self.results = []

    def fit(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        custom_metrics=None,
        verbose=False,
    ):
        """Fits the regression models to the training data and evaluates their performance.

        Args:
            X_train: (np.ndarray) - Training feature data.
            y_train: (np.ndarray) - Training target data.
            X_test: (np.ndarray) optional - Testing feature data (default is None).
            y_test: (np.ndarray) optional - Testing target data (default is None).
            custom_metrics: (dict: str -> callable), optional: Custom metrics for evaluation (default is None).
            verbose: (bool), optional - If True, prints progress (default is False).

        Returns:
            results (list) - A list of dictionaries containing model performance metrics.
            predictions (dict) - A dictionary of predictions for each model.
        """
        # Input validation
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise TypeError("X_train and y_train must be NumPy arrays.")
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("X_train and y_train cannot be empty.")
        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must have the same number of samples."
            )
        if len(np.unique(y_train)) < 2:
            raise ValueError("y_train must contain at least two classes.")
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            raise ValueError("X_train and y_train cannot contain NaN values.")
        if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
            raise ValueError("X_train and y_train cannot contain infinite values.")

        # Initialize neural network if not already set
        if self.models["BaseBackendNeuralNetwork"] is None:
            input_size = X_train.shape[1]
            output_size = 1
            layers = [128, 64]
            dropout_rate = 0.1
            reg_lambda = 0.01
            learning_rate = 0.00001
            epochs = 1000
            batch_size = 32
            optimizer = AdamOptimizer(learning_rate=learning_rate)
            sub_scheduler = lr_scheduler_step(
                optimizer, lr_decay=0.1, lr_decay_epoch=10
            )
            scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)
            loss_func = MeanSquaredErrorLoss()
            activations = ["relu"] * len(layers) + ["none"]
            self.models["BaseBackendNeuralNetwork"] = BaseBackendNeuralNetwork(
                [input_size] + layers + [output_size],
                dropout_rate=dropout_rate,
                reg_lambda=reg_lambda,
                activations=activations,
                loss_function=loss_func,
                regressor=True,
            )

        # Initialize progress bar if TQDM is available
        progress_bar = (
            tqdm(
                self.models.items(),
                desc="Fitting Models",
                disable=not verbose or not TQDM_AVAILABLE,
            )
            if TQDM_AVAILABLE
            else self.models.items()
        )

        # For each model, fit the data and evaluate
        for name, model in progress_bar:
            if TQDM_AVAILABLE:
                progress_bar.set_description(f"Fitting {name}")
            start_time = time.time()

            # Scale the data if needed - Neural Network requires scaling
            if name == "BaseBackendNeuralNetwork":
                scaler_X = Scaler(method="standard")
                X_train_scaled = scaler_X.fit_transform(X_train)
                if X_test is not None:
                    X_test_scaled = scaler_X.transform(X_test)

                scaler_y = Scaler(method="standard")
                y_train_scaled = scaler_y.fit_transform(
                    y_train.reshape(-1, 1)
                ).flatten()
                if y_test is not None:
                    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

                # Use scaled data for training
                if X_test is not None and y_test is not None:
                    model.train(
                        X_train_scaled,
                        y_train_scaled,
                        X_test_scaled,
                        y_test_scaled,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        epochs=epochs,
                        batch_size=batch_size,
                        early_stopping_threshold=10,
                        p=False,
                        use_tqdm=False,
                    )
                else:
                    model.train(
                        X_train_scaled,
                        y_train_scaled,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        epochs=epochs,
                        batch_size=batch_size,
                        early_stopping_threshold=10,
                        p=False,
                        use_tqdm=False,
                    )
            else:
                scaler_X = scaler_y = None
                # Fit using only training data
                model.fit(X_train, y_train)

            try:
                if name == "BaseBackendNeuralNetwork" and scaler_y:
                    y_pred = scaler_y.inverse_transform(
                        model.predict(
                            X_test_scaled if X_test is not None else X_train_scaled
                        ).reshape(-1, 1)
                    ).flatten()
                else:
                    y_pred = model.predict(X_test if X_test is not None else X_train)
            except IndexError as e:
                raise ValueError(
                    f"Model '{name}' encountered an error during prediction: {e}"
                ) from None
            elapsed_time = time.time() - start_time

            # Evaluate metrics
            metrics = {}
            if custom_metrics:
                for metric_name, metric_func in custom_metrics.items():
                    metrics[metric_name] = metric_func(
                        y_test if y_test is not None else y_train, y_pred
                    )
            else:
                metrics["R-Squared"] = r_squared(
                    y_test if y_test is not None else y_train, y_pred
                )
                metrics["RMSE"] = root_mean_squared_error(
                    y_test if y_test is not None else y_train, y_pred
                )
                metrics["MAPE"] = mean_absolute_percentage_error(
                    y_test if y_test is not None else y_train, y_pred
                )

            self.predictions[name] = y_pred
            self.results.append({"Model": name, **metrics, "Time Taken": elapsed_time})

        if TQDM_AVAILABLE:
            progress_bar.set_description("Fitting Completed")
        return self.results, self.predictions

    def predict(self, X, model=None):
        """Generates predictions for the given input data using all fitted models.

        Args:
            X: (np.ndarray) - Input feature data.
            model: (str), optional - Specific model name to use for predictions (default is None).

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
        """Evaluates the performance of the fitted models using the provided true values.

        Args:
            y_true: (np.ndarray) - True target values.
            custom_metrics: (dict: str -> callable), optional - Custom metrics for evaluation (default is None).
            model: (str), optional - Specific model name to evaluate (default is None).

        Returns:
            - evaluation_results (dict): A dictionary containing evaluation metrics for the specified model(s).
        """
        evaluation_results = {}

        # Select specific model or all models
        models_to_evaluate = {model: self.models[model]} if model else self.models

        for name, _model_instance in models_to_evaluate.items():
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
        """Returns the model instance for the specified model name.

        Args:
            model_name: (str) - The name of the model.

        Returns:
            model_instance: The model instance.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        return self.models[model_name]

    def summary(self):
        """Prints a summary of the performance of all fitted models, sorted by R-squared, RMSE, and time taken.

        Dynamically adjusts to display the metrics used during evaluation.
        """
        if not self.results:
            print("No models have been fitted yet.")
            return

        # Extract all metric keys dynamically from the results
        metric_keys = [
            key
            for key in self.results[0]
            if key not in ["Model", "Model Class", "Time Taken"]
        ]

        # Sort results by R-Squared (if available), RMSE (if available), and Time Taken
        sorted_results = sorted(
            self.results,
            key=lambda x: (
                -x.get("R-Squared", float("-inf")),  # Descending R-Squared
                x.get("RMSE", float("inf")),  # Ascending RMSE
                x["Time Taken"],  # Ascending Time Taken
            ),
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
        except Exception:
            # For environments where tabulate is not available, fallback to basic printing
            # Print header dynamically based on metrics
            header = (
                "| Model Class       | Model                 | "
                + " | ".join(metric_keys)
                + " | Time Taken |"
            )
            separator = (
                "|:------------------|:----------------------|"
                + "|".join([":-----------" for _ in metric_keys])
                + "|:------------|"
            )
            print(header)
            print(separator)

            # Print each result dynamically
            for result in sorted_results:
                metrics_values = " | ".join(
                    f"{result[key]:<9.6f}" for key in metric_keys
                )
                print(
                    f"| {result['Model Class']:<16} | {result['Model']:<22} | {metrics_values} | {result['Time Taken']:<10.6f} |"
                )
