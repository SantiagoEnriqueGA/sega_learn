import os
import sys

import pandas as pd
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.neural_networks import *
from sega_learn.utils import Scaler, train_test_split


def load_pima_diabetes_data(file_path):
    """Function to load and preprocess Pima Indians Diabetes dataset"""
    df = pd.read_csv(file_path)
    X = df.drop("y", axis=1).to_numpy()
    y = df["y"].to_numpy().reshape(-1, 1)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )

    # Standardize the features
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test


def load_breast_prognostic_data(file_path):
    """Function to load and preprocess Wisconsin Breast Prognostic dataset"""
    df = pd.read_csv(file_path)
    X = df.drop("diagnosis", axis=1).to_numpy()
    y = df["diagnosis"].to_numpy().reshape(-1, 1)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )

    # Standardize the features
    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test


def hyper_train_and_evaluate_model(
    X,
    y,
    X_train,
    X_test,
    y_train,
    y_test,
    param_grid,
    layers,
    lr_range,
    optimizers,
    epochs=100,
    batch_size=32,
):
    """Function to train and evaluate the Neural Network with hyperparameter tuning"""

    input_size = X_train.shape[1]
    output_size = 1

    # Initialize Neural Network
    nn = BaseBackendNeuralNetwork(
        [input_size] + [100, 50, 25] + [output_size], dropout_rate=0.5, reg_lambda=0.0
    )
    _optimizer = AdamOptimizer(learning_rate=0.0001)

    # Hyperparameter tuning with Adam optimizer
    best_params, best_accuracy = nn.tune_hyperparameters(
        X_train,
        y_train,
        X_test,
        y_test,
        param_grid,
        layer_configs=layers,
        optimizer_types=optimizers,
        lr_range=lr_range,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Create the optimizer with the best parameters
    best_optimizer = nn._create_optimizer(
        best_params["optimizer"], best_params["learning_rate"]
    )

    # Train the final model with best parameters
    nn = BaseBackendNeuralNetwork(
        [input_size] + best_params["layers"][1:-1] + [output_size],
        dropout_rate=best_params["dropout_rate"],
        reg_lambda=best_params["reg_lambda"],
    )
    nn.train(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=best_optimizer,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def main(diabetes=True, cancer=True, test_case=True):
    # Define parameter grid and tuning ranges
    param_grid = {"dropout_rate": [0.1, 0.2, 0.3], "reg_lambda": [0.0, 0.01]}
    layers = [[100, 50, 25], [50, 25, 10], [100, 100, 50, 25]]
    lr_range = (1e-5, 0.01, 3)  # (min_lr, max_lr, num_steps)
    optimizers = ["Adam", "SGD", "Adadelta"]

    if test_case:
        param_grid = {"dropout_rate": [0.1], "reg_lambda": [0.0]}
        layers = [[25, 25, 25]]
        lr_range = (1e-5, 1e-5, 1)
        optimizers = ["Adam"]

    if diabetes:
        # Train and evaluate on Pima Indians Diabetes dataset
        print("\n--- Training on Pima Indians Diabetes Dataset ---")

        X, y, X_train, X_test, y_train, y_test = load_pima_diabetes_data(
            "example_datasets/pima-indians-diabetes_prepared.csv"
        )
        hyper_train_and_evaluate_model(
            X,
            y,
            X_train,
            X_test,
            y_train,
            y_test,
            param_grid,
            layers,
            lr_range,
            optimizers,
        )

    if cancer:
        # Train and evaluate on Wisconsin Breast Prognostic dataset
        print("\n--- Training on Wisconsin Breast Prognostic Dataset ---")

        X, y, X_train, X_test, y_train, y_test = load_breast_prognostic_data(
            "example_datasets/Wisconsin_breast_prognostic.csv"
        )
        hyper_train_and_evaluate_model(
            X,
            y,
            X_train,
            X_test,
            y_train,
            y_test,
            param_grid,
            layers,
            lr_range,
            optimizers,
        )

    if not diabetes and not cancer:
        print("Please select at least one dataset to train and evaluate.")


if __name__ == "__main__":
    main(diabetes=True, cancer=True)
