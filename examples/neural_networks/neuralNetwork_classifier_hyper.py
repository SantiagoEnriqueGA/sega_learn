import os
import sys

from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sega_learn.neural_networks import *
from sega_learn.utils import make_classification, train_test_split


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
        [input_size] + layers[0] + [output_size], dropout_rate=0.5, reg_lambda=0.0
    )
    optimizer = AdamOptimizer(learning_rate=0.0001)

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
        activations=["tanh"] * (len(best_params["layers"]) - 1) + ["softmax"],
    )

    nn.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=best_optimizer,
    )

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def hyper_train_and_evaluate_model_numba(
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
    nn = NumbaBackendNeuralNetwork(
        [input_size] + layers[0] + [output_size],
        dropout_rate=0.5,
        reg_lambda=0.1,
        compile_numba=True,
        progress_bar=True,
    )
    optimizer = JITAdamOptimizer(learning_rate=0.0001)

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
    nn = NumbaBackendNeuralNetwork(
        [input_size] + best_params["layers"][1:-1] + [output_size],
        dropout_rate=best_params["dropout_rate"],
        reg_lambda=best_params["reg_lambda"],
        activations=["tanh"] * (len(best_params["layers"]) - 1) + ["softmax"],
        compile_numba=False,
    )

    nn.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=best_optimizer,
    )

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def main(test_case=True):
    # Define parameter grid and tuning ranges
    param_grid = {
        "dropout_rate": [0, 0.1, 0.2, 0.3],
        "reg_lambda": [0.0, 0.01, 0.1, 1.0],
    }
    layers = [[100, 50, 25], [50, 25, 10], [100, 100, 50, 25]]
    lr_range = (1e-6, 0.01, 5)  # (min_lr, max_lr, num_steps)
    optimizers = ["Adam", "SGD", "Adadelta"]

    if test_case:
        param_grid = {"dropout_rate": [0.1], "reg_lambda": [0.0]}
        layers = [[100, 50, 25]]
        lr_range = (1e-5, 1e-5, 1)
        optimizers = ["SGD", "Adam"]

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42,
        class_sep=0.5,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
        epochs=100,
        batch_size=32,
    )

    # To use the Numba backend:
    # hyper_train_and_evaluate_model_numba(X, y, X_train, X_test, y_train, y_test,
    #                                       param_grid, layers,
    #                                       lr_range, optimizers,
    #                                       epochs=100, batch_size=32)


if __name__ == "__main__":
    main()
