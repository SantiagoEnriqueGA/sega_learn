import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.neural_networks import *
from sega_learn.utils import Metrics, Scaler, make_regression, train_test_split

r2 = Metrics.r_squared
mse = Metrics.mean_squared_error
mae = Metrics.mean_absolute_error


def train_and_evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test,
    layers,
    output_size,
    lr,
    dropout,
    reg_lambda,
    hidden_activation="relu",
    output_activation="softmax",
    epochs=100,
    batch_size=32,
    loss_func=None,
):
    """Function to train and evaluate the Neural Network."""
    input_size = X_train.shape[1]

    activations = [hidden_activation] * len(layers) + [output_activation]

    # Initialize Neural Network
    nn = BaseBackendNeuralNetwork(
        [input_size] + layers + [output_size],
        dropout_rate=dropout,
        reg_lambda=reg_lambda,
        activations=activations,
        loss_function=loss_func,
        regressor=True,
    )

    # Select optimizer
    optimizer = AdamOptimizer(learning_rate=lr)
    # optimizer = SGDOptimizer(learning_rate=lr, momentum=0.25, reg_lambda=0.1)
    # optimizer = AdadeltaOptimizer(learning_rate=lr, rho=0.95, epsilon=1e-6, reg_lambda=0.0)

    # Select learning rate scheduler
    sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)
    scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)
    # scheduler = lr_scheduler_exp(optimizer, lr_decay=0.1, lr_decay_epoch=10)
    # scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)

    # Call the train method
    nn.train(
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_threshold=10,
        track_metrics=True,
        track_adv_metrics=True,
    )

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate metrics
    r2_score = r2(y_test, y_pred)
    mse_score = mse(y_test, y_pred)
    mae_score = mae(y_test, y_pred)
    print(f"R^2 Score: {r2_score:.4f}")
    print(f"Mean Squared Error: {mse_score:.4f}")
    print(f"Mean Absolute Error: {mae_score:.4f}")

    # Plot metrics
    # nn.plot_metrics()
    # nn.plot_metrics(save_dir="examples/neural_networks/plots/neuralNetwork_classifier_vanilla_metrics.png")


def main():
    """Main function to train and evaluate the Neural Network."""
    import random

    np.random.seed(1)
    random.seed(1)

    # Define parameter grid and tuning ranges
    dropout = 0.1
    reg_lambda = 0.1
    lr = 0.01
    layers = [500, 250, 50]
    output_size = 1
    loss_func = MeanSquaredErrorLoss()

    X, y = make_regression(n_samples=10_000, n_features=5, noise=0.1, random_state=42)

    # Scale the features to the range [0, 1]
    X_scaler = Scaler(method="minmax")
    y_scaler = Scaler(method="minmax")

    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    print(f"X min/max: {X.min()}/{X.max()}")
    print(f"y min/max: {y.min()}/{y.max()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_and_evaluate_model(
        X_train,
        X_test,
        y_train,
        y_test,
        layers,
        output_size,
        lr,
        dropout,
        reg_lambda,
        hidden_activation="relu",
        output_activation="none",
        epochs=1000,
        batch_size=32,
        loss_func=loss_func,
    )

    # To use the Numba backend:
    # train_and_evaluate_model_numba(X_train, X_test, y_train, y_test,
    #                               layers, output_size, lr, dropout, reg_lambda,
    #                               hidden_activation='relu', output_activation=None,
    #                               epochs=1000, batch_size=32)


if __name__ == "__main__":
    main()
