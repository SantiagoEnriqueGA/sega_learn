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
    output_activation="none",
    epochs=100,
    batch_size=32,
    loss_func=None,
    y_scaler=None,
):
    """Function to train and evaluate the Neural Network."""
    input_size = X_train.shape[1]

    # The output_activation parameter should be 'none' or 'linear' for regression
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
        track_adv_metrics=True,  # Not used for regression
    )

    # Evaluate the Model (evaluate returns primary metric (loss/MSE) and predictions)
    test_loss, y_pred_scaled = nn.evaluate(X_test, y_test)  # y_pred is scaled
    print(f"Test Loss (Scaled MSE): {test_loss:.4f}")

    # Inverse transform for interpretable metrics
    if y_scaler is None:
        print("Warning: y_scaler not provided, cannot inverse transform metrics.")
        y_test_orig = y_test  # Keep scaled if no scaler
        y_pred_orig = y_pred_scaled
    else:
        y_test_orig = y_scaler.inverse_transform(y_test)
        y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)

    # Calculate metrics on original scale
    r2_score = r2(y_test_orig, y_pred_orig)
    mse_score = mse(y_test_orig, y_pred_orig)
    mae_score = mae(y_test_orig, y_pred_orig)
    print(f"R^2 Score (Original Scale): {r2_score:.4f}")
    print(f"Mean Squared Error (Original Scale): {mse_score:.4f}")
    print(f"Mean Absolute Error (Original Scale): {mae_score:.4f}")

    # Plot metrics (will show scaled loss/metric from training history)
    # nn.plot_metrics()
    nn.plot_metrics(
        save_dir="examples/neural_networks/plots/neuralNetwork_regressor_vanilla_metrics.png"
    )


def train_and_evaluate_model_numba(
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
    epochs=100,
    batch_size=32,
    loss_func=None,
    y_scaler=None,
):
    """Function to train and evaluate the Neural Network."""
    input_size = X_train.shape[1]

    # The output_activation parameter should be 'none' or 'linear' for regression
    activations = [hidden_activation] * len(layers) + [output_activation]

    # Initialize Neural Network
    nn = NumbaBackendNeuralNetwork(
        [input_size] + layers + [output_size],
        dropout_rate=dropout,
        reg_lambda=reg_lambda,
        activations=activations,
        loss_function=loss_func,
        regressor=True,
        compile_numba=True,
        # compile_numba=False,
        progress_bar=True,
    )

    # Select optimizer
    optimizer = JITAdamOptimizer(learning_rate=lr)
    # optimizer = JITSGDOptimizer(learning_rate=lr, momentum=0.25, reg_lambda=0.1)
    # optimizer = JITAdadeltaOptimizer(learning_rate=lr, rho=0.95, epsilon=1e-6, reg_lambda=0.0)

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
        track_adv_metrics=False,  # Not used for regression
        use_tqdm=True,  # Use tqdm progress bar if available
    )

    # Evaluate the Model (evaluate returns primary metric (loss/MSE) and predictions)
    test_loss, y_pred_scaled = nn.evaluate(X_test, y_test)  # y_pred is scaled
    print(f"Test Loss (Scaled MSE): {test_loss:.4f}")

    # Inverse transform for interpretable metrics
    if y_scaler is None:
        print("Warning: y_scaler not provided, cannot inverse transform metrics.")
        y_test_orig = y_test  # Keep scaled if no scaler
        y_pred_orig = y_pred_scaled
    else:
        y_test_orig = y_scaler.inverse_transform(y_test)
        y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)

    # Calculate metrics on original scale
    r2_score = r2(y_test_orig, y_pred_orig)
    mse_score = mse(y_test_orig, y_pred_orig)
    mae_score = mae(y_test_orig, y_pred_orig)
    print(f"R^2 Score (Original Scale): {r2_score:.4f}")
    print(f"Mean Squared Error (Original Scale): {mse_score:.4f}")
    print(f"Mean Absolute Error (Original Scale): {mae_score:.4f}")

    # Plot metrics (will show scaled loss/metric from training history)
    # nn.plot_metrics()
    # nn.plot_metrics(
    #     save_dir="examples/neural_networks/plots/neuralNetwork_regressor_numba_metrics.png"
    # )


def main():
    """Main function to train and evaluate the Neural Network."""
    import random

    np.random.seed(1)
    random.seed(1)

    # Define parameter grid and tuning ranges
    dropout = 0.1
    reg_lambda = 0.01
    lr = 0.00001
    layers = [128, 64]
    output_size = 1
    loss_func = MeanSquaredErrorLoss()
    epochs = 1000

    X, y = make_regression(n_samples=1_000, n_features=5, noise=0.5, random_state=42)

    # Scale the features and target variable
    X_scaler = Scaler(method="standard")
    y_scaler = Scaler(method="standard")

    X = X_scaler.fit_transform(X)
    # Reshape y before scaling if it's 1D
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    y = y_scaler.fit_transform(y)

    print(f"X mean/std: {X.mean():.2f}/{X.std():.2f}")  # Verify scaling
    print(f"y mean/std: {y.mean():.2f}/{y.std():.2f}")  # Verify scaling

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
        epochs=epochs,
        batch_size=32,
        loss_func=loss_func,
        y_scaler=y_scaler,
    )

    # To use the Numba backend:
    # lr = 0.0001
    # loss_func = JITMeanSquaredErrorLoss()
    # train_and_evaluate_model_numba(
    #     X_train,
    #     X_test,
    #     y_train,
    #     y_test,
    #     layers,
    #     output_size,
    #     lr,
    #     dropout,
    #     reg_lambda,
    #     hidden_activation="relu",
    #     output_activation="none",
    #     epochs=epochs * 2,
    #     batch_size=32,
    #     loss_func=loss_func,
    #     y_scaler=y_scaler,
    # )


if __name__ == "__main__":
    main()
