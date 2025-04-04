import numpy as np


# Classifications loss functions
class CrossEntropyLoss:
    """Custom cross entropy loss implementation using numpy for multi-class classification.

    Formula: -sum(y * log(p) + (1 - y) * log(1 - p)) / m
    Methods:
        __call__(self, logits, targets): Calculate the cross entropy loss.
    """

    def __call__(self, logits, targets):
        """Calculate the cross entropy loss.

        Args:
            logits (np.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
            targets (np.ndarray): The target labels of shape (num_samples,).

        Returns:
            float: The cross entropy loss.
        """
        # One-hot encode targets if they are not already
        if targets.ndim == 1:
            targets = np.eye(logits.shape[1])[targets]

        exp_logits = np.exp(
            logits - np.max(logits, axis=1, keepdims=True)
        )  # Exponential of logits, subtract max to prevent overflow
        probs = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )  # Probabilities from logits
        loss = (
            -np.sum(targets * np.log(probs + 1e-15)) / logits.shape[0]
        )  # Cross-entropy loss

        return loss


class BCEWithLogitsLoss:
    """Custom binary cross entropy loss with logits implementation using numpy.

    Formula: -mean(y * log(p) + (1 - y) * log(1 - p))

    Methods:
        __call__(self, logits, targets): Calculate the binary cross entropy loss.
    """

    def __call__(self, logits, targets):
        """Calculate the binary cross entropy loss.

        Args:
            logits (np.ndarray): The logits (predicted values) of shape (num_samples,).
            targets (np.ndarray): The target labels of shape (num_samples,).

        Returns:
            float: The binary cross entropy loss.
        """
        probs = 1 / (
            1 + np.exp(-logits)
        )  # Apply sigmoid to logits to get probabilities
        loss = -np.mean(
            targets * np.log(probs + 1e-15) + (1 - targets) * np.log(1 - probs + 1e-15)
        )  # Binary cross-entropy loss

        return loss


# Regression loss functions
class MeanSquaredErrorLoss:
    """Custom mean squared error loss implementation using numpy.

    Formula: mean((y_true - y_pred) ** 2)

    Methods:
        __call__(self, y_true, y_pred): Calculate the mean squared error loss.
    """

    def __call__(self, y_true, y_pred):
        """Calculate the mean squared error loss.

        Args:
            y_true (np.ndarray): The true labels of shape (num_samples,).
            y_pred (np.ndarray): The predicted values of shape (num_samples,).

        Returns:
            float: The mean squared error loss.
        """
        loss = np.mean((y_true - y_pred) ** 2)  # Mean squared error loss

        return loss


class MeanAbsoluteErrorLoss:
    """Custom mean absolute error loss implementation using numpy.

    Formula: mean(abs(y_true - y_pred))

    Methods:
        __call__(self, y_true, y_pred): Calculate the mean absolute error loss.
    """

    def __call__(self, y_true, y_pred):
        """Calculate the mean absolute error loss.

        Args:
            y_true (np.ndarray): The true labels of shape (num_samples,).
            y_pred (np.ndarray): The predicted values of shape (num_samples,).

        Returns:
            float: The mean absolute error loss.
        """
        loss = np.mean(np.abs(y_true - y_pred))  # Mean absolute error loss

        return loss


class HuberLoss:
    """Custom Huber loss implementation using numpy.

    Formula: mean(delta * (abs(y_true - y_pred) - delta / 2)) if abs(y_true - y_pred) <= delta else mean(abs(y_true - y_pred))

    Methods:
        __call__(self, y_true, y_pred, delta=1.0): Calculate the Huber loss.
    """

    def __call__(self, y_true, y_pred, delta=1.0):
        """Calculate the Huber loss.

        Args:
            y_true (np.ndarray): The true labels of shape (num_samples,).
            y_pred (np.ndarray): The predicted values of shape (num_samples,).
            delta (float): The threshold for the Huber loss.

        Returns:
            float: The Huber loss.
        """
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        loss = np.where(is_small_error, squared_loss, linear_loss)

        return np.mean(loss)  # Mean Huber loss
