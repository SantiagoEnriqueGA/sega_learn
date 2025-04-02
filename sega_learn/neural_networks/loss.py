import numpy as np


class CrossEntropyLoss:
    """
    Custom cross entropy loss implementation using numpy for multi-class classification.
    Formula: -sum(y * log(p) + (1 - y) * log(1 - p)) / m
    Methods:
        __call__(self, logits, targets): Calculate the cross entropy loss.
    """

    def __call__(self, logits, targets):
        """
        Calculate the cross entropy loss.
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
    """
    Custom binary cross entropy loss with logits implementation using numpy.
    Formula: -mean(y * log(p) + (1 - y) * log(1 - p))
    Methods:
        __call__(self, logits, targets): Calculate the binary cross entropy loss.
    """

    def __call__(self, logits, targets):
        """
        Calculate the binary cross entropy loss.
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
