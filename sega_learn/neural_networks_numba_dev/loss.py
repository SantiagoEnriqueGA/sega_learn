import numpy as np
from numba import jit, njit, vectorize, float64, int32, prange
from numba import types
from numba.experimental import jitclass

spec_cross_entropy = [
    ('logits', float64[:, ::1]),
    ('targets', float64[:, ::1]),
]

@jitclass(spec_cross_entropy)
class CrossEntropyLoss:
    """
    Custom cross entropy loss implementation using numpy for multi-class classification.
    Formula: -sum(y * log(p) + (1 - y) * log(1 - p)) / m
    Methods:
        calculate_loss(self, logits, targets): Calculate the cross entropy loss.
    """
    def __init__(self):
        self.logits = np.zeros((1, 1))
        self.targets = np.zeros((1, 1))

    def calculate_loss(self, logits, targets):
        """
        Calculate the cross entropy loss.
        Args:
            logits (np.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
            targets (np.ndarray): The target labels of shape (num_samples, num_classes).
        Returns:
            float: The cross entropy loss.
        """
        exp_logits = np.exp(logits - max_reduce(logits)) # Exponential of logits, subtract max to prevent overflow
        probs = exp_logits / sum_reduce(exp_logits)      # Probabilities from logits
        loss = -np.sum(targets * np.log(probs + 1e-15)) / logits.shape[0]   # Cross-entropy loss
        
        return loss

spec_bce_with_logits = [
    ('logits', float64[:, ::1]),
    ('targets', float64[:, ::1]),
]

@jitclass(spec_bce_with_logits)
class BCEWithLogitsLoss:
    """
    Custom binary cross entropy loss with logits implementation using numpy.
    Formula: -mean(y * log(p) + (1 - y) * log(1 - p))
    Methods:
        calculate_loss(self, logits, targets): Calculate the binary cross entropy loss.
    """
    def __init__(self):
        self.logits = np.zeros((1, 1))
        self.targets = np.zeros((1, 1))

    def calculate_loss(self, logits, targets):
        """
        Calculate the binary cross entropy loss.
        Args:
            logits (np.ndarray): The logits (predicted values) of shape (num_samples,).
            targets (np.ndarray): The target labels of shape (num_samples,).
        Returns:
            float: The binary cross entropy loss.
        """
        probs = 1 / (1 + np.exp(-logits))                                                               # Apply sigmoid to logits to get probabilities
        loss = -np.mean(targets * np.log(probs + 1e-15) + (1 - targets) * np.log(1 - probs + 1e-15))    # Binary cross-entropy loss
        
        return loss

@njit
def max_reduce(arr):
    """
    Numba-compatible function to compute the maximum along axis 1 with keepdims=True.
    Args:
        arr (np.ndarray): Input array of shape (num_samples, num_classes).
    Returns:
        np.ndarray: Maximum values along axis 1 with keepdims=True.
    """
    max_vals = np.empty((arr.shape[0], 1), dtype=arr.dtype)
    for i in range(arr.shape[0]):
        max_vals[i, 0] = np.max(arr[i])
    return max_vals

@njit
def sum_reduce(arr):
    """
    Numba-compatible function to compute the sum along axis 1 with keepdims=True.
    Args:
        arr (np.ndarray): Input array of shape (num_samples, num_classes).
    Returns:
        np.ndarray: Sum values along axis 1 with keepdims=True.
    """
    sum_vals = np.empty((arr.shape[0], 1), dtype=arr.dtype)
    for i in range(arr.shape[0]):
        sum_vals[i, 0] = np.sum(arr[i])
    return sum_vals
