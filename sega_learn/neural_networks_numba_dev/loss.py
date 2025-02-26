import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True, nogil=True)
def max_reduce(arr):
    max_vals = np.empty((arr.shape[0], 1), dtype=arr.dtype)
    for i in prange(arr.shape[0]):
        max_vals[i, 0] = np.max(arr[i])
    return max_vals

@njit(parallel=True, fastmath=True, nogil=True)
def sum_reduce(arr):
    sum_vals = np.empty((arr.shape[0], 1), dtype=arr.dtype)
    for i in prange(arr.shape[0]):
        sum_vals[i, 0] = np.sum(arr[i])
    return sum_vals

@njit(fastmath=True, nogil=True)
def calculate_cross_entropy_loss(logits, targets):
    exp_logits = np.exp(logits - max_reduce(logits))  # Exponential of logits, subtract max to prevent overflow
    probs = exp_logits / sum_reduce(exp_logits)       # Probabilities from logits
    loss = -np.sum(targets * np.log(probs + 1e-15)) / logits.shape[0]  # Cross-entropy loss
    return loss

@njit(fastmath=True, nogil=True)
def calculate_bce_with_logits_loss(logits, targets):
    probs = 1 / (1 + np.exp(-logits))  # Apply sigmoid to logits to get probabilities
    loss = -np.mean(targets * np.log(probs + 1e-15) + (1 - targets) * np.log(1 - probs + 1e-15))  # Binary cross-entropy loss
    return loss

class CrossEntropyLoss:
    def __init__(self):
        self.logits = np.zeros((1, 1))
        self.targets = np.zeros((1, 1))

    def calculate_loss(self, logits, targets):
        return calculate_cross_entropy_loss(logits, targets)

class BCEWithLogitsLoss:
    def __init__(self):
        self.logits = np.zeros((1, 1))
        self.targets = np.zeros((1, 1))

    def calculate_loss(self, logits, targets):
        return calculate_bce_with_logits_loss(logits, targets)