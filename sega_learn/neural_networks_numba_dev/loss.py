import numpy as np
from numba import njit, prange

CACHE = False

@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def max_reduce(arr):
    max_vals = np.empty((arr.shape[0], 1), dtype=arr.dtype)
    for i in prange(arr.shape[0]):
        max_vals[i, 0] = np.max(arr[i])
    return max_vals

@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def sum_reduce(arr):
    sum_vals = np.empty((arr.shape[0], 1), dtype=arr.dtype)
    for i in prange(arr.shape[0]):
        sum_vals[i, 0] = np.sum(arr[i])
    return sum_vals

@njit(fastmath=True, nogil=True)
def calculate_cross_entropy_loss(logits, targets):
    n = logits.shape[0]
    loss = 0.0
    for i in prange(n):
        max_val = np.max(logits[i])
        exp_sum = 0.0
        for j in range(logits.shape[1]):
            exp_val = np.exp(logits[i, j] - max_val)
            exp_sum += exp_val
        for j in range(logits.shape[1]):
            prob = np.exp(logits[i, j] - max_val) / exp_sum
            loss -= targets[i, j] * np.log(prob + 1e-15)
    return loss / n

@njit(fastmath=True, nogil=True, cache=CACHE)
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