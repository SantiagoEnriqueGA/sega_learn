import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _forward_jit_impl(X, weights, biases, activations, dropout_rate, training, is_binary):
    # Store all layer activations for backprop
    layer_outputs = [X]
    
    # Forward pass through all layers except the last
    for i in range(len(weights) - 1):
        # Calculate linear transformation
        Z = np.dot(X, weights[i]) + biases[i]
        
        # Apply activation function
        if activations[i] == "relu":
            X = relu(Z)
        elif activations[i] == "leaky_relu":
            X = leaky_relu(Z)
        elif activations[i] == "tanh":
            X = tanh(Z)
        elif activations[i] == "sigmoid":
            X = sigmoid(Z)
        elif activations[i] == "softmax":
            X = softmax(Z)
        else:
            raise ValueError(f"Unsupported activation: {activations[i]}")
        
        # Apply dropout only during training
        if training and dropout_rate > 0:
            X = np.multiply(X, np.random.rand(*X.shape) < (1 - dropout_rate)) / (1 - dropout_rate)
        
        layer_outputs.append(X)
    
    # Last layer (output layer)
    Z = np.dot(X, weights[-1]) + biases[-1]
    if is_binary:
        output = sigmoid(Z)
    else:
        output = softmax(Z)
        
    layer_outputs.append(output)
    return layer_outputs

@njit(fastmath=True, cache=True)
def _backward_jit_impl(layer_outputs, y, weights, activations, reg_lambda, is_binary):
    m = y.shape[0]  # Number of samples
    dWs = []
    dbs = []

    # Reshape y for binary classification
    if is_binary:
        y = y.reshape(-1, 1).astype(np.float64)
    else:
        y = np.eye(layer_outputs[-1].shape[1])[y].astype(np.float64)

    # Calculate initial gradient based on loss function
    outputs = layer_outputs[-1]
    if is_binary:
        dA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15))
    else:
        dA = outputs - y

    # Backpropagate through layers in reverse
    for i in range(len(weights) - 1, -1, -1):
        prev_activation = layer_outputs[i]

        if i < len(weights) - 1:
            if activations[i] == "relu":
                dZ = dA * relu_derivative(layer_outputs[i + 1])
            elif activations[i] == "leaky_relu":
                dZ = dA * leaky_relu_derivative(layer_outputs[i + 1])
            elif activations[i] == "tanh":
                dZ = dA * tanh_derivative(layer_outputs[i + 1])
            elif activations[i] == "sigmoid":
                dZ = dA * sigmoid_derivative(layer_outputs[i + 1])
            elif activations[i] == "softmax":
                dZ = dA  # Softmax derivative is handled in the loss gradient
            else:
                raise ValueError(f"Unsupported activation: {activations[i]}")
        else:
            dZ = dA

        dW = np.dot(prev_activation.T, dZ) / m
        dW += reg_lambda * weights[i]  # Add L2 regularization
        db = sum_reduce(dZ) / m

        dWs.insert(0, dW)
        dbs.insert(0, db)

        if i > 0:
            dA = np.dot(dZ, weights[i].T)

    return dWs, dbs

# Activation functions and their derivatives
@njit(fastmath=True, cache=True)
def relu(z):
    return np.maximum(0, z)

@njit(fastmath=True, cache=True)
def relu_derivative(z):
    return (z > 0).astype(np.float64)  # Ensure return type is float64

@njit(fastmath=True, cache=True)
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

@njit(fastmath=True, cache=True)
def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha).astype(np.float64)  # Ensure return type is float64

@njit(fastmath=True, cache=True)
def tanh(z):
    return np.tanh(z)

@njit(fastmath=True, cache=True)
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

@njit(fastmath=True, cache=True)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@njit(fastmath=True, cache=True)
def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

@njit(fastmath=True, cache=True)
def softmax(z):
    max_z = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        max_z[i] = np.max(z[i])
    exp_z = np.empty_like(z)
    for i in range(z.shape[0]):
        exp_z[i] = np.exp(z[i] - max_z[i])
    sum_exp_z = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        sum_exp_z[i] = np.sum(exp_z[i])
    for i in range(z.shape[0]):
        exp_z[i] /= sum_exp_z[i]
    return exp_z

@njit(fastmath=True, cache=True)
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
