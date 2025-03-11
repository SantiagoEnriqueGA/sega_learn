import numpy as np
from numba import jit, njit, vectorize, float64, int32, prange
from numba import types
from numba.experimental import jitclass

from .numba_utils import *

spec = [
    ('weights', float64[:,::1]),            # 2D array for weights
    ('biases', float64[:,::1]),             # 2D array for biases
    ('activation', types.unicode_type),     # String for activation function
    ('weight_gradients', float64[:,::1]),   # 2D array for weight gradients
    ('bias_gradients', float64[:,::1]),     # 2D array for bias gradients
    ('input_cache', float64[:,::1]),        # 2D array for input cache
    ('output_cache', float64[:,::1]),       # 2D array for output cache
    ('input_size', int32),
    ('output_size', int32),
]
@jitclass(spec)
class JITDenseLayer:
    """
    Initializes a fully connected layer object, where each neuron is connected to all neurons in the previous layer.
    Each layer consists of weights, biases, and an activation function.
    Args:
        input_size (int): The size of the input to the layer.
        output_size (int): The size of the output from the layer.
        activation (str): The activation function to be used in the layer.
        
    Attributes:
        weights (np.ndarray): Weights of the layer.
        biases (np.ndarray): Biases of the layer.
        activation (str): Activation function name.
        weight_gradients (np.ndarray): Gradients of the weights.
        bias_gradients (np.ndarray): Gradients of the biases.
        input_cache (np.ndarray): Cached input for backpropagation.
        output_cache (np.ndarray): Cached output for backpropagation.
        
    Methods:
        zero_grad(): Resets the gradients of the weights and biases to zero.
        forward(X): Performs the forward pass of the layer.
        backward(dA, reg_lambda): Performs the backward pass of the layer.
        activate(Z): Applies the activation function.
        activation_derivative(Z): Applies the derivative of the activation function.
    """
    def __init__(self, input_size, output_size, activation="relu"):
        # He initialization for weights
        if activation in ["relu", "leaky_relu"]:
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)
            
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.weight_gradients = np.zeros((input_size, output_size)) # Initialize weight gradients to zeros
        self.bias_gradients = np.zeros((1, output_size))            # Initialize bias gradients to zeros
        self.input_cache = np.zeros((1, input_size))        
        self.output_cache = np.zeros((1, output_size))
        self.input_size = input_size
        self.output_size = output_size
        
    def zero_grad(self):
        """Reset the gradients of the weights and biases to zero."""
        self.weight_gradients = np.zeros_like(self.weight_gradients)
        self.bias_gradients = np.zeros_like(self.bias_gradients)
        
    def forward(self, X):
        Z = np.dot(X, self.weights) + self.biases
        self.input_cache = X
        self.output_cache = self.activate(Z)
        return self.output_cache

    def backward(self, dA, reg_lambda):
        m = self.input_cache.shape[0]
        dZ = dA * self.activation_derivative(self.output_cache)
        dW = np.dot(self.input_cache.T, dZ) / m + reg_lambda * self.weights
        db = sum_axis0(dZ) / m
        dA_prev = np.dot(dZ, self.weights.T)
        
        self.weight_gradients = dW
        self.bias_gradients = db
        
        return dA_prev

    def activate(self, Z):
        """Apply activation function."""
        if self.activation == "relu":
            return relu(Z)
        elif self.activation == "leaky_relu":
            return leaky_relu(Z)
        elif self.activation == "tanh":
            return tanh(Z)
        elif self.activation == "sigmoid":
            return sigmoid(Z)
        elif self.activation == "softmax":
            return softmax(Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
    def activation_derivative(self, Z):
        """Apply activation derivative."""
        if self.activation == "relu":
            return relu_derivative(Z)
        elif self.activation == "leaky_relu":
            return leaky_relu_derivative(Z)
        elif self.activation == "tanh":
            return tanh_derivative(Z)
        elif self.activation == "sigmoid":
            return sigmoid_derivative(Z)
        elif self.activation == "softmax":
            return np.ones_like(Z)  # Identity for compatibility
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

