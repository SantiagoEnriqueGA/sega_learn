import numpy as np
from numba import jit, njit, vectorize, float64, int32, prange
from numba import types
from numba.experimental import jitclass

from .numba_utils import *

spec = [
    ('weights', float64[:,:]),            # 2D array for weights
    ('biases', float64[:,:]),             # 2D array for biases
    ('activation', types.unicode_type),   # String for activation function
    ('weight_gradients', float64[:,:]),   # 2D array for weight gradients
    ('bias_gradients', float64[:,:]),     # 2D array for bias gradients
    ('input_size', int32),
    ('output_size', int32),
]
@jitclass(spec)
class JITLayer:
    """
    Initializes a Layer object.
    Args:
        input_size (int): The size of the input to the layer.
        output_size (int): The size of the output from the layer.
        activation (str): The activation function to be used in the layer.
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
        self.weight_gradients = np.zeros((input_size, output_size))  # Initialize weight gradients to zeros
        self.bias_gradients = np.zeros((1, output_size))  # Initialize bias gradients to zeros
        
    def zero_grad(self):
        """Reset the gradients of the weights and biases to zero."""
        self.weight_gradients = np.zeros_like(self.weight_gradients)
        self.bias_gradients = np.zeros_like(self.bias_gradients)

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

