import numpy as np
from numba import jit, njit, vectorize, float64, int32, prange
from numba import types
from numba.experimental import jitclass

from .numba_utils import *

# Define the unified layer specification, including a field for flatten layers.
layer_spec = [
    ('layer_type', types.unicode_type),          # "dense", "conv", or "flatten"
    ('input_size', int32),                       # For dense: # of inputs; for conv: # channels; for flatten: dummy
    ('output_size', int32),                      # For dense: # of neurons; for conv: # filters; for flatten: flattened size
    ('activation', types.unicode_type),          # Activation function (used for dense and conv; ignored for flatten)
    # Dense-specific fields:
    ('dense_weights', float64[:, ::1]),
    ('dense_biases', float64[:, ::1]),
    ('dense_weight_grad', float64[:, ::1]),  # Gradient of weights
    ('dense_bias_grad', float64[:, ::1]),    # Gradient of biases
    ('dense_input_cache', float64[:, ::1]),  # Cache for input during forward pass
    ('dense_output_cache', float64[:, ::1]), # Cache for output during forward pass
    # Convolution-specific fields:
    ('conv_weights', float64[:,:,:,:]),
    ('conv_biases', float64[:, ::1]),
    ('kernel_size', int32),
    ('stride', int32),
    ('padding', int32),
    # Flatten-specific field:
    ('flatten_input_shape', types.UniTuple(int32, 3)),
]
@jitclass(layer_spec)
class JITLayer:
    def __init__(self, layer_type, input_size, output_size, activation="relu",
                 kernel_size=0, stride=1, padding=0, flatten_input_shape=(0, 0, 0)):
        self.layer_type = layer_type
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        if layer_type == "dense":
            # For dense layers, use He initialization (for relu-like activations)
            if activation in ["relu", "leaky_relu"]:
                scale = np.sqrt(2.0 / input_size)
            else:
                scale = np.sqrt(1.0 / input_size)
            self.dense_weights = np.random.randn(input_size, output_size) * scale
            self.dense_biases = np.zeros((1, output_size))
            self.dense_weight_grad = np.zeros_like(self.dense_weights)
            self.dense_bias_grad = np.zeros_like(self.dense_biases)
            # Cache for input and output during forward pass
            self.dense_input_cache = np.zeros((1, input_size))
            self.dense_output_cache = np.zeros((1, output_size))
            
            # Dummy initialization for conv/flatten fields
            self.conv_weights = np.zeros((0, 0, 0, 0))
            self.conv_biases = np.zeros((0, 0))
            self.kernel_size = 0
            self.stride = 1
            self.padding = 0
            self.flatten_input_shape = (0, 0, 0)
        
        elif layer_type == "conv":
            # For convolutional layers, input_size is number of input channels.
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            scale = np.sqrt(2.0 / (input_size * kernel_size * kernel_size))
            self.conv_weights = np.random.randn(output_size, input_size, kernel_size, kernel_size) * scale
            self.conv_biases = np.zeros((output_size, 1))
            # Dummy initialization for dense/flatten fields
            self.dense_weights = np.zeros((0, 0))
            self.dense_biases = np.zeros((0, 0))
            self.flatten_input_shape = (0, 0, 0)
        elif layer_type == "flatten":
            # Flatten layers do not have weights. Store the input shape on forward pass.
            self.flatten_input_shape = flatten_input_shape  # Will be updated in forward
            self.dense_weights = np.zeros((0, 0))
            self.dense_biases = np.zeros((0, 0))
            self.conv_weights = np.zeros((0, 0, 0, 0))
            self.conv_biases = np.zeros((0, 0))
            self.kernel_size = 0
            self.stride = 1
            self.padding = 0
        else:
            raise ValueError("Unsupported layer type")
    
    def zero_grad(self):
        """Zero out gradients for the layer."""
        if self.layer_type == "dense":
            self.dense_weight_grad = np.zeros_like(self.dense_weights)
            self.dense_bias_grad = np.zeros_like(self.dense_biases)
        elif self.layer_type == "conv":
            self.conv_weights = np.zeros_like(self.conv_weights)
            self.conv_biases = np.zeros_like(self.conv_biases)
        elif self.layer_type == "flatten":
            pass
    
    def forward(self, X):
        if self.layer_type == "dense":
            return self._forward_dense(X)
        elif self.layer_type == "conv":
            pass
        elif self.layer_type == "flatten":
            pass
        else:
            raise ValueError("Unsupported layer type")
    
    def backward(self, dA, reg_lambda):
        if self.layer_type == "dense":
            return self._backward_dense(dA, reg_lambda)
        elif self.layer_type == "conv":
            pass
        elif self.layer_type == "flatten":
            pass
        else:
            raise ValueError("Unsupported layer type")
        
    def _forward_dense(self, X):
        """Forward pass for dense layer."""
        Z = np.dot(X, self.dense_weights) + self.dense_biases
        self.dense_input_cache = X
        self.dense_output_cache = self.activate(Z)
        return self.dense_output_cache
    
    def _backward_dense(self, dA, reg_lambda):
        """Backward pass for dense layer."""
        m = self.dense_input_cache.shape[0]
        dZ = dA * self.activation_derivative(self.dense_output_cache)
        dW = np.dot(self.dense_input_cache.T, dZ) / m + reg_lambda * self.dense_weights
        db = sum_axis0(dZ) / m
        dA_prev = np.dot(dZ, self.dense_weights.T)
        
        self.dense_weight_grad = dW
        self.dense_bias_grad = db
        
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