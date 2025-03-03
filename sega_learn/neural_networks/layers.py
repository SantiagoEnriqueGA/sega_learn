import numpy as np
from .activations import Activation

class Layer:
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
        self.gradients = None
        
    def zero_grad(self):
        """Reset the gradients of the weights and biases to zero."""
        self.gradients = None

    def activate(self, Z):
        """Apply activation function."""
        activation_functions = {
            "relu": Activation.relu,
            "leaky_relu": Activation.leaky_relu,
            "tanh": Activation.tanh,
            "sigmoid": Activation.sigmoid,
            "softmax": Activation.softmax
        }
        
        if self.activation in activation_functions:
            return activation_functions[self.activation](Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def activation_derivative(self, Z):
        """Apply activation derivative."""
        if self.activation == "relu":
            return Activation.relu_derivative(Z)
        elif self.activation == "leaky_relu":
            return Activation.leaky_relu_derivative(Z)
        elif self.activation == "tanh":
            return Activation.tanh_derivative(Z)
        elif self.activation == "sigmoid":
            return Activation.sigmoid_derivative(Z)
        elif self.activation == "softmax":
            # Softmax derivative handled in loss function
            return np.ones_like(Z)  # Identity for compatibility
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
