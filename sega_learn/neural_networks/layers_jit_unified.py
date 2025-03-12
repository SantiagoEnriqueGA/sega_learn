from numba import jitclass, float64, int32, types
import numpy as np

# Define the unified layer specification, including a field for flatten layers.
layer_spec = [
    ('layer_type', types.unicode_type),          # "dense", "conv", or "flatten"
    ('input_size', int32),                       # For dense: # of inputs; for conv: # channels; for flatten: dummy
    ('output_size', int32),                      # For dense: # of neurons; for conv: # filters; for flatten: flattened size
    ('activation', types.unicode_type),          # Activation function (used for dense and conv; ignored for flatten)
    # Dense-specific fields:
    ('dense_weights', float64[:, ::1]),
    ('dense_biases', float64[:, ::1]),
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
            scale = np.sqrt(2.0 / input_size)
            self.dense_weights = np.random.randn(input_size, output_size) * scale
            self.dense_biases = np.zeros((1, output_size))
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
    
    def forward(self, X):
        if self.layer_type == "dense":
            # Simple dense layer: Z = X.dot(W) + b, with ReLU as example
            Z = np.dot(X, self.dense_weights) + self.dense_biases
            if self.activation == "relu":
                return np.maximum(0, Z)
            else:
                return Z  # Extend with other activations as needed.
        elif self.layer_type == "conv":
            # Pseudocode for convolution forward pass.
            # Here you would compute output dimensions, apply padding, use im2col, etc.
            # For brevity, assume a helper function `conv_forward(self, X)` exists.
            return conv_forward(self, X)
        elif self.layer_type == "flatten":
            # Store the shape (excluding the batch dimension) for use in backward
            batch_size = X.shape[0]
            self.flatten_input_shape = (X.shape[1], X.shape[2], X.shape[3])
            flat_size = X.shape[1] * X.shape[2] * X.shape[3]
            self.output_size = flat_size  # Update output_size dynamically.
            # Manually flatten to ensure contiguity.
            out = np.empty((batch_size, flat_size), dtype=np.float64)
            for b in range(batch_size):
                idx = 0
                for c in range(X.shape[1]):
                    for h in range(X.shape[2]):
                        for w in range(X.shape[3]):
                            out[b, idx] = X[b, c, h, w]
                            idx += 1
            return out
        else:
            raise ValueError("Unsupported layer type")
    
    def backward(self, dA, reg_lambda):
        if self.layer_type == "dense":
            # Pseudocode for dense backward pass.
            # You would compute gradients with respect to weights, biases, and the input.
            m = dA.shape[0]
            # Assume activation derivative is 1 for simplicity; plug in the correct derivative.
            dZ = dA  
            # Compute gradients (this is a simplified version).
            dW = np.dot(self.input_cache.T, dZ) / m + reg_lambda * self.dense_weights
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA_prev = np.dot(dZ, self.dense_weights.T)
            # (Store gradients as needed.)
            return dA_prev
        elif self.layer_type == "conv":
            # Pseudocode for convolutional backward pass.
            # For example, assume a helper function `conv_backward(self, dA)` exists.
            return conv_backward(self, dA)
        elif self.layer_type == "flatten":
            # Reshape dA (of shape [batch_size, flat_size]) back to the original input dimensions.
            batch_size = dA.shape[0]
            C, H, W = self.flatten_input_shape
            out = np.empty((batch_size, C, H, W), dtype=np.float64)
            for b in range(batch_size):
                idx = 0
                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            out[b, c, h, w] = dA[b, idx]
                            idx += 1
            return out
        else:
            raise ValueError("Unsupported layer type")
