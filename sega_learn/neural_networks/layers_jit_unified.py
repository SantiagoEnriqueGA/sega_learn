import numpy as np
from numba import jit, njit, types, float64, int32, prange
from numba.experimental import jitclass
from math import sqrt
from .numba_utils import (
    relu, relu_derivative,
    leaky_relu, leaky_relu_derivative,
    tanh, tanh_derivative,
    sigmoid, sigmoid_derivative,
    softmax, sum_axis0
)

# Define the unified spec.
spec = [
    # Common fields
    ('layer_type', int32),          # 0 = dense, 1 = conv
    ('activation', types.unicode_type),
    ('input_size', int32),          # For dense: number of inputs; for conv: in_channels
    ('output_size', int32),         # For dense: number of outputs; for conv: out_channels

    # Dense-specific fields
    ('dense_weights', float64[:, ::1]),       # shape: (input_size, output_size)
    ('dense_biases', float64[:, ::1]),          # shape: (1, output_size)
    ('dense_weight_gradients', float64[:, ::1]),
    ('dense_bias_gradients', float64[:, ::1]),
    ('dense_input_cache', float64[:, ::1]),
    ('dense_output_cache', float64[:, ::1]),

    # Convolution-specific fields
    ('in_channels', int32),         # equals input_size for conv layers
    ('out_channels', int32),        # equals output_size for conv layers
    ('kernel_size', int32),
    ('stride', int32),
    ('padding', int32),
    ('conv_weights', float64[:,:,:,:]),       # shape: (out_channels, in_channels, kernel_size, kernel_size)
    ('conv_biases', float64[:, ::1]),          # shape: (out_channels, 1)
    ('conv_weight_gradients', float64[:,:,:,:]),
    ('conv_bias_gradients', float64[:, ::1]),
    ('conv_input_cache', float64[:,:,:,:]),
    ('X_cols', float64[:,:,:]),
    ('X_padded', float64[:,:,:,:]),
    ('h_out', int32),
    ('w_out', int32)
]

@jitclass(spec)
class UnifiedJITLayer:
    """
    Unified JIT layer supporting both dense (fully-connected) and convolutional layers.
    
    The field `layer_type` distinguishes the two:
      - 0: Dense layer.
      - 1: Convolutional layer.
    
    For dense layers, the attributes prefixed with 'dense_' are used.
    For convolutional layers, the attributes prefixed with 'conv_' (and related) are used.
    Unused attributes for a given layer type are allocated as dummy arrays.
    """
    def __init__(self, layer_type, activation, 
                 input_size, output_size,
                 # Conv parameters (only used if layer_type==1)
                 in_channels=0, out_channels=0, kernel_size=0, stride=1, padding=0):
        self.layer_type = layer_type
        self.activation = activation
        
        if layer_type == 0:
            # Dense layer initialization.
            self.input_size = input_size
            self.output_size = output_size
            if activation in ["relu", "leaky_relu"]:
                scale = sqrt(2.0 / input_size)
            else:
                scale = sqrt(1.0 / input_size)
            self.dense_weights = np.random.randn(input_size, output_size) * scale
            self.dense_biases = np.zeros((1, output_size))
            self.dense_weight_gradients = np.zeros((input_size, output_size))
            self.dense_bias_gradients = np.zeros((1, output_size))
            # Cache arrays for forward/backward.
            self.dense_input_cache = np.zeros((1, input_size))
            self.dense_output_cache = np.zeros((1, output_size))
            # Set conv fields to dummy values.
            self.in_channels = 0
            self.out_channels = 0
            self.kernel_size = 0
            self.stride = 1
            self.padding = 0
            self.conv_weights = np.zeros((1, 1, 1, 1))
            self.conv_biases = np.zeros((1, 1))
            self.conv_weight_gradients = np.zeros((1, 1, 1, 1))
            self.conv_bias_gradients = np.zeros((1, 1))
            self.conv_input_cache = np.zeros((1, 1, 1, 1))
            self.X_cols = np.zeros((1, 1, 1))
            self.X_padded = np.zeros((1, 1, 1, 1))
            self.h_out = 1
            self.w_out = 1
        elif layer_type == 1:
            # Convolutional layer initialization.
            self.input_size = in_channels
            self.output_size = out_channels
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            # He initialization for conv weights.
            scale = sqrt(2.0 / (in_channels * kernel_size * kernel_size))
            self.conv_weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
            self.conv_biases = np.zeros((out_channels, 1))
            self.conv_weight_gradients = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
            self.conv_bias_gradients = np.zeros((out_channels, 1))
            # Cache arrays.
            self.conv_input_cache = np.zeros((1, in_channels, 1, 1))
            self.X_cols = np.zeros((1, 1, 1))
            self.X_padded = np.zeros((1, 1, 1, 1))
            self.h_out = 0
            self.w_out = 0
            # Set dense fields to dummy values.
            self.dense_weights = np.zeros((1, 1))
            self.dense_biases = np.zeros((1, 1))
            self.dense_weight_gradients = np.zeros((1, 1))
            self.dense_bias_gradients = np.zeros((1, 1))
            self.dense_input_cache = np.zeros((1, 1))
            self.dense_output_cache = np.zeros((1, 1))
        else:
            raise ValueError("Unsupported layer_type. Use 0 for dense or 1 for convolutional.")

    def zero_grad(self):
        if self.layer_type == 0:
            self.dense_weight_gradients = np.zeros_like(self.dense_weight_gradients)
            self.dense_bias_gradients = np.zeros_like(self.dense_bias_gradients)
        elif self.layer_type == 1:
            self.conv_weight_gradients = np.zeros_like(self.conv_weight_gradients)
            self.conv_bias_gradients = np.zeros_like(self.conv_bias_gradients)

    def forward(self, X):
        if self.layer_type == 0:
            # Dense forward: X shape (batch_size, input_size)
            Z = np.dot(X, self.dense_weights) + self.dense_biases
            self.dense_input_cache = X
            self.dense_output_cache = self.activate(Z)
            return self.dense_output_cache
        elif self.layer_type == 1:
            # Convolutional forward: X shape (batch_size, in_channels, height, width)
            self.conv_input_cache = X.copy()
            batch_size = X.shape[0]
            h_in = X.shape[2]
            w_in = X.shape[3]
            h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
            w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
            self.h_out = h_out
            self.w_out = w_out

            # Apply padding if needed.
            if self.padding > 0:
                X_padded = np.zeros((batch_size, self.in_channels,
                                     h_in + 2 * self.padding, w_in + 2 * self.padding))
                for b in range(batch_size):
                    for c in range(self.in_channels):
                        for i in range(h_in):
                            for j in range(w_in):
                                X_padded[b, c, i + self.padding, j + self.padding] = X[b, c, i, j]
            else:
                X_padded = X.copy()
            self.X_padded = X_padded

            # im2col transformation.
            X_cols = self._im2col(X_padded, h_out, w_out)
            self.X_cols = X_cols

            # Convolution: compute dot products.
            output = np.zeros((batch_size, self.out_channels, h_out * w_out))
            for b in range(batch_size):
                for o in range(self.out_channels):
                    for i in range(h_out * w_out):
                        val = self.conv_biases[o, 0]
                        for c in range(self.in_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    filter_idx = c * self.kernel_size * self.kernel_size + kh * self.kernel_size + kw
                                    val += self.conv_weights[o, c, kh, kw] * X_cols[b, filter_idx, i]
                        output[b, o, i] = val
            # Reshape output to (batch_size, out_channels, h_out, w_out).
            output_reshaped = np.zeros((batch_size, self.out_channels, h_out, w_out))
            for b in range(batch_size):
                for c in range(self.out_channels):
                    for i in range(h_out):
                        for j in range(w_out):
                            idx = i * w_out + j
                            output_reshaped[b, c, i, j] = output[b, c, idx]
            return self.activate(output_reshaped)
    
    def backward(self, dA, reg_lambda):
        if self.layer_type == 0:
            # Dense backward.
            m = self.dense_input_cache.shape[0]
            dZ = dA * self.activation_derivative(self.dense_output_cache)
            dW = np.dot(self.dense_input_cache.T, dZ) / m + reg_lambda * self.dense_weights
            db = sum_axis0(dZ) / m
            dA_prev = np.dot(dZ, self.dense_weights.T)
            self.dense_weight_gradients = dW
            self.dense_bias_gradients = db
            return dA_prev
        elif self.layer_type == 1:
            # Convolutional backward.
            batch_size = self.conv_input_cache.shape[0]
            # Apply activation derivative (here we assume elementwise derivative).
            d_activated = dA * self.activation_derivative(dA)
            # Reshape d_activated to (batch_size, out_channels, h_out*w_out).
            d_out_reshaped = np.zeros((batch_size, self.out_channels, self.h_out * self.w_out))
            for b in range(batch_size):
                for c in range(self.out_channels):
                    for i in range(self.h_out):
                        for j in range(self.w_out):
                            idx = i * self.w_out + j
                            d_out_reshaped[b, c, idx] = d_activated[b, c, i, j]
            d_weights = np.zeros_like(self.conv_weights)
            d_biases = np.zeros_like(self.conv_biases)
            d_X_cols = np.zeros_like(self.X_cols)
            for b in range(batch_size):
                # Bias gradients.
                for o in range(self.out_channels):
                    for i in range(self.h_out * self.w_out):
                        d_biases[o, 0] += d_out_reshaped[b, o, i]
                # Weight gradients.
                for o in range(self.out_channels):
                    for c in range(self.in_channels):
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                filter_idx = c * self.kernel_size * self.kernel_size + kh * self.kernel_size + kw
                                for i in range(self.h_out * self.w_out):
                                    d_weights[o, c, kh, kw] += d_out_reshaped[b, o, i] * self.X_cols[b, filter_idx, i]
                # Input gradients (accumulate contributions for each patch).
                for i in range(self.h_out * self.w_out):
                    for filter_idx in range(self.in_channels * self.kernel_size * self.kernel_size):
                        val = 0.0
                        for o in range(self.out_channels):
                            # Recover indices from filter_idx.
                            c_idx = filter_idx // (self.kernel_size * self.kernel_size)
                            rem = filter_idx % (self.kernel_size * self.kernel_size)
                            kh = rem // self.kernel_size
                            kw = rem % self.kernel_size
                            val += d_out_reshaped[b, o, i] * self.conv_weights[o, c_idx, kh, kw]
                        d_X_cols[b, filter_idx, i] = val
            d_X = self._col2im(d_X_cols, self.conv_input_cache.shape)
            if reg_lambda > 0:
                for o in range(self.out_channels):
                    for c in range(self.in_channels):
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                d_weights[o, c, kh, kw] += reg_lambda * self.conv_weights[o, c, kh, kw]
            self.conv_weight_gradients = d_weights / batch_size
            self.conv_bias_gradients = d_biases / batch_size
            return d_X

    def activate(self, Z):
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
            raise ValueError("Unsupported activation")

    def activation_derivative(self, Z):
        if self.activation == "relu":
            return relu_derivative(Z)
        elif self.activation == "leaky_relu":
            return leaky_relu_derivative(Z)
        elif self.activation == "tanh":
            return tanh_derivative(Z)
        elif self.activation == "sigmoid":
            return sigmoid_derivative(Z)
        elif self.activation == "softmax":
            return np.ones_like(Z)
        else:
            raise ValueError("Unsupported activation")
    
    # Helper methods for convolution.
    def _im2col(self, x, h_out, w_out):
        batch_size = x.shape[0]
        channels = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        col = np.zeros((batch_size, channels * self.kernel_size * self.kernel_size, h_out * w_out))
        for b in range(batch_size):
            col_idx = 0
            for i in range(0, h - self.kernel_size + 1, self.stride):
                for j in range(0, w - self.kernel_size + 1, self.stride):
                    flat_idx = 0
                    for c in range(channels):
                        for ki in range(self.kernel_size):
                            for kj in range(self.kernel_size):
                                col[b, flat_idx, col_idx] = x[b, c, i + ki, j + kj]
                                flat_idx += 1
                    col_idx += 1
        return col

    def _col2im(self, dcol, x_shape):
        batch_size = x_shape[0]
        channels = x_shape[1]
        h = x_shape[2]
        w = x_shape[3]
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding
        dx_padded = np.zeros((batch_size, channels, h_padded, w_padded))
        for b in range(batch_size):
            col_idx = 0
            for i in range(0, h_padded - self.kernel_size + 1, self.stride):
                for j in range(0, w_padded - self.kernel_size + 1, self.stride):
                    for c in range(channels):
                        for ki in range(self.kernel_size):
                            for kj in range(self.kernel_size):
                                flat_idx = c * self.kernel_size * self.kernel_size + ki * self.kernel_size + kj
                                dx_padded[b, c, i + ki, j + kj] += dcol[b, flat_idx, col_idx]
                    col_idx += 1
        if self.padding > 0:
            result = np.zeros((batch_size, channels, h, w))
            for b in range(batch_size):
                for c in range(channels):
                    for i in range(h):
                        for j in range(w):
                            result[b, c, i, j] = dx_padded[b, c, i + self.padding, j + self.padding]
            return result
        return dx_padded
