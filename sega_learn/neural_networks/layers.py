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
        self.weight_gradients = None
        self.bias_gradients = None
        
    def zero_grad(self):
        """Reset the gradients of the weights and biases to zero."""
        self.weight_gradients = None
        self.bias_gradients = None
        
    def forward(self, X):
        Z = np.dot(X, self.weights) + self.biases
        self.input_cache = X
        self.output_cache = self.activate(Z)
        return self.output_cache

    def backward(self, dA, reg_lambda):
        m = self.input_cache.shape[0]
        dZ = dA * self.activation_derivative(self.output_cache)
        dW = np.dot(self.input_cache.T, dZ) / m + reg_lambda * self.weights
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.weights.T)
        
        self.weight_gradients = dW
        self.bias_gradients = db
        
        return dA_prev

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


# class ConvLayer:
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation="relu"):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size  # assume square kernels
#         self.stride = stride
#         self.padding = padding
#         # He initialization for convolutional weights
#         self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
#         self.biases = np.zeros((out_channels, 1))
#         self.activation = activation
#         # Placeholders for gradients and cache for backpropagation
#         self.weight_gradients = None
#         self.bias_gradients = None
#         self.input_cache = None

#     def forward(self, X):
#         """
#         X: numpy array with shape (batch_size, in_channels, height, width)
#         Returns:
#             Output feature maps after convolution and activation.
#         """
#         self.input_cache = X
#         batch_size, in_channels, h_in, w_in = X.shape
#         # Calculate output dimensions
#         h_out = int((h_in + 2 * self.padding - self.kernel_size) / self.stride) + 1
#         w_out = int((w_in + 2 * self.padding - self.kernel_size) / self.stride) + 1

#         # Apply padding if needed
#         if self.padding > 0:
#             X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
#         else:
#             X_padded = X

#         output = np.zeros((batch_size, self.out_channels, h_out, w_out))
#         # Convolution operation (naively implemented)
#         for b in range(batch_size):
#             for c in range(self.out_channels):
#                 for i in range(h_out):
#                     for j in range(w_out):
#                         h_start = i * self.stride
#                         h_end = h_start + self.kernel_size
#                         w_start = j * self.stride
#                         w_end = w_start + self.kernel_size
#                         region = X_padded[b, :, h_start:h_end, w_start:w_end]
#                         output[b, c, i, j] = np.sum(region * self.weights[c]) + self.biases[c]

#         # Apply activation
#         if self.activation == "relu":
#             return np.maximum(0, output)
#         elif self.activation == "sigmoid":
#             return 1 / (1 + np.exp(-output))
#         elif self.activation == "tanh":
#             return np.tanh(output)
#         else:
#             return output

#     def backward(self, d_out):
#         """
#         A simple (and not highly optimized) backward pass.
#         d_out: Gradient of the loss with respect to the layer output,
#                shape (batch_size, out_channels, h_out, w_out)
#         Returns:
#             dX: Gradient with respect to the input X.
#         """
#         X = self.input_cache
#         batch_size, in_channels, h_in, w_in = X.shape

#         # Pad input if necessary
#         if self.padding > 0:
#             X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
#         else:
#             X_padded = X

#         # Initialize gradients
#         d_weights = np.zeros_like(self.weights)
#         d_biases = np.zeros_like(self.biases)
#         d_X_padded = np.zeros_like(X_padded)
#         batch_size, out_channels, h_out, w_out = d_out.shape

#         for b in range(batch_size):
#             for c in range(out_channels):
#                 for i in range(h_out):
#                     for j in range(w_out):
#                         h_start = i * self.stride
#                         h_end = h_start + self.kernel_size
#                         w_start = j * self.stride
#                         w_end = w_start + self.kernel_size
#                         region = X_padded[b, :, h_start:h_end, w_start:w_end]
#                         d_weights[c] += d_out[b, c, i, j] * region
#                         d_biases[c] += d_out[b, c, i, j]
#                         d_X_padded[b, :, h_start:h_end, w_start:w_end] += d_out[b, c, i, j] * self.weights[c]

#         # Remove padding from gradient if applied
#         if self.padding > 0:
#             d_X = d_X_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
#         else:
#             d_X = d_X_padded

#         self.weight_gradients = d_weights
#         self.bias_gradients = d_biases
#         return d_X


# class RNNLayer:
#     def __init__(self, input_size, hidden_size, activation="tanh"):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         # Initialize weights (small random values)
#         self.Wxh = np.random.randn(input_size, hidden_size) * 0.01  # Input-to-hidden weights
#         self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden-to-hidden weights
#         self.bh = np.zeros((1, hidden_size))                       # Biases
#         self.activation = activation
#         self.weight_gradients = {"Wxh": None, "Whh": None, "bh": None}
#         # Caches for backpropagation through time
#         self.last_inputs = None
#         self.last_hs = None

#     def forward(self, X):
#         """
#         X: numpy array with shape (batch_size, time_steps, input_size)
#         Returns:
#             hidden_states: numpy array with shape (batch_size, time_steps, hidden_size)
#         """
#         batch_size, time_steps, _ = X.shape
#         # Initialize hidden state to zeros
#         h = np.zeros((batch_size, self.hidden_size))
#         self.last_hs = {-1: h}
#         self.last_inputs = X
#         outputs = []
#         # Process each time step
#         for t in range(time_steps):
#             x_t = X[:, t, :]
#             h = np.tanh(np.dot(x_t, self.Wxh) + np.dot(h, self.Whh) + self.bh)
#             self.last_hs[t] = h
#             outputs.append(h)
#         # Stack outputs along time dimension
#         return np.stack(outputs, axis=1)

#     def backward(self, d_out, learning_rate=1e-2):
#         """
#         d_out: Gradient of the loss with respect to the hidden states,
#                shape (batch_size, time_steps, hidden_size)
#         learning_rate: Learning rate for parameter updates
#         Returns:
#             d_h_next: Gradient to propagate to previous network layers (from t=0)
#         """
#         X = self.last_inputs
#         batch_size, time_steps, _ = X.shape
#         dWxh = np.zeros_like(self.Wxh)
#         dWhh = np.zeros_like(self.Whh)
#         dbh = np.zeros_like(self.bh)
#         d_h_next = np.zeros((batch_size, self.hidden_size))

#         # Backpropagation through time
#         for t in reversed(range(time_steps)):
#             h = self.last_hs[t]
#             h_prev = self.last_hs[t-1]
#             # Total gradient at current time step
#             dh = d_out[:, t, :] + d_h_next
#             # Derivative of tanh activation
#             dtanh = (1 - h * h) * dh
#             dWxh += np.dot(X[:, t, :].T, dtanh)
#             dWhh += np.dot(h_prev.T, dtanh)
#             dbh += np.sum(dtanh, axis=0, keepdims=True)
#             d_h_next = np.dot(dtanh, self.Whh.T)

#         # Store gradients
#         self.weight_gradients = {"Wxh": dWxh, "Whh": dWhh, "bh": dbh}
#         # (Optionally, update weights here or do it externally with an optimizer)
#         # For example:
#         # self.Wxh -= learning_rate * dWxh
#         # self.Whh -= learning_rate * dWhh
#         # self.bh -= learning_rate * dbh
#         return d_h_next  # This could be used to propagate gradients to earlier layers
