import numpy as np
from numba import njit, prange, float64

CACHE = False

# -------------------------------------------------------------------------------------------------
# Forward and backward passes
# -------------------------------------------------------------------------------------------------
@njit(fastmath=True, nogil=True, cache=CACHE)
def forward_jit(X, weights, biases, activations, dropout_rate, training, is_binary):
    num_layers = len(weights)
    # Initialize layer_outputs with empty 2D float64 arrays to set the correct type
    layer_outputs = [np.empty((0, 0), dtype=np.float64) for _ in range(num_layers + 1)]
    layer_outputs[0] = X
    
    # Forward pass through all layers except the last
    for i in range(num_layers - 1):
        # Calculate linear transformation
        Z = np.dot(layer_outputs[i], weights[i]) + biases[i]
        if activations[i] == "relu":
            layer_outputs[i + 1] = relu(Z)
        elif activations[i] == "leaky_relu":
            layer_outputs[i + 1] = leaky_relu(Z)
        elif activations[i] == "tanh":
            layer_outputs[i + 1] = tanh(Z)
        elif activations[i] == "sigmoid":
            layer_outputs[i + 1] = sigmoid(Z)
        elif activations[i] == "softmax":
            layer_outputs[i + 1] = softmax(Z)
        else:
            raise ValueError(f"Unsupported activation: {activations[i]}")
        
        # Apply dropout only during training
        if training and dropout_rate > 0:
            # layer_outputs[i + 1] = np.multiply(layer_outputs[i + 1], 
            #                                   np.random.rand(*layer_outputs[i + 1].shape) < (1 - dropout_rate)) / (1 - dropout_rate)
            layer_outputs[i + 1] = apply_dropout_jit(layer_outputs[i + 1], dropout_rate)
                        
    # Last layer (output layer)
    Z = np.dot(layer_outputs[-2], weights[-1]) + biases[-1]
    if is_binary:
        layer_outputs[-1] = sigmoid(Z)
    else:
        layer_outputs[-1] = softmax(Z)
    
    return layer_outputs

@njit(fastmath=True, nogil=True, cache=CACHE)
def backward_jit(layer_outputs, y, weights, activations, reg_lambda, is_binary, dWs, dbs):
    m = y.shape[0]
    num_layers = len(weights)

    # Reshape y for binary classification
    # Calculate initial gradient based on loss function
    outputs = layer_outputs[-1]
    if is_binary:
        y = y.reshape(-1, 1).astype(np.float64)
        dA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15))
    else:
        dA = outputs.copy()
        # Replace advanced indexing with a loop
        for i in range(m):
            dA[i, y[i]] -= 1  # Subtract 1 from the correct class index for each sample
        
    
    # Backpropagate through layers in reverse
    for i in range(num_layers - 1, -1, -1): 
        prev_activation = layer_outputs[i]

        if i < num_layers - 1:
            output = layer_outputs[i + 1]
            if activations[i] == "relu":
                dZ = dA * relu_derivative(output)
            elif activations[i] == "leaky_relu":
                alpha = 0.01  # Consider passing as a parameter if needed
                dZ = dA * leaky_relu_derivative(output, alpha)
            elif activations[i] == "tanh":
                dZ = dA * tanh_derivative(output)
            elif activations[i] == "sigmoid":
                dZ = dA * sigmoid_derivative(output)
            elif activations[i] == "softmax":
                dZ = dA  # Typically not used in hidden layers
            else:
                raise ValueError(f"Unsupported activation: {activations[i]}")
        else:
            dZ = dA

        dW = np.dot(prev_activation.T, dZ) / m + reg_lambda * weights[i]
        # db = sum_reduce(dZ) / m
        db = sum_axis0(dZ) / m

        dWs[i] += dW
        dbs[i] += db

        if i > 0:
            dA = np.dot(dZ, weights[i].T)

    return dWs, dbs



# -------------------------------------------------------------------------------------------------
# Loss functions and accuracy
# -------------------------------------------------------------------------------------------------
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_loss_from_outputs_binary(outputs, y, weights, reg_lambda):
    # Apply loss function
    loss = calculate_bce_with_logits_loss(outputs, y)
    
    # If weights is a list, compute L2 regularization for each weight matrix
    # If not a list, only one value, so no regularization needed
    if isinstance(weights, list):
        # Add L2 regularization
        l2_reg = reg_lambda * _compute_l2_reg(weights)
        # Add L2 regularization to loss
        loss += l2_reg
        
    return float(loss)

@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_loss_from_outputs_multi(outputs, y, weights, reg_lambda):
    # Apply loss function
    loss = calculate_cross_entropy_loss(outputs, y)
    
    # If weights is a list, compute L2 regularization for each weight matrix
    if isinstance(weights, list):
        # Add L2 regularization
        l2_reg = reg_lambda * _compute_l2_reg(weights)
        # Add L2 regularization to loss
        loss += l2_reg

    return float(loss)
    

@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_cross_entropy_loss(logits, targets):
    n = logits.shape[0]
    loss = 0.0
    for i in prange(n):
        max_val = np.max(logits[i])
        exp_sum = 0.0
        for j in range(logits.shape[1]):
            exp_sum += np.exp(logits[i, j] - max_val)
        log_sum_exp = max_val + np.log(exp_sum)
        c_i = np.argmax(targets[i])  # True class index, assuming one-hot targets
        loss += -logits[i, c_i] + log_sum_exp
    return loss / n

@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_bce_with_logits_loss(logits, targets):
    probs = 1 / (1 + np.exp(-logits))  # Apply sigmoid to logits to get probabilities
    loss = -np.mean(targets * np.log(probs + 1e-15) + (1 - targets) * np.log(1 - probs + 1e-15))  # Binary cross-entropy loss
    return loss


@njit(fastmath=True, nogil=True, parallel=True, cache=CACHE)
def _compute_l2_reg(weights):
    total = 0.0
    for i in prange(len(weights)):
        total += np.sum(weights[i] ** 2)
    return total

@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_batch(y_hat, y_true, is_binary):
    if is_binary:
        predicted = (y_hat > 0.5).astype(np.int32).flatten()
        accuracy = np.mean(predicted == y_true.flatten())
    else:
        predicted = np.argmax(y_hat, axis=1).astype(np.int32)
        accuracy = np.mean(predicted == y_true)
    return accuracy



# -------------------------------------------------------------------------------------------------
# Activation functions
# -------------------------------------------------------------------------------------------------
@njit(fastmath=True, cache=CACHE)
def relu(z):
    return np.maximum(0, z)

@njit(fastmath=True, cache=CACHE)
def relu_derivative(z):
    return (z > 0).astype(np.float64)  # Ensure return type is float64

@njit(fastmath=True, cache=CACHE)
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

@njit(fastmath=True, cache=CACHE)
def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha).astype(np.float64)  # Ensure return type is float64

@njit(fastmath=True, cache=CACHE)
def tanh(z):
    return np.tanh(z)

@njit(fastmath=True, cache=CACHE)
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

@njit(fastmath=True, cache=CACHE)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@njit(fastmath=True, cache=CACHE)
def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

@njit(parallel=True, fastmath=True, cache=CACHE)
def softmax(z):
    out = np.empty_like(z)
    for i in prange(z.shape[0]):
        row = z[i]
        max_val = np.max(row)
        shifted = row - max_val
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        out[i] = exp_vals / sum_exp
    return out



# -------------------------------------------------------------------------------------------------
# Other utility functions
# -------------------------------------------------------------------------------------------------
@njit(fastmath=True, cache=CACHE)
def sum_reduce(arr):
    sum_vals = np.empty((arr.shape[0], 1), dtype=arr.dtype)
    for i in range(arr.shape[0]):
        sum_vals[i, 0] = np.sum(arr[i])
    return sum_vals

@njit(fastmath=True, nogil=True, cache=CACHE)
def sum_axis0(arr):
    sum_vals = np.zeros((1, arr.shape[1]), dtype=arr.dtype)
    for j in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            sum_vals[0, j] += arr[i, j]
    return sum_vals



# -------------------------------------------------------------------------------------------------
# Neural Network functions
# -------------------------------------------------------------------------------------------------
@njit(fastmath=True, nogil=True, cache=CACHE)
def apply_dropout_jit(X, dropout_rate):
    """
    Numba JIT-compiled function to apply dropout.
    Args:
        X (ndarray): Activation values.
        dropout_rate (float): Dropout rate.
    Returns:
        ndarray: Activation values after applying dropout.
    """
    # Generate the entire mask at once - more vectorized approach
    mask = (np.random.random(X.shape) < (1 - dropout_rate)).astype(np.float64)
    return (X * mask) / (1 - dropout_rate)

@njit(fastmath=True, nogil=True, cache=CACHE)
def compute_l2_reg(weights):
    total = 0.0
    for i in prange(len(weights)):
        total += np.sum(weights[i] ** 2)
    return total     

@njit(fastmath=True, nogil=True, cache=CACHE)
def one_hot_encode(y, num_classes):
    m = y.shape[0]
    y_ohe = np.zeros((m, num_classes), dtype=np.float64)
    for i in range(m):
        y_ohe[i, y[i]] = 1.0
    return y_ohe

@njit(fastmath=True, nogil=True, cache=CACHE)
def process_batches_binary(X_shuffled, y_shuffled, batch_size, weights, biases, activations, dropout_rate, reg_lambda, dWs_acc, dbs_acc):
    num_samples = X_shuffled.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    running_loss = 0.0
    running_accuracy = 0.0
    
    for i in prange(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        # Forward pass
        layer_outputs = forward_jit(X_batch, weights, biases, activations, dropout_rate, True, True)
        
        # Backward pass
        dWs, dbs = backward_jit(layer_outputs, y_batch, weights, activations, reg_lambda, True, dWs_acc, dbs_acc)
        
        # Calculate loss and accuracy for binary classification
        running_loss += calculate_loss_from_outputs_binary(layer_outputs[-1], y_batch, reg_lambda, weights)
        running_accuracy += evaluate_batch(layer_outputs[-1], y_batch, True)
        
        # Accumulate gradients
        for j in range(len(dWs)):
            dWs_acc[j] += dWs[j]
            dbs_acc[j] += dbs[j]
    
    # Average the accumulated gradients, loss, and accuracy
    for j in range(len(dWs_acc)):
        dWs_acc[j] /= num_batches
        dbs_acc[j] /= num_batches
    running_loss /= num_batches
    running_accuracy /= num_batches
    
    return dWs_acc, dbs_acc, running_loss, running_accuracy

@njit(fastmath=True, nogil=True, cache=CACHE)
def process_batches_multi(X_shuffled, y_shuffled, batch_size, weights, biases, activations, dropout_rate, reg_lambda, dWs_acc, dbs_acc):
    num_samples = X_shuffled.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    running_loss = 0.0
    running_accuracy = 0.0
    
    for i in prange(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        # Forward pass
        layer_outputs = forward_jit(X_batch, weights, biases, activations, dropout_rate, True, False)
        
        # Backward pass
        dWs, dbs = backward_jit(layer_outputs, y_batch, weights, activations, reg_lambda, False, dWs_acc, dbs_acc)
        
        # One-hot encode for multi-class loss
        y_batch_ohe = one_hot_encode(y_batch, weights[-1].shape[1])
        running_loss += calculate_loss_from_outputs_multi(layer_outputs[-1], y_batch_ohe, reg_lambda, weights)
        running_accuracy += evaluate_batch(layer_outputs[-1], y_batch, False)
        
        # # Accumulate gradients
        # for j in range(len(dWs)):
        #     dWs_acc[j] += dWs[j]
        #     dbs_acc[j] += dbs[j]
    
    # Average the accumulated gradients, loss, and accuracy
    for j in range(len(dWs_acc)):
        dWs_acc[j] /= num_batches
        dbs_acc[j] /= num_batches
    running_loss /= num_batches
    running_accuracy /= num_batches
    
    return dWs_acc, dbs_acc, running_loss, running_accuracy

@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_jit(y_hat, y_true, is_binary):
    """
    Numba JIT-compiled function to evaluate model performance.
    Args:
        y_hat (ndarray): Model predictions.
        is_binary (bool): Whether the model is binary or multi-class.
    Returns:
        tuple: Accuracy and predicted labels.
    """
    if is_binary:
        predicted = (y_hat > 0.5).astype(np.int32).flatten()
        accuracy = np.mean(predicted == y_true.flatten())
    else:
        predicted = np.argmax(y_hat, axis=1).astype(np.int32)
        accuracy = np.mean(predicted == y_true)
    return accuracy, predicted