from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer
from .numba_utils import *

import numpy as np
from numba import njit, float64, int32, prange
from numba import types
from numba.experimental import jitclass
from tqdm.auto import tqdm

CACHE = False

class NeuralNetwork:
    """
    Neural network class for training and evaluating a custom neural network model.
    Parameters:
        - layer_sizes (list): A list of integers representing the sizes of each layer in the neural network.
        - dropout_rate (float): The dropout rate to be applied during training. Default is 0.2.
        - reg_lambda (float): The regularization lambda value. Default is 0.01.
        - activations (list): A list of activation functions for each layer. Default is ['relu', 'relu', ... 'softmax'].
    """
    
    def __init__(self, layer_sizes, dropout_rate=0.2, reg_lambda=0.01, activations=None, compile_numba=True, progress_bar=True):
        # Initialize neural network parameters
        self.compiled = False
        self.progress_bar = progress_bar
        
        self.layer_sizes = layer_sizes                                          # List of layer sizes
        self.dropout_rate = dropout_rate                                        # Dropout rate
        self.reg_lambda = reg_lambda                                            # Regularization lambda
        
        # Set default activation functions if not provided
        if activations is None:
            self.activations = ['relu'] * (len(layer_sizes) - 2) + ['softmax']       # Default to ReLU for hidden layers and Softmax for the output layer
        else:
            self.activations = activations
            
        # Initialize layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(
                layer_sizes[i], 
                layer_sizes[i+1], 
                self.activations[i]
            ))
            
        # Initialize weights
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01  # Small random weights
            bias = np.zeros((1, layer_sizes[i + 1]))  # Initialize biases to zeros
            self.weights.append(weight)
            self.biases.append(bias)
            
        # Cache for forward/backward pass
        self.layer_outputs = None
        self.is_binary = layer_sizes[-1] == 1

        # Cache for optimizer update
        self.dWs_cache = [np.zeros_like(w) for w in self.weights]
        self.dbs_cache = [np.zeros_like(b) for b in self.biases]
        
        # Compile Numba functions for performance
        if compile_numba and not self.compiled:
            self.compile_numba_functions(progress_bar)
        
            # Reset layer outputs and caches
            self.layer_outputs = None
            self.dWs_cache = [np.zeros_like(w) for w in self.weights]
            self.dbs_cache = [np.zeros_like(b) for b in self.biases]
        
            self.compiled = True
            print("Numba functions compiled successfully.")
        

    def compile_numba_functions(self, progress_bar=True):
        """
        Compiles all Numba JIT functions to improve performance.
        """
        # TODO: Ensure same as init_compile() from test_nn_numba.py
        if progress_bar:
            progress_bar = tqdm(total=33, desc="Compiling Numba functions")
        else:
            progress_bar = None
        # Neural network functions
        # --------------------------------------------------------------------
        _ = self._apply_dropout_jit(np.random.randn(10, 10), self.dropout_rate)
        if progress_bar: progress_bar.update(1)
        _ = self.compute_l2_reg(self.weights)
        if progress_bar: progress_bar.update(1)
        _ = self._process_batches(X_shuffled=np.random.randn(10, self.layer_sizes[0]), y_shuffled=np.random.randint(0, 2, 10),
                                  batch_size=32, weights=self.weights, biases=self.biases, activations=self.activations,
                                  dropout_rate=self.dropout_rate, is_binary=self.is_binary, reg_lambda=self.reg_lambda,
                                  dWs_acc=self.dWs_cache, dbs_acc=self.dbs_cache)
        if progress_bar: progress_bar.update(1)
        _ = self._evaluate_jit(np.random.randn(10, self.layer_sizes[0]), np.random.randint(0, 2, 10), self.is_binary)
        if progress_bar: progress_bar.update(1)
        
        # Initialize dummy layer outputs for backward pass
        self.layer_outputs = [np.random.randn(10, size) for size in self.layer_sizes]
        
        # Numba Utils functions
        # --------------------------------------------------------------------
        # Forward and backward passes
        _ = forward_jit(X=np.random.randn(10, self.layer_sizes[0]), weights=self.weights, biases=self.biases, activations=self.activations,
                         dropout_rate=self.dropout_rate, training=True, is_binary=self.is_binary)
        if progress_bar: progress_bar.update(1)
        _ = backward_jit(self.layer_outputs, np.random.randint(0, 2, 10), self.weights, self.activations, self.reg_lambda,
                          self.is_binary, self.dWs_cache, self.dbs_cache)
        if progress_bar: progress_bar.update(1)
        # Loss functions and evaluation
        _ = calculate_loss_from_outputs_binary(np.random.randn(10, 1), np.random.randint(0, 2, 10).astype(np.float64), self.reg_lambda, self.weights)
        if progress_bar: progress_bar.update(1)
        _ = calculate_loss_from_outputs_multi(np.random.randn(10, self.layer_sizes[-1]), np.eye(self.layer_sizes[-1])[np.random.randint(0, self.layer_sizes[-1], 10)], self.reg_lambda, self.weights)
        if progress_bar: progress_bar.update(1)
        _ = evaluate_batch(np.random.randn(10, self.layer_sizes[-1]), np.random.randint(0, 2, 10), self.is_binary)
        if progress_bar: progress_bar.update(1)
        # Activation functions
        _ = relu(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = relu_derivative(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = leaky_relu(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = leaky_relu_derivative(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = tanh(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = tanh_derivative(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = sigmoid(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = sigmoid_derivative(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = softmax(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        # Other utility functions
        _ = sum_reduce(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        _ = sum_axis0(np.random.randn(10, 10))
        if progress_bar: progress_bar.update(1)
        
        # Optimizers
        # --------------------------------------------------------------------
        # Adam
        _adam = AdamOptimizer()
        if progress_bar: progress_bar.update(1)
        _adam.initialize(self.layers)
        if progress_bar: progress_bar.update(1)
        _adam.update_layers(self.layers, self.dWs_cache, self.dbs_cache)
        if progress_bar: progress_bar.update(1)
        # SGD
        _sgd = SGDOptimizer()
        if progress_bar: progress_bar.update(1)
        _sgd.initialize(self.layers)
        if progress_bar: progress_bar.update(1)
        _sgd.update_layers(self.layers, self.dWs_cache, self.dbs_cache)
        if progress_bar: progress_bar.update(1)
        # Adadelta
        _adadelta = AdadeltaOptimizer()
        if progress_bar: progress_bar.update(1)
        _adadelta.initialize(self.layers)
        if progress_bar: progress_bar.update(1)
        _adadelta.update_layers(self.layers, self.dWs_cache, self.dbs_cache)
        if progress_bar: progress_bar.update(1)
        
        # Loss Modules
        # --------------------------------------------------------------------
        _cross_entropy = CrossEntropyLoss()
        if progress_bar: progress_bar.update(1)
        _cross_entropy.calculate_loss(np.random.randn(10, self.layer_sizes[-1]), np.eye(self.layer_sizes[-1])[np.random.randint(0, self.layer_sizes[-1], 10)])
        if progress_bar: progress_bar.update(1)
        _bce = BCEWithLogitsLoss()
        if progress_bar: progress_bar.update(1)
        _bce.calculate_loss(np.random.randn(10, 1), np.random.randint(0, 2, 10).astype(np.float64).reshape(-1, 1))
        if progress_bar: progress_bar.update(1)
        del _adam, _sgd, _adadelta, _cross_entropy, _bce, _
        if progress_bar:
            progress_bar.close()
        
    
    def apply_dropout(self, X):
        """
        Applies dropout to the activation X.
        Args:
            X (ndarray): Activation values.
        Returns:
            ndarray: Activation values after applying dropout.
        """
        # Use JIT-compiled function for performance
        return self._apply_dropout_jit(X, self.dropout_rate)
        
    @staticmethod
    @njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
    def _apply_dropout_jit(X, dropout_rate):
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

    def forward(self, X, training=True):
        """
        Performs forward propagation through the neural network.
        Args: 
            X (ndarray): Input data of shape (batch_size, input_size).
            training (bool): Whether the network is in training mode (applies dropout).
        Returns: 
            ndarray: Output predictions of shape (batch_size, output_size).
        """              
        self.layer_outputs = [X]
        weights = [layer.weights for layer in self.layers]
        biases = [layer.biases for layer in self.layers]
        layer_outputs = forward_jit(X, weights, biases, self.activations, self.dropout_rate, training, self.is_binary)
        self.layer_outputs = layer_outputs  # Update layer_outputs with the correct shapes
        
        return self.layer_outputs[-1]
             
    def backward(self, y):
        """
        Performs backward propagation to calculate the gradients.
        Parameters: 
            y (ndarray): Target labels of shape (m, output_size).
        """            
        # Reset gradients to zero
        for i in range(len(self.dWs_cache)):
            self.dWs_cache[i].fill(0)
            self.dbs_cache[i].fill(0)
            
        # Use cached arrays in JIT function
        dWs, dbs = backward_jit(self.layer_outputs, y, self.weights, 
                                      self.activations, self.reg_lambda, 
                                      self.is_binary, self.dWs_cache, self.dbs_cache)
        
        for i, layer in enumerate(self.layers):
            layer.weight_gradients = dWs[i]
            layer.bias_gradients = dbs[i]

    @staticmethod
    @njit(fastmath=True, nogil=True, cache=CACHE)
    def _process_batches(X_shuffled, y_shuffled, batch_size, weights, biases, activations, dropout_rate, is_binary, reg_lambda, dWs_acc, dbs_acc):
        """
        Process multiple batches in parallel using Numba.
        
        Args:
            X_shuffled (ndarray): Shuffled input features
            y_shuffled (ndarray): Shuffled target labels
            batch_size (int): Size of each mini-batch
            weights (list): List of weight matrices
            biases (list): List of bias vectors
            activations (list): List of activation functions
            dropout_rate (float): Dropout rate
            is_binary (bool): Whether the task is binary classification
            reg_lambda (float): Regularization parameter
            dWs_acc (list): Accumulated weight gradients as zeros
            dbs_acc (list): Accumulated bias gradients as zeros
            
        Returns:
            tuple: Lists of weight and bias gradients for each batch, loss, and accuracy
        """
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
            layer_outputs = forward_jit(X_batch, weights, biases, activations, dropout_rate, True, is_binary)
            
            # Backward pass
            dWs, dbs = backward_jit(layer_outputs, y_batch, weights, activations, reg_lambda, is_binary, dWs_acc, dbs_acc)
            
            # Calculate loss
            if is_binary:
                running_loss += calculate_loss_from_outputs_binary(layer_outputs[-1], y_batch, reg_lambda, weights)
            else:
                y_batch_ohe = np.eye(weights[-1].shape[1])[y_batch]  # One-hot encode y for multi-class
                running_loss += calculate_loss_from_outputs_multi(layer_outputs[-1], y_batch_ohe, reg_lambda, weights)
                
            # Calculate accuracy
            running_accuracy += evaluate_batch(layer_outputs[-1], y_batch, is_binary)
                        
            # Accumulate gradients directly
            for j in range(len(dWs)):
                dWs_acc[j] += dWs[j]
                dbs_acc[j] += dbs[j]
        
        # Average the accumulated gradients
        for j in range(len(dWs_acc)):
            dWs_acc[j] /= num_batches
            dbs_acc[j] /= num_batches
            
        # Calculate average loss and accuracy
        running_loss /= num_batches
        running_accuracy /= num_batches
            
        return dWs_acc, dbs_acc, running_loss, running_accuracy
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              optimizer=None, epochs=100, batch_size=32, 
              early_stopping_threshold=10, lr_scheduler=None, p=True, use_tqdm=True, n_jobs=1):
        """
        Trains the neural network model.
        Parameters:
            - X_train (ndarray): Training data features.
            - y_train (ndarray): Training data labels.
            - X_val (ndarray): Validation data features, optional.
            - y_val (ndarray): Validation data labels, optional.
            - optimizer (Optimizer): Optimizer for updating parameters (default: Adam, lr=0.0001).
            - epochs (int): Number of training epochs (default: 100).
            - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
            - early_stopping_patience (int): Patience for early stopping (default: 10).
            - lr_scheduler (Scheduler): Learning rate scheduler (default: None).
            - verbose (bool): Whether to print training progress (default: True).
            - use_tqdm (bool): Whether to use tqdm for progress bar (default: True).
            - n_jobs (int): Number of jobs for parallel processing (default: 1).
        """
        # Default optimizer if not provided
        if optimizer is None:
            optimizer = AdamOptimizer(learning_rate=0.0001)
            
        # Initialize optimizer
        optimizer.initialize(self.layers)
        
        # Track best model for early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_weights = [layer.weights.copy() for layer in self.layers]
        best_biases = [layer.biases.copy() for layer in self.layers]
        
        # Number of threads for parallel processing
        # If n_jobs > 1, use that many threads, otherwise let Numba decide
        if n_jobs > 1:
            import os
            os.environ['NUMBA_NUM_THREADS'] = str(n_jobs)

        # Training loop with progress bar
        progress_bar = tqdm(range(epochs)) if use_tqdm else range(epochs)
        for epoch in progress_bar:
            # Reset gradients
            for layer in self.layers:
                layer.zero_grad()
            
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Prepare layer parameters for JIT function
            weights = [layer.weights for layer in self.layers]
            biases = [layer.biases for layer in self.layers]
            activations = [layer.activation for layer in self.layers]
           
            # Initialize accumulated gradients with zeros 
            dWs_zeros = [np.zeros_like(w) for w in weights]
            dbs_zeros = [np.zeros_like(b) for b in biases]
           
            # Process all batches in parallel and get averaged gradients, loss
            dWs_acc, dbs_acc, train_loss, train_accuracy = self._process_batches(
                X_shuffled, y_shuffled, batch_size, weights, biases, 
                activations, self.dropout_rate, self.is_binary, self.reg_lambda,
                dWs_zeros, dbs_zeros
            )
                       
            # Update weights and biases using the optimizer
            optimizer.update_layers(self.layers, dWs_acc, dbs_acc)
                        
            # Validation metrics
            val_metrics = ""
            if X_val is not None:
                val_loss = self.calculate_loss(X_val, y_val)
                val_accuracy, _ = self.evaluate(X_val, y_val)
                val_metrics = f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    best_weights = [layer.weights.copy() for layer in self.layers]
                    best_biases = [layer.biases.copy() for layer in self.layers]
                else:
                    patience_counter += 1
            else:
                # Use training loss for early stopping if no validation set
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                    best_weights = [layer.weights.copy() for layer in self.layers]
                    best_biases = [layer.biases.copy() for layer in self.layers]
                else:
                    patience_counter += 1
            
            # Update progress bar or print metrics
            if p:
                if use_tqdm and isinstance(progress_bar, tqdm):
                    progress_bar.set_description(
                        f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}{val_metrics}"
                    )
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}{val_metrics}")
            
            # Learning rate scheduler step
            if lr_scheduler:
                if isinstance(lr_scheduler, lr_scheduler_plateau):
                    msg = lr_scheduler.step(epoch, train_loss if X_val is None else val_loss)
                    if p and msg:
                        tqdm.write(msg)
                else:
                    msg = lr_scheduler.step(epoch)
                    if p and msg:
                        tqdm.write(msg)
                    
            # Early stopping
            if patience_counter >= early_stopping_threshold:
                if p and use_tqdm:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        for i, layer in enumerate(self.layers):
            layer.weights = best_weights[i]
            layer.biases = best_biases[i]
            

    @staticmethod
    @njit(fastmath=True, nogil=True, parallel=True, cache=CACHE)
    def compute_l2_reg(weights):
        total = 0.0
        for i in prange(len(weights)):
            total += np.sum(weights[i] ** 2)
        return total        
    
    def calculate_loss(self, X, y, class_weights=None):
        """
        Calculates the loss with L2 regularization.
        Parameters:
            - X (ndarray): Input data
            - y (ndarray): Target labels
            - class_weights (ndarray, optional): Weights for each class
        Returns: 
            float: The calculated loss value
        """
        # Get predictions
        outputs = self.forward(X, training=False)
        
        # # Apply class weights if provided
        # if class_weights is None:
        #     class_weights = np.ones_like(y)
        # elif isinstance(class_weights, np.ndarray):
        #     class_weights = np.asarray(class_weights)
        
        # One-hot encode y for multi-class classification
        if not self.is_binary and y.ndim == 1:
            y = np.eye(self.layer_sizes[-1])[y]
        
        # Select appropriate loss function
        if self.is_binary:
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn.calculate_loss(outputs, y.reshape(-1, 1))
        else:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn.calculate_loss(outputs, y)
        
        # Add L2 regularization
        weights = [layer.weights for layer in self.layers]
        l2_reg = self.reg_lambda * self.compute_l2_reg(weights)
        loss += l2_reg
        
        # Convert to Python float
        return float(loss)

    def evaluate(self, X, y):
        """
        Evaluates the model performance.
        Parameters:
            - X (ndarray): Input data
            - y (ndarray): Target labels
        Returns:
            - accuracy (float): Model accuracy
            - predicted (ndarray): Predicted labels
        """
        # Get predictions
        y_hat = self.forward(X, training=False)
        # JIT-compiled function for evaluation
        accuracy, predicted = self._evaluate_jit(y_hat, y, self.is_binary)
        return accuracy, predicted
        
    @staticmethod
    @njit(fastmath=True, nogil=True, cache=CACHE)
    def _evaluate_jit(y_hat, y_true, is_binary):
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
        
    def predict(self, X):
        """
        Generate predictions for input data.
        Parameters:
            - X (ndarray): Input data
        Returns:
            - predictions: Model predictions (class probabilities or labels)
        """
        # Get raw predictions
        outputs = self.forward(X, training=False)
        
        # For binary classification, return class probabilities
        if self.is_binary:
            return outputs
        # For multiclass, return class labels
        else:
            return np.argmax(outputs, axis=1)
            
    def _create_optimizer(self, optimizer_type, learning_rate):
        """Helper method to create optimizer instances."""
        if optimizer_type == 'Adam':
            return AdamOptimizer(learning_rate)
        elif optimizer_type == 'SGD':
            return SGDOptimizer(learning_rate)
        elif optimizer_type == 'Adadelta':
            return AdadeltaOptimizer(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
    def create_scheduler(self, scheduler_type, optimizer, **kwargs):
        """Creates a learning rate scheduler."""
        if scheduler_type == 'step':
            return lr_scheduler_step(optimizer, **kwargs)
        elif scheduler_type == 'plateau':
            return lr_scheduler_plateau(optimizer, **kwargs)
        elif scheduler_type == 'exp':
            return lr_scheduler_exp(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

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

