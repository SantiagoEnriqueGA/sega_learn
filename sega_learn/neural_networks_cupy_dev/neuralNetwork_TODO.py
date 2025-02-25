# Optimization Opportunities

# Matrix Operations Efficiency: There are opportunities to optimize the forward and backward passes.
# Memory Management: Some operations create unnecessary intermediate arrays.
# Code Organization: The Layer and NeuralNetwork classes could benefit from better structure.
# Hyperparameter Tuning: The tuning process could be optimized.
# GPU Memory Utilization: CuPy operations could be managed more efficiently.

# Let's implement these optimizations:
# Optimized Neural Network ImplementationClick to open codeTap to open
# Key Optimizations
# I've made several significant improvements to the neural network implementation:
# 1. Performance Improvements

# Improved Matrix Operations: Reduced redundant calculations in the forward and backward passes
# Efficient Memory Usage: Minimized temporary array creation for better GPU memory utilization
# Numerical Stability: Added clipping and max-subtraction for numerical stability in softmax and sigmoid functions
# He Weight Initialization: Used He initialization for ReLU/LeakyReLU layers for faster convergence

# 2. Code Quality Improvements

# Cleaner Architecture: Restructured the Layer and NeuralNetwork classes for better organization
# Better Type Handling: More consistent handling of CuPy/NumPy arrays
# Simpler API: Added a predict() method for more intuitive inference
# Progress Tracking: Used tqdm for better training progress visualization

# 3. Training Enhancements

# Improved Early Stopping: Added best model tracking to restore optimal weights
# Better Hyperparameter Tuning: Simplified process with logarithmic learning rate search
# Training/Inference Modes: Added a training flag to conditionally apply dropout

# 4. GPU-Specific Optimizations

# Float32 Usage: Used float32 for derivatives to reduce memory usage
# Memory Management: Avoided unnecessary GPU memory allocations

# These changes should significantly improve both the performance and usability of your neural network implementation. Would you like me to explain any specific part of the optimization in more detail?


from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer

import cupy as cp
import numpy as np
from tqdm import tqdm

class NeuralNetwork:
    """
    Neural network class for training and evaluating a custom neural network model.
    Parameters:
        - layer_sizes (list): A list of integers representing the sizes of each layer in the neural network.
        - dropout_rate (float): The dropout rate to be applied during training. Default is 0.2.
        - reg_lambda (float): The regularization lambda value. Default is 0.01.
        - activations (list): A list of activation functions for each layer. Default is ['relu', 'relu', ... 'softmax'].
    """
    
    def __init__(self, layer_sizes, dropout_rate=0.2, reg_lambda=0.01, activations=None):
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.reg_lambda = reg_lambda
        
        # Set default activations if not provided
        if activations is None:
            self.activations = ['relu'] * (len(layer_sizes) - 2) + ['softmax']
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
        
        # Cache for forward/backward pass
        self.layer_outputs = None
        self.is_binary = layer_sizes[-1] == 1

    def __repr__(self):
        layers_info = []
        for i, layer in enumerate(self.layers):
            layers_info.append(f"Layer {i}: {layer.weights.shape[0]} → {layer.weights.shape[1]}, activation: {layer.activation}")
        
        return f"NeuralNetwork(\n  layer_sizes={self.layer_sizes},\n  layers=[\n    " + \
               "\n    ".join(layers_info) + "\n  ],\n  " + \
               f"dropout_rate={self.dropout_rate},\n  reg_lambda={self.reg_lambda}\n)"

    def __str__(self):
        return self.__repr__()

    def forward(self, X, training=True):
        """
        Performs forward propagation through the neural network.
        Args: 
            X (ndarray): Input data of shape (batch_size, input_size).
            training (bool): Whether the network is in training mode (applies dropout).
        Returns: 
            ndarray: Output predictions of shape (batch_size, output_size).
        """
        # Convert numpy array to CuPy if needed
        if isinstance(X, np.ndarray):
            X = cp.asarray(X)
            
        # Store all layer activations for backprop
        self.layer_outputs = [X]
        
        # Forward pass through all layers except the last
        A = X
        for i, layer in enumerate(self.layers[:-1]):
            Z = cp.dot(A, layer.weights) + layer.biases
            A = layer.activate(Z)
            
            # Apply dropout only during training
            if training and self.dropout_rate > 0:
                mask = cp.random.rand(*A.shape) < (1 - self.dropout_rate)
                A = cp.multiply(A, mask) / (1 - self.dropout_rate)
                
            self.layer_outputs.append(A)
        
        # Last layer (output layer)
        Z = cp.dot(A, self.layers[-1].weights) + self.layers[-1].biases
        
        # Apply appropriate activation for output layer
        if self.is_binary:
            output = Activation.sigmoid(Z)
        else:
            output = Activation.softmax(Z)
            
        self.layer_outputs.append(output)
        return output

    def backward(self, y):
        """
        Performs backward propagation to calculate the gradients.
        Parameters: 
            y (ndarray): Target labels of shape (m, output_size).
        """
        # Convert numpy array to CuPy if needed
        if isinstance(y, np.ndarray):
            y = cp.asarray(y)
            
        m = y.shape[0]  # Number of samples
        
        # Reshape y for binary classification
        if self.is_binary:
            y = y.reshape(-1, 1)
            
        # Calculate initial gradient based on loss function
        outputs = self.layer_outputs[-1]
        if self.is_binary:
            # Gradient for binary cross-entropy
            dA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15))
        else:
            # Gradient for categorical cross-entropy with softmax
            dA = outputs - y
        
        # Backpropagate through layers in reverse
        for i in reversed(range(len(self.layers))):
            # Get activation from previous layer
            prev_activation = self.layer_outputs[i]
            
            # For all except the last layer, apply activation derivative
            if i < len(self.layers) - 1:
                dZ = dA * self.layers[i].activation_derivative(self.layer_outputs[i+1])
            else:
                dZ = dA
                
            # Calculate gradients
            dW = cp.dot(prev_activation.T, dZ) / m
            # Add L2 regularization
            dW += self.reg_lambda * self.layers[i].weights
            db = cp.sum(dZ, axis=0, keepdims=True) / m
            
            # Store gradients in layer
            self.layers[i].gradients = (dW, db)
            
            # Calculate dA for next iteration (previous layer)
            if i > 0:  # No need to calculate for input layer
                dA = cp.dot(dZ, self.layers[i].weights.T)

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              optimizer=None, epochs=100, batch_size=32, 
              early_stopping_patience=10, lr_scheduler=None, verbose=True):
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
        """
        # Convert arrays to CuPy
        X_train = cp.asarray(X_train) if isinstance(X_train, np.ndarray) else X_train
        y_train = cp.asarray(y_train) if isinstance(y_train, np.ndarray) else y_train
        
        if X_val is not None:
            X_val = cp.asarray(X_val) if isinstance(X_val, np.ndarray) else X_val
            y_val = cp.asarray(y_val) if isinstance(y_val, np.ndarray) else y_val
        
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
        
        # Training loop with progress bar
        progress_bar = tqdm(range(epochs)) if verbose else range(epochs)
        for epoch in progress_bar:
            # Reset gradients
            for layer in self.layers:
                layer.zero_grad()
            
            # Shuffle training data
            indices = cp.arange(X_train.shape[0])
            cp.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward and backward passes
                self.forward(X_batch, training=True)
                self.backward(y_batch)
                
                # Update weights and biases
                for idx, layer in enumerate(self.layers):
                    dW, db = layer.gradients
                    optimizer.update(layer, dW, db, idx)
            
            # Calculate metrics
            train_loss = self.calculate_loss(X_train, y_train)
            train_accuracy, _ = self.evaluate(X_train, y_train)
            
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
            if verbose:
                if isinstance(progress_bar, tqdm):
                    progress_bar.set_description(
                        f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}{val_metrics}"
                    )
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}{val_metrics}")
            
            # Learning rate scheduler step
            if lr_scheduler:
                if isinstance(lr_scheduler, lr_scheduler_plateau):
                    lr_scheduler.step(epoch, train_loss if X_val is None else val_loss)
                else:
                    lr_scheduler.step(epoch)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        for i, layer in enumerate(self.layers):
            layer.weights = best_weights[i]
            layer.biases = best_biases[i]

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
        # Convert to CuPy arrays if needed
        X = cp.asarray(X) if isinstance(X, np.ndarray) else X
        y = cp.asarray(y) if isinstance(y, np.ndarray) else y
        
        # Get predictions
        outputs = self.forward(X, training=False)
        
        # Apply class weights if provided
        if class_weights is None:
            class_weights = cp.ones_like(y)
        elif isinstance(class_weights, np.ndarray):
            class_weights = cp.asarray(class_weights)
        
        # Select appropriate loss function
        if self.is_binary:
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(outputs, y.reshape(-1, 1))
        else:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(outputs, y)
        
        # Add L2 regularization
        l2_reg = self.reg_lambda * sum(cp.sum(layer.weights**2) for layer in self.layers)
        loss += l2_reg
        
        # Convert to Python float
        return float(loss.get()) if hasattr(loss, 'get') else float(loss)

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
        # Convert to CuPy arrays if needed
        X = cp.asarray(X) if isinstance(X, np.ndarray) else X
        y = cp.asarray(y) if isinstance(y, np.ndarray) else y
        
        # Get predictions
        y_hat = self.forward(X, training=False)
        
        # Calculate accuracy based on problem type
        if self.is_binary:
            predicted = (y_hat > 0.5).astype(int)
            accuracy = float(cp.mean(predicted.flatten() == y.reshape(-1, 1).flatten()).get())
        else:
            predicted = cp.argmax(y_hat, axis=1)
            accuracy = float(cp.mean(predicted == y).get())
        
        # Convert to numpy for easier handling
        predicted_np = cp.asnumpy(predicted)
        
        return accuracy, predicted_np

    def predict(self, X):
        """
        Generate predictions for input data.
        Parameters:
            - X (ndarray): Input data
        Returns:
            - predictions: Model predictions (class probabilities or labels)
        """
        # Convert to CuPy if needed
        X = cp.asarray(X) if isinstance(X, np.ndarray) else X
        
        # Get raw predictions
        outputs = self.forward(X, training=False)
        
        # For binary classification, return class probabilities
        if self.is_binary:
            return cp.asnumpy(outputs)
        # For multiclass, return class labels
        else:
            return cp.asnumpy(cp.argmax(outputs, axis=1))

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, param_grid,
                             layer_configs=None, optimizer_types=None, 
                             lr_range=(0.0001, 0.01, 5), epochs=30, batch_size=32):
        """
        Performs hyperparameter tuning using grid search.
        Parameters:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data
            - param_grid: Dict of parameters to try
            - layer_configs: List of layer configurations
            - optimizer_types: List of optimizer types
            - lr_range: (min_lr, max_lr, num_steps) for learning rates
            - epochs: Max epochs for each trial
            - batch_size: Batch size for training
        Returns:
            - best_params: Best hyperparameters found
            - best_accuracy: Best validation accuracy
        """
        from itertools import product
        import warnings
        warnings.filterwarnings('ignore')
        
        # Convert to CuPy
        X_train = cp.asarray(X_train) if isinstance(X_train, np.ndarray) else X_train
        y_train = cp.asarray(y_train) if isinstance(y_train, np.ndarray) else y_train
        X_val = cp.asarray(X_val) if isinstance(X_val, np.ndarray) else X_val
        y_val = cp.asarray(y_val) if isinstance(y_val, np.ndarray) else y_val
        
        # Default values if not provided
        if layer_configs is None:
            layer_configs = [[64], [128], [64, 32]]
            
        if optimizer_types is None:
            optimizer_types = ['Adam', 'SGD']
            
        # Output size based on target data
        output_size = 1 if len(y_train.shape) == 1 or y_train.shape[1] == 1 else y_train.max() + 1
        
        # Generate learning rates
        min_lr, max_lr, num_steps = lr_range
        lr_options = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps).tolist()
        
        # Extract parameter combinations
        keys, values = zip(*param_grid.items())
        
        # Calculate total iterations for progress tracking
        total_iterations = (
            len(layer_configs) *
            len(lr_options) *
            len(optimizer_types) *
            np.prod([len(value) for value in values])
        )
        
        # Track best results
        best_accuracy = 0
        best_params = {}
        best_optimizer_type = None
        
        # Grid search with progress bar
        with tqdm(total=total_iterations, desc="Tuning Hyperparameters") as pbar:
            # Iterate through all combinations
            for optimizer_type in optimizer_types:
                for layer_structure in layer_configs:
                    full_layer_structure = [X_train.shape[1]] + layer_structure + [int(output_size)]
                    
                    for combination in product(*values):
                        params = dict(zip(keys, combination))
                        
                        for lr in lr_options:
                            # Create model with current hyperparameters
                            nn = NeuralNetwork(
                                full_layer_structure, 
                                dropout_rate=params['dropout_rate'], 
                                reg_lambda=params['reg_lambda']
                            )
                            
                            # Create optimizer
                            optimizer = self._create_optimizer(optimizer_type, lr)
                            
                            # Train model (with early stopping for efficiency)
                            nn.train(
                                X_train, y_train, X_val, y_val,
                                optimizer=optimizer,
                                epochs=epochs,
                                batch_size=batch_size,
                                early_stopping_patience=5,
                                verbose=False
                            )
                            
                            # Evaluate on validation set
                            accuracy, _ = nn.evaluate(X_val, y_val)
                            
                            # Update best if improved
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = {
                                    **params,
                                    'layers': full_layer_structure,
                                    'learning_rate': lr
                                }
                                best_optimizer_type = optimizer_type
                                
                                print(f"New best: {best_accuracy:.4f} with {optimizer_type}, "
                                      f"lr={lr}, layers={full_layer_structure}, params={params}")
                            
                            # Update progress
                            pbar.update(1)
        
        print(f"\nBest configuration: {best_optimizer_type} optimizer with lr={best_params['learning_rate']}")
        print(f"Layers: {best_params['layers']}")
        print(f"Parameters: dropout={best_params['dropout_rate']}, reg_lambda={best_params['reg_lambda']}")
        print(f"Validation accuracy: {best_accuracy:.4f}")
        
        return best_params, best_accuracy

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


class Layer:
    """
    Neural network layer implementation.
    Args:
        input_size (int): Input size
        output_size (int): Output size
        activation (str): Activation function name
    """
    def __init__(self, input_size, output_size, activation="relu"):
        # He initialization for weights
        if activation in ["relu", "leaky_relu"]:
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)
            
        self.weights = cp.random.randn(input_size, output_size) * scale
        self.biases = cp.zeros((1, output_size))
        self.activation = activation
        self.gradients = None
        
    def zero_grad(self):
        """Reset gradients."""
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
            return cp.ones_like(Z)  # Identity for compatibility
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


class Activation:
    """
    Neural network activation functions and their derivatives.
    """
    
    @staticmethod
    def relu(z):
        """ReLU activation function."""
        return cp.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """ReLU derivative."""
        return (z > 0).astype(cp.float32)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """Leaky ReLU activation."""
        return cp.maximum(alpha * z, z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """Leaky ReLU derivative."""
        return cp.where(z > 0, 1, alpha)

    @staticmethod
    def tanh(z):
        """Hyperbolic tangent activation."""
        return cp.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        """Tanh derivative."""
        return 1 - cp.tanh(z) ** 2

    @staticmethod
    def sigmoid(z):
        """Sigmoid activation."""
        # Clip for numerical stability
        z = cp.clip(z, -88, 88)  # Prevent overflow
        return 1 / (1 + cp.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        """Sigmoid derivative."""
        sig = Activation.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def softmax(z):
        """Softmax activation with numerical stability."""
        # Subtract max for numerical stability
        exp_z = cp.exp(z - cp.max(z, axis=1, keepdims=True))
        return exp_z / cp.sum(exp_z, axis=1, keepdims=True)