from re import L
from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer

import numpy as np
from joblib import Parallel, delayed
from numba import jit, njit, vectorize, float64, int32, prange
from numba import types
from numba.experimental import jitclass
from tqdm.auto import tqdm

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
        
    def __repr__(self):
        layers = ""
        for i in range(len(self.layers)):
            layers += f"\n\tLayer {i}: {self.layers[i].weights.shape[0]} neurons with {self.layers[i].weights.shape[1]} weights"
        
        return f"NeuralNetwork(\nlayer_sizes={self.layer_sizes}, \nlayers={layers}, \ndropout_rate={self.dropout_rate}, \nreg_lambda={self.reg_lambda}, \nweights={self.weights}, \nbiases={self.biases}, \nactivations={self.activations})"

    def __str__(self):
        layers = ""
        for i in range(len(self.layers)):
            layers += f"\n\tLayer {i}: {self.layers[i].weights.shape[0]} neurons with {self.layers[i].weights.shape[1]} weights"
        return f"Neural Network with layer sizes {self.layer_sizes}, \nlayer details: {layers}, \ndropout rate: {self.dropout_rate}, \nregularization lambda: {self.reg_lambda}"

    def apply_dropout(self, X):
        """
        Applies dropout to the activation X.
        Args:
            X (ndarray): Activation values.
        Returns:
            ndarray: Activation values after applying dropout.
        """
        # mask = np.random.rand(*X.shape) < (1 - self.dropout_rate)
        # return np.multiply(X, mask) / (1 - self.dropout_rate)
        
        # Use JIT-compiled function for performance
        return self._apply_dropout_jit(X, self.dropout_rate)
    
    @staticmethod
    @njit
    def _apply_dropout_jit(X, dropout_rate):
        """
        Numba JIT-compiled function to apply dropout.
        Args:
            X (ndarray): Activation values.
            dropout_rate (float): Dropout rate.
        Returns:
            ndarray: Activation values after applying dropout.
        """
        return np.multiply(X, np.random.rand(*X.shape) < (1 - dropout_rate)) / (1 - dropout_rate)

    def forward(self, X, training=True):
        """
        Performs forward propagation through the neural network.
        Args: 
            X (ndarray): Input data of shape (batch_size, input_size).
            training (bool): Whether the network is in training mode (applies dropout).
        Returns: 
            ndarray: Output predictions of shape (batch_size, output_size).
        """
        # # Store all layer activations for backprop
        # self.layer_outputs = [X]
        
        # # Forward pass through all layers except the last
        # A = X
        # for i, layer in enumerate(self.layers[:-1]):
        #     Z = np.dot(A, layer.weights) + layer.biases
        #     A = layer.activate(Z)
            
        #     # Apply dropout only during training
        #     if training and self.dropout_rate > 0:
        #         A = self.apply_dropout(A)
                
        #     self.layer_outputs.append(A)
        
        # # Last layer (output layer)
        # Z = np.dot(A, self.layers[-1].weights) + self.layers[-1].biases
        
        # # Apply appropriate activation for output layer
        # if self.is_binary:
        #     output = sigmoid(Z)
        # else:
        #     output = softmax(Z)
            
        # self.layer_outputs.append(output)
               
        self.layer_outputs = [X]
        weights = [layer.weights for layer in self.layers]
        biases = [layer.biases for layer in self.layers]
        layer_outputs = self._forward_jit(X, weights, biases, self.activations, self.dropout_rate, training, self.is_binary)
        self.layer_outputs = layer_outputs  # Update layer_outputs with the correct shapes
        
        return self.layer_outputs[-1]
            
    @staticmethod
    @njit
    def _forward_jit(X, weights, biases, activations, dropout_rate, training, is_binary):
        # Store all layer activations for backprop
        layer_outputs = [X]
        
        # Forward pass through all layers except the last
        for i in range(len(weights) - 1):
            # Calculate linear transformation
            Z = np.dot(X, weights[i]) + biases[i]
            
            # Apply activation function
            if activations[i] == "relu":
                X = relu(Z)
            elif activations[i] == "leaky_relu":
                X = leaky_relu(Z)
            elif activations[i] == "tanh":
                X = tanh(Z)
            elif activations[i] == "sigmoid":
                X = sigmoid(Z)
            elif activations[i] == "softmax":
                X = softmax(Z)
            else:
                raise ValueError(f"Unsupported activation: {activations[i]}")
            
            # Apply dropout only during training
            if training and dropout_rate > 0:
                X = np.multiply(X, np.random.rand(*X.shape) < (1 - dropout_rate)) / (1 - dropout_rate)
            
            layer_outputs.append(X)
        
        # Last layer (output layer)
        Z = np.dot(X, weights[-1]) + biases[-1]
        if is_binary:
            output = sigmoid(Z)
        else:
            output = softmax(Z)
            
        layer_outputs.append(output)
        return layer_outputs

 
    def backward(self, y):
        """
        Performs backward propagation to calculate the gradients.
        Parameters: 
            y (ndarray): Target labels of shape (m, output_size).
        """
        m = y.shape[0]  # Number of samples
        
        # Reshape y for binary classification
        if self.is_binary:
            y = y.reshape(-1, 1)
        else:
            # One-hot encode y for multi-class classification
            y = np.eye(self.layer_sizes[-1])[y]
            
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
            dW = np.dot(prev_activation.T, dZ) / m
            
            # Add L2 regularization
            dW += self.reg_lambda * self.layers[i].weights
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Store gradients in layer
            self.layers[i].weight_gradients = dW
            self.layers[i].bias_gradients = db
            
            # Calculate dA for next iteration (previous layer)
            if i > 0:  # No need to calculate for input layer
                dA = np.dot(dZ, self.layers[i].weights.T)

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
        
        def process_batch(start_idx):
            X_batch = X_shuffled[start_idx:start_idx+batch_size]
            y_batch = y_shuffled[start_idx:start_idx+batch_size]
            
            # Forward and backward passes
            self.forward(X_batch, training=True)
            self.backward(y_batch)
            
            # Update weights and biases
            for idx, layer in enumerate(self.layers):
                dW = layer.weight_gradients
                db = layer.bias_gradients
                # Apply optimizer update
                optimizer.update(layer, dW, db, idx)
        
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
            
            # Mini-batch training with parallel processing
            Parallel(n_jobs=n_jobs)(delayed(process_batch)(i) for i in range(0, X_train.shape[0], batch_size))
            
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
        
        # Apply class weights if provided
        if class_weights is None:
            class_weights = np.ones_like(y)
        elif isinstance(class_weights, np.ndarray):
            class_weights = np.asarray(class_weights)
        
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
        l2_reg = self.reg_lambda * sum(np.sum(layer.weights**2) for layer in self.layers)
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
        
        # Calculate accuracy based on problem type
        if self.is_binary:
            predicted = (y_hat > 0.5).astype(int)
            accuracy = float(np.mean(predicted.flatten() == y.reshape(-1, 1).flatten()))
        else:
            predicted = np.argmax(y_hat, axis=1)
            accuracy = float(np.mean(predicted == y))
              
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
                                early_stopping_threshold=5,
                                p=False
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
                                
                                tqdm.write(f"New best: {best_accuracy:.4f} with {optimizer_type}, "
                                      f"lr={lr}, layers={full_layer_structure}, params={params}")
                            
                            # Update progress
                            pbar.update(1)
        
        print(f"\nBest configuration: {best_optimizer_type} optimizer with lr={best_params['learning_rate']}")
        print(f"Layers: {best_params['layers']}")
        print(f"Parameters: dropout={best_params['dropout_rate']}, reg_lambda={best_params['reg_lambda']}")
        print(f"Validation accuracy: {best_accuracy:.4f}")
        
        # Add best optimizer type to best_params
        best_params['optimizer'] = best_optimizer_type
        
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
    

# Activation functions and their derivatives
# Using Numba for JIT compilation to speed up the activation functions
@njit
def relu(z):
    return np.maximum(0, z)

@njit
def relu_derivative(z):
    return (z > 0).astype(np.float64)  # Ensure return type is float64

@njit
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

@njit
def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha).astype(np.float64)  # Ensure return type is float64

@njit
def tanh(z):
    return np.tanh(z)

@njit
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

@njit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@njit
def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

@njit
def softmax(z):
    max_z = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        max_z[i] = np.max(z[i])
    exp_z = np.empty_like(z)
    for i in range(z.shape[0]):
        exp_z[i] = np.exp(z[i] - max_z[i])
    sum_exp_z = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        sum_exp_z[i] = np.sum(exp_z[i])
    for i in range(z.shape[0]):
        exp_z[i] /= sum_exp_z[i]
    return exp_z