from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer

import numpy as np
import cupy as cp
from cupy import fuse
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
            
        # Initialize weights and biases (for reference, not used in layer updates)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            weight = cp.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01  # Small random weights
            bias = cp.zeros((1, layer_sizes[i + 1]))  # Initialize biases to zeros
            self.weights.append(weight)
            self.biases.append(bias)
            
        # Cache for forward/backward pass
        self.layer_outputs = None
        self.is_binary = layer_sizes[-1] == 1

        # Initialize CUDA stream for asynchronous operations
        self.stream = cp.cuda.Stream()
        
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

    @staticmethod
    @fuse()  # Fused kernel for dropout
    def fused_dropout(x, dropout_rate, random_vals):
        # For each element, if the random value is less than (1 - dropout_rate),
        # return the scaled activation; otherwise return 0.
        return cp.where(random_vals < (1 - dropout_rate), x / (1 - dropout_rate), 0)
    
    def apply_dropout(self, X):
        # Pre-generate random values and apply fused dropout
        random_vals = cp.random.rand(*X.shape)
        return self.fused_dropout(X, self.dropout_rate, random_vals)

    def forward(self, X, training=True):
        """
        Performs forward propagation through the neural network.
        Args: 
            X (ndarray): Input data of shape (batch_size, input_size).
            training (bool): Whether the network is in training mode (applies dropout).
        Returns: 
            ndarray: Output predictions of shape (batch_size, output_size).
        """
        # # Convert numpy array to CuPy if needed
        # if isinstance(X, np.ndarray): X = cp.asarray(X)

        with self.stream:
            # Store all layer activations for backprop
            self.layer_outputs = [X]
            
            # Forward pass through all layers except the last
            A = X
            for i, layer in enumerate(self.layers[:-1]):
                Z = cp.dot(A, layer.weights) + layer.biases
                A = layer.activate(Z)
                
                # Apply dropout only during training
                if training and self.dropout_rate > 0:
                    A = self.apply_dropout(A)
                    
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
        # # Convert numpy array to CuPy if needed
        # if isinstance(y, np.ndarray): y = cp.asarray(y)

        with self.stream:
            m = y.shape[0]  # Number of samples
            
            # Reshape y for binary classification
            if self.is_binary:
                y = y.reshape(-1, 1)
            else:
                # One-hot encode y for multi-class classification
                y = cp.eye(self.layer_sizes[-1])[y]
                
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
                
                # Update preallocated gradient buffers in the layer
                self.layers[i].grad_dW[:] = dW
                self.layers[i].grad_db[:] = db
                
                # Calculate dA for next iteration (previous layer)
                if i > 0:  # No need to calculate for input layer
                    dA = cp.dot(dZ, self.layers[i].weights.T)

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
        # Convert numpy arrays to CuPy arrays
        if isinstance(X_train, np.ndarray): X_train = cp.asarray(X_train)
        if isinstance(y_train, np.ndarray): y_train = cp.asarray(y_train)
        if X_val is not None and isinstance(X_val, np.ndarray): X_val = cp.asarray(X_val)
        if y_val is not None and isinstance(y_val, np.ndarray): y_val = cp.asarray(y_val)

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
        progress_bar = tqdm(range(epochs)) if use_tqdm else range(epochs)
        for epoch in progress_bar:
            # Reset preallocated gradients on each layer
            for layer in self.layers:
                layer.zero_grad()
            
            # Shuffle training data (using CuPy for GPU-based shuffling)
            indices = cp.arange(X_train.shape[0])
            cp.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Process batches entirely on the GPU without intermediate synchronizations
            for start_idx in range(0, X_train.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X_train.shape[0])
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                self.forward(X_batch, training=True)
                self.backward(y_batch)
                
                # Use the preallocated gradients for the update step.
                for layer in self.layers:
                    optimizer.update(layer, layer.grad_dW, layer.grad_db)
            
            # Synchronize GPU stream to ensure all operations are complete
            self.stream.synchronize()
            
            # Calculate metrics (avoid CPU transfers except for final values)
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
        # # Convert numpy array to CuPy if needed
        # if isinstance(X, np.ndarray): X = cp.asarray(X)
        # if isinstance(y, np.ndarray): y = cp.asarray(y)

        with self.stream:
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
            
            # Synchronize before getting the value
            self.stream.synchronize()
            
            # Convert to Python float (only at the final step)
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
        # # Convert numpy array to CuPy if needed
        # if isinstance(X, np.ndarray): X = cp.asarray(X)
        # if isinstance(y, np.ndarray): y = cp.asarray(y)

        with self.stream:
            # Get predictions
            y_hat = self.forward(X, training=False)
            
            # Calculate accuracy based on problem type
            if self.is_binary:
                predicted = (y_hat > 0.5).astype(int)
                accuracy = cp.mean(predicted.flatten() == y.reshape(-1, 1).flatten())
            else:
                predicted = cp.argmax(y_hat, axis=1)
                accuracy = cp.mean(predicted == y)
            
            # Synchronize before getting the value
            self.stream.synchronize()
            
            # Convert to Python float (only at the final step)
            return float(accuracy.get()) if hasattr(accuracy, 'get') else float(accuracy), predicted
              
    def predict(self, X):
        """
        Generate predictions for input data.
        Parameters:
            - X (ndarray): Input data
        Returns:
            - predictions: Model predictions (class probabilities or labels)
        """
        with self.stream:
            # Get raw predictions
            outputs = self.forward(X, training=False)
            
            # For binary classification, return class probabilities
            if self.is_binary:
                return outputs
            # For multiclass, return class labels
            else:
                return cp.argmax(outputs, axis=1)
            

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
    Initializes a Layer object.
    Args:
        input_size (int): The size of the input to the layer.
        output_size (int): The size of the output from the layer.
        activation (str): The activation function to be used in the layer.
    """
    def __init__(self, input_size, output_size, activation="relu"):
        # He initialization for weights
        if activation in ["relu", "leaky_relu"]:
            scale = cp.sqrt(2.0 / input_size)
        else:
            scale = cp.sqrt(1.0 / input_size)
            
        self.weights = cp.random.randn(input_size, output_size) * scale
        self.biases = cp.zeros((1, output_size))
        self.activation = activation
        # Preallocate gradient buffers to avoid repeated allocations.
        self.grad_dW = cp.zeros_like(self.weights)
        self.grad_db = cp.zeros_like(self.biases)
        
    def zero_grad(self):
        """Reset the gradients of the weights and biases to zero."""
        self.grad_dW[...] = 0
        self.grad_db[...] = 0

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
    This class contains various activation functions and their corresponding derivatives for use in neural networks.
    relu: Rectified Linear Unit activation function. Returns the input directly if it's positive, otherwise returns 0.
    leaky_relu: Leaky ReLU activation function. A variant of ReLU that allows a small gradient when the input is negative. 
    tanh: Hyperbolic tangent activation function. Maps input to range [-1, 1]. Commonly used for normalized input.
    sigmoid: Sigmoid activation function. Maps input to range [0, 1]. Commonly used for binary classification.
    softmax: Softmax activation function. Maps input into a probability distribution over multiple classes.
    """
    
    @staticmethod
    def relu(z):
        """
        ReLU (Rectified Linear Unit) activation function: f(z) = max(0, z)
        Returns the input directly if it's positive, otherwise returns 0.
        """
        return cp.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """
        Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0
        Returns 1 for positive input, and 0 for negative input.
        """
        return (z > 0).astype(cp.float32)  

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """
        Leaky ReLU activation function: f(z) = z if z > 0, else alpha * z
        Allows a small, non-zero gradient when the input is negative to address the dying ReLU problem.
        """
        return cp.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """
        Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha
        Returns 1 for positive input, and alpha for negative input.
        """
        return cp.where(z > 0, 1, alpha)

    @staticmethod
    def tanh(z):
        """
        Hyperbolic tangent (tanh) activation function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        Maps input to the range [-1, 1], typically used for normalized input.
        """
        return cp.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        """
        Derivative of the tanh function: f'(z) = 1 - tanh(z)^2
        Used for backpropagation through the tanh activation.
        """
        return 1 - cp.tanh(z) ** 2

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function: f(z) = 1 / (1 + exp(-z))
        Maps input to the range [0, 1], commonly used for binary classification.
        """
        return 1 / (1 + cp.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        """
        Derivative of the sigmoid function: f'(z) = sigmoid(z) * (1 - sigmoid(z))
        Used for backpropagation through the sigmoid activation.
        """
        sig = Activation.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def softmax(z):
        """
        Softmax activation function: f(z)_i = exp(z_i) / sum(exp(z_j)) for all j
        Maps input into a probability distribution over multiple classes. Used for multiclass classification.
        """
        # Subtract the max value from each row to prevent overflow (numerical stability)
        exp_z = cp.exp(z - cp.max(z, axis=1, keepdims=True))
        return exp_z / cp.sum(exp_z, axis=1, keepdims=True)

