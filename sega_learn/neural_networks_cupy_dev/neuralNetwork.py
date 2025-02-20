from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer


# import numpy as np
import cupy as np

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
        self.layers = []                                                        # List to store the layers of the neural network
        self.dropout_rate = dropout_rate                                        # Dropout rate
        self.reg_lambda = reg_lambda                                            # Regularization lambda
        
        # Initialize weights
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01  # Small random weights
            bias = np.zeros((1, layer_sizes[i + 1]))  # Initialize biases to zeros
            self.weights.append(weight)
            self.biases.append(bias)
        
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['softmax']       # Default to ReLU for hidden layers and Softmax for the output layer
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], activations[i-1]))  # Create layers with input and output sizes

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

    def forward(self, X):
        """
        Performs forward propagation through the neural network.
        Args: X (ndarray): Input data of shape (batch_size, input_size).
        Returns: ndarray: Output predictions of shape (batch_size, output_size).
        """
        A = X                                                               # Input layer
        self.activations = [X]                                              # Store activations for backpropagation
        for layer in self.layers[:-1]:                                      # Loop through all layers except the last one
            Z = np.dot(A, layer.weights) + layer.biases                     # Linear transformation
            A = layer.activate(Z)                                           # Activation function
            if self.dropout_rate > 0:
                A = self.apply_dropout(A)                                   # Apply dropout regularization, if applicable
            self.activations.append(A)                                      # Store activations for backpropagation
            
        Z = np.dot(A, self.layers[-1].weights) + self.layers[-1].biases     # Output layer linear transformation
        
        # Determine the activation for the output layer based on the number of output classes
        if self.layers[-1].weights.shape[1] == 1:                            # Binary classification
            outputs = Activation.sigmoid(Z)                                  # Use sigmoid for binary classification
        else:                                                                # Multiclass classification
            outputs = Activation.softmax(Z)                                  # Use softmax for multiclass classification

        self.activations.append(outputs)                                    # Store activations for backpropagation
        
        return outputs

    def apply_dropout(self, A):
        """
        Applies dropout regularization to the input array.
        Parameters: A: numpy.ndarray: Input array to apply dropout regularization to.
        Returns: numpy.ndarray: Array with dropout regularization applied.
        """
        if self.dropout_rate > 0:                           # If dropout rate is greater than 0
            keep_prob = 1 - self.dropout_rate               # Calculate keep probability
            mask = np.random.rand(*A.shape) < keep_prob     # Create a mask for the dropout
            A = np.multiply(A, mask)                        # Apply the mask to the input array
            A /= keep_prob                                  # Scale the output of the dropout layer
        return A
 
    def backward(self, y):
        """
        Performs backward propagation to calculate the gradients of the weights and biases in the neural network.
        Parameters: y (numpy.ndarray): Target labels of shape (m, 1), where m is the number of samples.
        Returns: None
        """
        m = y.shape[0]          # Number of samples
        y = y.reshape(-1, 1)    # Reshape y to ensure it is a column vector

        outputs = self.activations[-1]                                      # Output predictions
        
        # For binary classification
        if self.layers[-1].weights.shape[1] == 1: 
            dA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15))  # Gradient for binary cross-entropy
        else:  # For multi-class classification
            dA = outputs - y  # Gradient for cross-entropy loss with softmax
        
        for i in reversed(range(len(self.layers))):
            dZ = dA * self.layers[i].activation_derivative(self.activations[i + 1]) if i < len(self.layers) - 1 else dA
            
            dW = np.dot(self.activations[i].T, dZ) / m + self.reg_lambda * self.layers[i].weights   # Gradient of the loss function
            db = np.sum(dZ, axis=0, keepdims=True) / m                                              # Bias of the loss function 
            
            dA = np.dot(dZ, self.layers[i].weights.T)                                               # dA for the next iteration

            self.layers[i].gradients = (dW, db)                                                     # Store the gradients

    def train(self, X_train, y_train, X_test=None, y_test=None, optimizer=AdamOptimizer(learning_rate=0.0001), epochs=100, batch_size=32, early_stopping_threshold=10, lr_scheduler=None, p=True):
        """
        Trains the neural network model.
        Parameters:
            - X_train (numpy.ndarray): Training data features.
            - y_train (numpy.ndarray): Training data labels.
            - X_test (numpy.ndarray): Test data features, optional (default: None).
            - y_test (numpy.ndarray): Test data labels, optional (default: None).
            - optimizer (Optimizer): The optimizer used for updating the model parameters (default: Adam, lr=0.0001).
            - epochs (int): Number of training epochs (default: 100).
            - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
            - early_stopping_threshold (int): Number of epochs to wait for improvement in training loss before early stopping (default: 5).
            - lr_scheduler (Scheduler): Learning rate scheduler (default: None).
            - p (bool): Whether to print training progress (default: True).
        Returns: None
        """
        optimizer.initialize(self.layers)   # Initialize the optimizer
        best_loss = float('inf')            # Initialize best loss to infinity
        patience = 0                        # Initialize patience for early stopping
        
        for epoch in range(epochs):
            for layer in self.layers: layer.zero_grad()             # Reset gradients to zero for each layer
            
            indices = np.arange(X_train.shape[0])                   # Create indices for shuffling
            np.random.shuffle(indices)                              # Shuffle indices
            X_train, y_train = X_train[indices], y_train[indices]   # Shuffle the training data

            for i in range(0, X_train.shape[0], batch_size):        # Loop through the training data in batches
                X_batch = X_train[i:i+batch_size]                   # X of the current batch
                y_batch = y_train[i:i+batch_size]                   # y of the current batch

                outputs = self.forward(X_batch)                     # Perform forward pass, get predictions
                self.backward(y_batch)                              # Perform backward pass, calculate gradients

                for idx, layer in enumerate(self.layers):   # For each layer in the neural network..
                    dW, db = layer.gradients                # Get the gradients, weights and biases
                    optimizer.update(layer, dW, db, idx)    # Update the weights and biases using the optimizer

            # Calculate training loss and accuracy
            train_loss = self.calculate_loss(X_train, y_train)
            train_accuracy, train_pred = self.evaluate(X_train, y_train)
            
            # Calculate test loss and accuracy
            if X_test is not None and y_test is not None:
                test_loss = self.calculate_loss(X_test, y_test)
                test_accuracy, test_pred = self.evaluate(X_test, y_test)
            
            # Print the training progress
            if p: 
                if X_test is not None and y_test is not None:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Implement early stopping
            if train_loss < best_loss:      # If training loss improves
                best_loss = train_loss      # Update best loss
                patience = 0                # Reset patience
            else:                           # If training loss does not improve
                patience += 1               # Increment patience
            if patience >= early_stopping_threshold:    # If patience exceeds the threshold, stop training
                if p: print("Early stopping triggered")       
                break
            
            if lr_scheduler:                                        # If learning rate scheduler is provided
                if isinstance(lr_scheduler, lr_scheduler_plateau):  # If plateau learning rate scheduler
                    lr_scheduler.step(epoch,train_loss)
                else:
                    lr_scheduler.step(epoch)

    def calculate_loss(self, X, y, class_weights=None):
        """
        Calculates the loss of the neural network model.
        Formula: loss = loss_fn(outputs, y) + reg_lambda * sum([sum(layer.weights**2) for layer in self.layers])
        Parameters:
            - X (numpy.ndarray): Input data of shape (num_samples, num_features).
            - y (numpy.ndarray): Target labels of shape (num_samples,).
            - class_weights (numpy.ndarray, optional): Weights for each class. Default is None.
        Returns: loss (float): The calculated loss value.
        """
        outputs = self.forward(X)               # Perform forward pass to get predictions
        if class_weights is None:
            class_weights = np.ones_like(y)     # If class weights are not provided, set them to 1
        
        if self.layers[-1].weights.shape[1] > 1:        # Multi-class classification, cross-entropy loss
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(outputs, y.reshape(-1, 1)) 
        else:                                           # Binary classification, binary cross-entropy loss
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(outputs, y)
        
        loss += self.reg_lambda * np.sum([np.sum(layer.weights**2) for layer in self.layers])   # Loss with L2 regularization
        
        return loss

    def evaluate(self, X, y):
        """
        Evaluates the performance of the neural network model on the given input data.
        Parameters:
            - X (numpy.ndarray): The input data for evaluation.
            - y (numpy.ndarray): The target labels for evaluation.
        Returns:
            - accuracy (float): The accuracy of the model's predictions.
            - predicted (numpy.ndarray): The labels predicted by the model.
        """
        y_hat = self.forward(X)                             # Perform forward pass to get predictions
        
        if self.layers[-1].weights.shape[1] > 1:            # Multi-class classification
            predicted = np.argmax(y_hat, axis=1)            # Get the class with the highest probability
            accuracy = np.mean(predicted == y)              # Calculate accuracy
        
        else:                                               # Binary classification
            predicted = (y_hat > 0.5).astype(int)           # Convert probabilities to binary predictions
            accuracy = np.mean(predicted == y.reshape(-1, 1))  # Calculate accuracy correctly for binary classification                   
        
        return accuracy, predicted                          # Return accuracy and predicted labels
    
    def tune_hyperparameters(self, param_grid, layers, output_size,
                             X_train, y_train, X_val, y_val, 
                             optimizers, lr_range, epochs=100, batch_size=32):
        """
        Performs hyperparameter tuning using grid search.
        
        Parameters:
            - param_grid (dict): A dictionary where keys are parameter names and values are lists of values to try.
            - layers (list): List of layer sizes.
            - output_size (int): The size of the output layer.
            - X_train (numpy.ndarray): Training data features.
            - y_train (numpy.ndarray): Training data labels.
            - X_val (numpy.ndarray): Validation data features.
            - y_val (numpy.ndarray): Validation data labels.
            - optimizers (list): List of optimizer types to try (e.g., ['Adam', 'SGD', 'Adadelta']).
            - lr_range (tuple): A tuple (min_lr, max_lr, num_steps) for learning rates.
            - epochs (int): Number of training epochs (default: 100).
            - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
        
        Returns:
            best_params (dict): The best hyperparameters found during tuning.
            best_accuracy (float): The best validation accuracy achieved.
        """
        from itertools import product
        from tqdm import tqdm
        import warnings
        warnings.filterwarnings('ignore')       # Suppress warnings, large logits will trigger overflow warnings

        best_accuracy = 0
        best_params = {}

        keys, values = zip(*param_grid.items()) # Unzip the parameter grid
        
        # Generate learning rate options
        min_lr, max_lr, num_steps = lr_range
        lr_options = np.linspace(min_lr, max_lr, num_steps).tolist()
        
        # Calculate total iterations
        total_iterations = (len(layers) *
                            len(lr_options) *
                            len(optimizers) *
                            np.prod([len(value) for value in values])
                            )
        
        with tqdm(total=total_iterations, desc="Tuning Hyperparameters") as pbar:
            for optimizer_type in optimizers:                   # For each optimizer type
                for layer_structure in layers:                  # For each layer structure
                    layer_structure = [X_train.shape[1]] + layer_structure + [output_size]
                    
                    for combination in product(*values):
                        params = dict(zip(keys, combination))
                        
                        for lr in lr_options:
                            # print(f"\nTesting combination: {params} with layers {layer_structure} and learning rate {lr}")

                            # Initialize the neural network with the current combination of hyperparameters
                            nn = NeuralNetwork(layer_structure, dropout_rate=params['dropout_rate'], reg_lambda=params['reg_lambda'])

                            # Define the optimizer
                            optimizer = self.create_optimizer(optimizer_type, lr)
                            
                            # Train the neural network
                            nn.train(X_train, y_train, X_val, y_val, optimizer, epochs=epochs, batch_size=batch_size, p=False)

                            # Evaluate on validation set
                            accuracy, _ = nn.evaluate(X_val, y_val)

                            # print(f"\t accuracy: {accuracy}")

                            # Check if this is the best accuracy
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                print(f"\n--New best accuracy: {best_accuracy:.4f} with combination: {optimizer_type}, {params} with layers {layer_structure} and learning rate {lr}")
                                
                                best_params = {**params, 'layers': layer_structure, 'learning_rate': lr, 'optimizer': optimizer}
                                best_optimizer = optimizer_type
                            
                            pbar.update(1)  # Update the progress bar after each combination

        print(f"\nBest parameters: Optimizer: {best_optimizer}, {best_params} with accuracy: {best_accuracy:.4f}")
        return best_params, best_accuracy


    def create_optimizer(self, optimizer_type, learning_rate):
        """
        Creates an optimizer instance based on the specified type and learning rate.
        
        Parameters:
            optimizer_type (str): The type of optimizer (e.g., 'SGD', 'Adam').
            learning_rate (float): The learning rate for the optimizer.
        
        Returns:
            optimizer: An instance of the specified optimizer.
        """
        if optimizer_type == 'Adam':
            return AdamOptimizer(learning_rate)
        elif optimizer_type == 'SGD':
            return SGDOptimizer(learning_rate)
        elif optimizer_type == 'Adadelta':
            return AdadeltaOptimizer(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
    def create_scheduler(self, scheduler_type, optimizer, **kwargs):
        """
        Creates a learning rate scheduler instance based on the specified type and parameters.
        
        Parameters:
            scheduler_type (str): The type of scheduler (e.g., 'step', 'plateau', 'exp').
            optimizer: The optimizer instance to be used with the scheduler.
            **kwargs: Additional parameters for the scheduler.
        
        Returns:
            scheduler: An instance of the specified learning rate scheduler.
        """
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
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Initialize weights with small random values
        self.biases = np.zeros((1, output_size))                        # Initialize biases with zeros
        self.activation = activation                                    # Activation function name
        self.gradients = None                                           # Initialize gradients to None
        
    def zero_grad(self):
        """Reset the gradients of the weights and biases to zero."""
        self.gradients = None

    def activate(self, Z):
        """Apply the activation function based on the layer's configuration."""
        
        if self.activation == "relu":
            return Activation.relu(Z)
        elif self.activation == "leaky_relu":
            return Activation.leaky_relu(Z)
        elif self.activation == "tanh":
            return Activation.tanh(Z)
        elif self.activation == "sigmoid":
            return Activation.sigmoid(Z)
        elif self.activation == "softmax":
            return Activation.softmax(Z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def activation_derivative(self, Z):
        """Apply the derivative of the activation function for backpropagation."""
        
        if self.activation == "relu":
            return Activation.relu_derivative(Z)
        elif self.activation == "leaky_relu":
            return Activation.leaky_relu_derivative(Z)
        elif self.activation == "tanh":
            return Activation.tanh_derivative(Z)
        elif self.activation == "sigmoid":
            return Activation.sigmoid_derivative(Z)
        elif self.activation == "softmax":
            # Special case: softmax derivative is typically used in combination with cross-entropy loss.
            # Cross-entropy takes care of derivative, so no need to implement softmax derivative here.
            raise ValueError("Softmax derivative is not typically used directly")
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

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
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """
        Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0
        Returns 1 for positive input, and 0 for negative input.
        """
        return (z > 0).astype(float)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """
        Leaky ReLU activation function: f(z) = z if z > 0, else alpha * z
        Allows a small, non-zero gradient when the input is negative to address the dying ReLU problem.
        """
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """
        Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha
        Returns 1 for positive input, and alpha for negative input.
        """
        return np.where(z > 0, 1, alpha)

    @staticmethod
    def tanh(z):
        """
        Hyperbolic tangent (tanh) activation function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        Maps input to the range [-1, 1], typically used for normalized input.
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        """
        Derivative of the tanh function: f'(z) = 1 - tanh(z)^2
        Used for backpropagation through the tanh activation.
        """
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function: f(z) = 1 / (1 + exp(-z))
        Maps input to the range [0, 1], commonly used for binary classification.
        """
        return 1 / (1 + np.exp(-z))

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
        exp_logits = np.exp(z - np.max(z, axis=1, keepdims=True))  
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


