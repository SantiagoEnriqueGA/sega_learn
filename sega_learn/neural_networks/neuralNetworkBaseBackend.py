from .neuralNetworkBase import NeuralNetworkBase
from .layers import *
from .activations import Activation
from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer

import warnings
import numpy as np
from joblib import Parallel, delayed

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except:
    TQDM_AVAILABLE = False

class BaseBackendNeuralNetwork(NeuralNetworkBase):
    def __init__(self, layers, dropout_rate=0.2, reg_lambda=0.01, activations=None):
        """
        Initializes the Numba backend neural network.
        Args:
            layers (list): List of layer sizes or Layer objects.
            dropout_rate (float): Dropout rate for regularization.
            reg_lambda (float): L2 regularization parameter.
            activations (list): List of activation functions for each layer.
        """
        super().__init__(layers, dropout_rate, reg_lambda, activations)
        
        # if layers are empty list, initialize them
        if len(self.layers) == 0:
            self.initialize_new_layers()          
        
    def initialize_new_layers(self):
        """
        Initializes the layers of the neural network.
        Each layer is created with the specified number of neurons and activation function.
        """
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(DenseLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.activations[i]))
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X, training=True):
        """
        Performs forward propagation through the neural network.
        Args: 
            X (ndarray): Input data of shape (batch_size, input_size).
            training (bool): Whether the network is in training mode (applies dropout).
        Returns: 
            ndarray: Output predictions of shape (batch_size, output_size).
        """
        # Store the outputs of each layer for backpropagation
        self.layer_outputs = [X]
        A = X
        
        # Pass through each layer
        for i, layer in enumerate(self.layers):
            A = layer.forward(A)
            # Apply dropout only to DenseLayer outputs during training
            if training and self.dropout_rate > 0 and isinstance(layer, DenseLayer):
                A = self.apply_dropout(A)
            self.layer_outputs.append(A)
        
        # Final output handling (sigmoid for binary, softmax for multi-class)
        if self.is_binary:
            return Activation.sigmoid(A)
        else:
            return Activation.softmax(A) if A.ndim > 1 else A  # Softmax if multi-class and 2D output

    def backward(self, y):
        """
        Performs backward propagation to calculate the gradients.
        Parameters: 
            y (ndarray): Target labels of shape (m, output_size).
        """
        outputs = self.layer_outputs[-1]
        
        # If binary classification, calculate the gradient for binary cross-entropy
        if self.is_binary:
            y = y.reshape(-1, 1)
            dA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15))
        # If multi-class classification, calculate the gradient for categorical cross-entropy
        else:
            if y.ndim == 1:     # Assume one-hot encoding
                y = np.eye(self.layer_sizes[-1])[y]
            dA = outputs - y
            
        # Backpropagation through the network in reverse order
        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backward(dA, self.reg_lambda)

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              optimizer=None, epochs=100, batch_size=32, 
              early_stopping_threshold=10, lr_scheduler=None, 
              p=True, use_tqdm=True, n_jobs=1, 
              track_metrics=False, track_adv_metrics=False, 
              save_animation=False, save_path='training_animation.mp4', 
              fps=1, dpi=100, frame_every=1):
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
            - early_stopping_threshold (int): Patience for early stopping (default: 10).
            - lr_scheduler (Scheduler): Learning rate scheduler (default: None).
            - p (bool): Whether to print training progress (default: True).
            - use_tqdm (bool): Whether to use tqdm for progress bar (default: True).
            - n_jobs (int): Number of jobs for parallel processing (default: 1).
            - track_metrics (bool): Whether to track training metrics (default: False).
            - track_adv_metrics (bool): Whether to track advanced metrics (default: False).
            - save_animation (bool): Whether to save the animation of metrics (default: False).
            - save_path (str): Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
            - fps (int): Frames per second for the saved animation (default: 1).
            - dpi (int): DPI for the saved animation (default: 100).
            - frame_every (int): Capture frame every N epochs (to reduce file size) (default: 1).
        """
        if use_tqdm and not TQDM_AVAILABLE:
            warnings.warn("TQDM is not available. Disabling progress bar.", UserWarning, stacklevel=2)
            use_tqdm = False
        
        # If track_adv_metrics is True, X_val and y_val must be provided
        if track_adv_metrics and (X_val is None or y_val is None):
            track_adv_metrics = False
            warnings.warn("track_adv_metrics is True but X_val and y_val are not provided. Disabling track_adv_metrics.", UserWarning, stacklevel=2)

        # If track_adv_metrics is True, set track_metrics to True
        if track_adv_metrics: track_metrics = True
    
        # If save_animation is True but track_metrics is False, set track_metrics to True, try to set track_adv_metrics to True
        if save_animation and not track_metrics:
            track_metrics = True
            if (X_val is not None) and (y_val is not None):
                track_adv_metrics = True        

        if save_animation:
            import os
            from .animation import TrainingAnimator

            # Animation metrics to track
            metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
            if lr_scheduler:
                metrics.append('learning_rate')

            # Initialize the animator
            animator = TrainingAnimator(figure_size=(18, 10), dpi=dpi)

            # Initialize animator with metrics
            animator.initialize(metrics, has_validation=(X_val is not None))
            
            # Setup the training video capture with error handling
            try:
                # Ensure directory exists
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    
                animator.setup_training_video(save_path, fps=fps, dpi=dpi)
            except Exception as e:
                print(f"Failed to setup animation: {str(e)}")
                print("Training will continue without animation capture.")
                save_animation = False

        # Default optimizer if not provided
        if optimizer is None:
            optimizer = AdamOptimizer(learning_rate=0.0001)
            
        # Initialize optimizer
        optimizer.initialize(self.layers)
        
        # Track best model for early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_weights = [layer.weights.copy() for layer in self.layers if hasattr(layer, 'weights')]
        best_biases = [layer.biases.copy() for layer in self.layers if hasattr(layer, 'biases')]
        
        # Set metrics to track
        if track_metrics:
            self.train_loss = []
            self.train_accuracy = []
            self.learning_rates = []
            if X_val is not None:
                self.val_loss = []
                self.val_accuracy = []
        
        # Set advanced metrics to track
        if track_adv_metrics:
            self.train_precision = []
            self.train_recall = []
            self.train_f1 = []
            if X_val is not None:
                self.val_precision = []
                self.val_recall = []
                self.val_f1 = []
        
        def process_batch(start_idx):
            X_batch = X_shuffled[start_idx:start_idx+batch_size]
            y_batch = y_shuffled[start_idx:start_idx+batch_size]
            
            # Forward and backward passes
            self.forward(X_batch, training=True)
            self.backward(y_batch)
            
            # Update weights and biases
            trainable_idx = 0  # Separate index for trainable layers
            for layer in self.layers:
                if hasattr(layer, 'weight_gradients') and layer.weight_gradients is not None:
                    dW = layer.weight_gradients
                    db = layer.bias_gradients
                    optimizer.update(layer, dW, db, trainable_idx)
                    trainable_idx += 1
        
        # Training loop with progress bar
        progress_bar = tqdm(range(epochs)) if use_tqdm else range(epochs)
        for epoch in progress_bar:
            # Reset gradients
            for layer in self.layers:
                if hasattr(layer, 'zero_grad'):
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
                    best_weights = [layer.weights.copy() for layer in self.layers if hasattr(layer, 'weights')]
                    best_biases = [layer.biases.copy() for layer in self.layers if hasattr(layer, 'biases')]
                else:
                    patience_counter += 1
            else:
                # Use training loss for early stopping if no validation set
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                    best_weights = [layer.weights.copy() for layer in self.layers if hasattr(layer, 'weights')]
                    best_biases = [layer.biases.copy() for layer in self.layers if hasattr(layer, 'biases')]
                else:
                    patience_counter += 1
                    
            # Store metrics
            if track_metrics:
                self.train_loss.append(train_loss)
                self.train_accuracy.append(train_accuracy)
                self.learning_rates.append(optimizer.learning_rate)
                if X_val is not None:
                    self.val_loss.append(val_loss)
                    self.val_accuracy.append(val_accuracy)
            
            # Store advanced metrics
            if track_adv_metrics:
                train_precision, train_recall, train_f1 = self.calculate_precision_recall_f1(X_train, y_train)
                self.train_precision.append(train_precision)
                self.train_recall.append(train_recall)
                self.train_f1.append(train_f1)
                if X_val is not None:
                    val_precision, val_recall, val_f1 = self.calculate_precision_recall_f1(X_val, y_val)
                    self.val_precision.append(val_precision)
                    self.val_recall.append(val_recall)
                    self.val_f1.append(val_f1)
            
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
                    lr = optimizer.learning_rate
                    if use_tqdm and p and msg:
                        tqdm.write(msg)
                    elif p and msg:
                        print(msg)
                else:
                    msg = lr_scheduler.step(epoch)
                    lr = optimizer.learning_rate
                    if use_tqdm and p and msg:
                        tqdm.write(msg)
                    elif p and msg:
                        print(msg)                    
            
            # Update animator with metrics
            if save_animation:
                train_metrics = {
                    'loss': train_loss,
                    'accuracy': train_accuracy,
                    'precision': train_precision,
                    'recall': train_recall,
                    'f1': train_f1
                }
                if lr: train_metrics['learning_rate'] = lr
                animator.update_metrics(train_metrics, validation=False)
                val_metrics = {
                    'loss': val_loss,
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1
                }
                animator.update_metrics(val_metrics, validation=True)

                # Add frame to the animation if needed
                if epoch % frame_every == 0 or epoch == epochs - 1:
                    try:
                        animator.add_training_frame()
                    except Exception as e:
                        print(f"Failed to add animation frame: {str(e)}")
                        save_animation = False

            # Early stopping
            if patience_counter >= early_stopping_threshold:
                # Capture final frame if not already captured and if animation is enabled
                if save_animation and epoch % frame_every != 0:
                    try:
                        animator.add_training_frame()
                    except Exception:
                        pass
                
                # Stop training
                if p and use_tqdm:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                elif p:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        trainable_layers = [l for l in self.layers if hasattr(l, 'weights')]
        for i, layer in enumerate(trainable_layers):
            layer.weights = best_weights[i]
            layer.biases = best_biases[i]

        # Finish and save the animation if enabled
        if save_animation:
            try:
                animator.finish_training_video()
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Failed to finish animation: {str(e)}")

                # Alternative: generate static plot
                try:
                    static_plot_path = save_path.rsplit('.', 1)[0] + '.png'
                    self.plot_metrics(save_dir=static_plot_path)
                    print(f"Static metrics plot saved to {static_plot_path} instead")
                except Exception:
                    pass
        else:
            # Generate static plot as fallback
            try:
                static_plot_path = save_path.rsplit('.', 1)[0] + '.png'
                self.plot_metrics(save_dir=static_plot_path)
                print(f"Static metrics plot saved to {static_plot_path}")
            except Exception as e:
                print(f"Failed to save static plot: {str(e)}")

        # Return the animator for further use if needed
        return animator if save_animation else None

    def evaluate(self, X, y):
        y_hat = self.forward(X, training=False)
        if self.is_binary:
            predicted = (y_hat > 0.5).astype(int)
            predicted = predicted.reshape(y.shape)
            accuracy = float(np.mean(predicted.flatten() == y.reshape(-1, 1).flatten()))
        else:
            predicted = np.argmax(y_hat, axis=1)
            accuracy = float(np.mean(predicted == y))
        return accuracy, predicted

    def predict(self, X):
        outputs = self.forward(X, training=False)
        return outputs if self.is_binary else np.argmax(outputs, axis=1)

    def calculate_loss(self, X, y):
        """
        Calculates the loss with L2 regularization.
        Parameters:
            - X (ndarray): Input data
            - y (ndarray): Target labels
        Returns: 
            float: The calculated loss value
        """
        # Get the output of the network (forward pass w/o dropout)
        outputs = self.forward(X, training=False)
        
        # For binary classification, use binary cross-entropy loss
        if self.is_binary:
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(outputs, y.reshape(-1, 1))
        # For multi-class classification, use categorical cross-entropy loss
        else:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(outputs, y)
        
        # Add L2 regularization term to the loss
        l2_reg = self.reg_lambda * self.compute_l2_reg([layer.weights for layer in self.layers if hasattr(layer, 'weights')])
        loss += l2_reg
        return float(loss)
    
    def _create_optimizer(self, optimizer_type, learning_rate, JIT=False):
        """Helper method to create optimizer instances."""
        if optimizer_type == 'Adam':
            return AdamOptimizer(learning_rate)
        elif optimizer_type == 'SGD':
            return SGDOptimizer(learning_rate)
        elif optimizer_type == 'Adadelta':
            return AdadeltaOptimizer(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
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
        if not TQDM_AVAILABLE:
            # TODO: Make tqdm optional for hyperparameter tuning
            raise ImportError("TQDM is currently required for hyperparameter tuning. Please install it using 'pip install tqdm'.")
        
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
                            nn = BaseBackendNeuralNetwork(
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
                                use_tqdm=False,
                                p=False,
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
        
    def train_with_animation_capture(self, X_train, y_train, X_val=None, y_val=None, 
                                optimizer=None, epochs=100, batch_size=32, 
                                early_stopping_threshold=10, lr_scheduler=None,
                                save_path='training_animation.mp4', fps=1, dpi=100, frame_every=1):
        """
        Trains the neural network model while capturing training metrics in real-time animation.
        
        Parameters:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data (optional)
            - optimizer: Optimizer for updating parameters
            - epochs: Number of training epochs
            - batch_size: Batch size for mini-batch gradient descent
            - early_stopping_threshold: Patience for early stopping
            - lr_scheduler: Learning rate scheduler
            - save_path: Path to save the animation file
            - fps: Frames per second for the saved animation
            - dpi: DPI for the saved animation
            - writer: Animation writer ('ffmpeg', 'pillow', etc.)
            - frame_every: Capture frame every N epochs (to reduce file size)
            
        Returns:
            - None
        """
        import os
        from .animation import TrainingAnimator
        
        # Initialize the animator
        animator = TrainingAnimator(figure_size=(18, 10), dpi=dpi)
        
        # Define metrics to track
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
        if lr_scheduler:
            metrics.append('learning_rate')
        
        # Initialize animator with metrics
        animator.initialize(metrics, has_validation=(X_val is not None))
        
        # Setup the training video capture with error handling
        try:
            # Ensure directory exists
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            animator.setup_training_video(save_path, fps=fps, dpi=dpi)
            animation_enabled = True
        except Exception as e:
            print(f"Failed to setup animation: {str(e)}")
            print("Training will continue without animation capture.")
            animation_enabled = False
                
        # Initialize metrics storage
        self.train_loss = []
        self.train_accuracy = []
        self.learning_rates = []
        self.train_precision = []
        self.train_recall = []
        self.train_f1 = []
        
        if X_val is not None:
            self.val_loss = []
            self.val_accuracy = []
            self.val_precision = []
            self.val_recall = []
            self.val_f1 = []
        
        # Default optimizer if not provided
        if optimizer is None:
            optimizer = AdamOptimizer(learning_rate=0.0001)
        
        # Initialize optimizer
        optimizer.initialize(self.layers)
        
        # Track best model for early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_weights = [layer.weights.copy() for layer in self.layers if hasattr(layer, 'weights')]
        best_biases = [layer.biases.copy() for layer in self.layers if hasattr(layer, 'biases')]
        
        def process_batch(start_idx, X_shuffled, y_shuffled):
            X_batch = X_shuffled[start_idx:start_idx+batch_size]
            y_batch = y_shuffled[start_idx:start_idx+batch_size]
            
            # Forward and backward passes
            self.forward(X_batch, training=True)
            self.backward(y_batch)
            
            # Update weights and biases
            trainable_idx = 0  # Separate index for trainable layers
            for layer in self.layers:
                if hasattr(layer, 'weight_gradients') and layer.weight_gradients is not None:
                    dW = layer.weight_gradients
                    db = layer.bias_gradients
                    optimizer.update(layer, dW, db, trainable_idx)
                    trainable_idx += 1
        
        # Training loop
        for epoch in range(epochs):
            # Reset gradients
            for layer in self.layers:
                if hasattr(layer, 'zero_grad'):
                    layer.zero_grad()
            
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Process all batches
            for i in range(0, X_train.shape[0], batch_size):
                process_batch(i, X_shuffled, y_shuffled)
            
            # Calculate metrics
            train_loss = self.calculate_loss(X_train, y_train)
            train_accuracy, _ = self.evaluate(X_train, y_train)
            train_precision, train_recall, train_f1 = self.calculate_precision_recall_f1(X_train, y_train)
            
            # Store metrics
            self.train_loss.append(train_loss)
            self.train_accuracy.append(train_accuracy)
            self.train_precision.append(train_precision)
            self.train_recall.append(train_recall)
            self.train_f1.append(train_f1)
            
            # Prepare metrics for animator
            train_metrics = {
                'loss': train_loss,
                'accuracy': train_accuracy,
                'precision': train_precision,
                'recall': train_recall,
                'f1': train_f1
            }           
            
            # Validation metrics
            val_metrics_str = ""
            if X_val is not None:
                val_loss = self.calculate_loss(X_val, y_val)
                val_accuracy, _ = self.evaluate(X_val, y_val)
                val_precision, val_recall, val_f1 = self.calculate_precision_recall_f1(X_val, y_val)
                
                # Store validation metrics
                self.val_loss.append(val_loss)
                self.val_accuracy.append(val_accuracy)
                self.val_precision.append(val_precision)
                self.val_recall.append(val_recall)
                self.val_f1.append(val_f1)
                
                val_metrics_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                
                # Update animator with validation metrics
                val_metrics = {
                    'loss': val_loss,
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1
                }
                animator.update_metrics(val_metrics, validation=True)
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    best_weights = [layer.weights.copy() for layer in self.layers if hasattr(layer, 'weights')]
                    best_biases = [layer.biases.copy() for layer in self.layers if hasattr(layer, 'biases')]
                else:
                    patience_counter += 1
            else:
                # Use training loss for early stopping if no validation set
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                    best_weights = [layer.weights.copy() for layer in self.layers if hasattr(layer, 'weights')]
                    best_biases = [layer.biases.copy() for layer in self.layers if hasattr(layer, 'biases')]
                else:
                    patience_counter += 1

            # Update learning rate if scheduler is provided
            if lr_scheduler:
                if isinstance(lr_scheduler, lr_scheduler_plateau):
                    msg = lr_scheduler.step(epoch, train_loss if X_val is None else val_loss)
                    if msg: print(msg)
                else:
                    msg = lr_scheduler.step(epoch)
                    if msg: print(msg)

                self.learning_rates.append(optimizer.learning_rate)
                train_metrics['learning_rate'] = optimizer.learning_rate
            
            # Update animator with training metrics
            animator.update_metrics(train_metrics, validation=False)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}{val_metrics_str}")
            
            # Add frame to the animation if needed and if animation is enabled
            if animation_enabled and (epoch % frame_every == 0 or epoch == epochs - 1):
                try:
                    animator.add_training_frame()
                except Exception as e:
                    print(f"Failed to add animation frame: {str(e)}")
                    animation_enabled = False
            
            # Early stopping
            if patience_counter >= early_stopping_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                # Capture final frame if not already captured and if animation is enabled
                if animation_enabled and epoch % frame_every != 0:
                    try:
                        animator.add_training_frame()
                    except Exception:
                        pass
                break
        
        # Restore best weights
        trainable_layers = [l for l in self.layers if hasattr(l, 'weights')]
        for i, layer in enumerate(trainable_layers):
            layer.weights = best_weights[i]
            layer.biases = best_biases[i]
        
        # Finish and save the animation if enabled
        if animation_enabled:
            try:
                animator.finish_training_video()
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Failed to finish animation: {str(e)}")
                
                # Alternative: generate static plot
                try:
                    static_plot_path = save_path.rsplit('.', 1)[0] + '.png'
                    self.plot_metrics(save_dir=static_plot_path)
                    print(f"Static metrics plot saved to {static_plot_path} instead")
                except Exception:
                    pass
        else:
            # Generate static plot as fallback
            try:
                static_plot_path = save_path.rsplit('.', 1)[0] + '.png'
                self.plot_metrics(save_dir=static_plot_path)
                print(f"Static metrics plot saved to {static_plot_path}")
            except Exception as e:
                print(f"Failed to save static plot: {str(e)}")
        
        