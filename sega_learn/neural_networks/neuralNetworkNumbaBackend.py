import warnings

import numpy as np

from .neuralNetworkBase import NeuralNetworkBase
from .schedulers import lr_scheduler_plateau

try:
    from .layers_jit import *
    from .loss_jit import *
    from .numba_utils import *
    from .optimizers_jit import *

    NUMBA_AVAILABLE = True
except ImportError:
    raise ImportError(
        "Numba is not installed. Please install it to use the Numba backend."
    ) from None

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Notes: Using different JITLayers is currently broken, IDEAS:
# ---------------------------------------------------------------------------------------
# 1. Split the Layers into Homogeneous Lists
# Separate your layers by type and then initialize/update them separately.
# trainable_conv_layers = [layer for layer in self.layers if isinstance(layer, JITConvLayer)]
# trainable_dense_layers = [layer for layer in self.layers if isinstance(layer, JITDenseLayer)]
# 2. Refactor to a Single Layer Type
# Refactoring code so that both convolutional and dense layers share a unified interface and are implemented as a single jitclass type.
# Store all layers in one homogeneous list.


class NumbaBackendNeuralNetwork(NeuralNetworkBase):
    """A neural network implementation using Numba for Just-In-Time (JIT) compilation to optimize performance.

    This class supports forward and backward propagation, training, evaluation, and hyperparameter tuning
    with various optimizers and activation functions.

    Attributes:
        compiled (bool): Indicates whether Numba functions are compiled.
        trainable_layers (list): Layers with trainable parameters (weights and biases).
        progress_bar (bool): Whether to display a progress bar during training.

    Methods:
        __init__(layers, dropout_rate, reg_lambda, activations, compile_numba, progress_bar):
            Initializes the neural network with the specified parameters.
        store_init_layers():
            Stores the initial layers and their parameters for restoration after initialization.
        restore_layers():
            Restores the layers and their parameters after initialization.
        initialize_new_layers():
            Initializes the layers of the neural network with specified sizes and activation functions.
        forward(X, training):
            Performs forward propagation through the neural network.
        backward(y):
            Performs backward propagation to calculate gradients.
        is_not_instance_of_classes(obj, classes):
            Checks if an object is not an instance of any class in a list of classes.
        train(X_train, y_train, X_val, y_val, optimizer, epochs, batch_size, early_stopping_threshold,
              lr_scheduler, p, use_tqdm, n_jobs, track_metrics, track_adv_metrics, save_animation,
              save_path, fps, dpi, frame_every):
            Trains the neural network model with the specified parameters.
        evaluate(X, y):
            Evaluates the neural network on the given data and returns accuracy and predictions.
        predict(X):
            Predicts the output for the given input data.
        calculate_loss(X, y):
            Calculates the loss with L2 regularization.
        _create_optimizer(optimizer_type, learning_rate, JIT):
            Helper method to create optimizer instances.
        tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid, layer_configs, optimizer_types,
                             lr_range, epochs, batch_size):
            Performs hyperparameter tuning using grid search.
        compile_numba_functions(progress_bar):
            Compiles all Numba JIT functions to improve performance.
    """

    def __init__(
        self,
        layers,
        dropout_rate=0.2,
        reg_lambda=0.01,
        activations=None,
        compile_numba=True,
        progress_bar=True,
    ):
        """Initializes the Numba backend neural network.

        Args:
            layers: (list) - List of layer sizes or Layer objects.
            dropout_rate: (float) - Dropout rate for regularization.
            reg_lambda: (float) - L2 regularization parameter.
            activations: (list) - List of activation functions for each layer.
            compile_numba: (bool) - Whether to compile Numba functions.
            progress_bar: (bool) - Whether to display a progress bar.
        """
        super().__init__(layers, dropout_rate, reg_lambda, activations)
        self.compiled = False
        # if layers are empty list, initialize them
        if len(self.layers) == 0:
            self.initialize_new_layers()

        # Identify trainable layers
        self.trainable_layers = [
            layer
            for layer in self.layers
            if hasattr(layer, "weights") and hasattr(layer, "biases")
        ]

        if progress_bar and not TQDM_AVAILABLE:
            warnings.warn(
                "tqdm is not installed. Progress bar will not be displayed.",
                stacklevel=2,
            )
            self.progress_bar = False
        else:
            self.progress_bar = progress_bar

        if compile_numba and not self.compiled:
            self.store_init_layers()
            self.compile_numba_functions(self.progress_bar)
            self.restore_layers()
            self.compiled = True

    def store_init_layers(self):
        """Stores the layers to restore after initialization."""
        self._layers = self.layers.copy()
        self._weights = [
            layer.weights.copy() for layer in self.layers if hasattr(layer, "weights")
        ]
        self._biases = [
            layer.biases.copy() for layer in self.layers if hasattr(layer, "biases")
        ]

        self.weights = [
            layer.weights.copy() for layer in self.layers if hasattr(layer, "weights")
        ]
        self.biases = [
            layer.biases.copy() for layer in self.layers if hasattr(layer, "biases")
        ]
        self.dWs_cache = [np.zeros_like(w) for w in self.weights]
        self.dbs_cache = [np.zeros_like(b) for b in self.biases]

    def restore_layers(self):
        """Restores the layers after initialization."""
        self.layers = self._layers.copy()
        self.weights = [
            layer.weights.copy() for layer in self.layers if hasattr(layer, "weights")
        ]
        self.biases = [
            layer.biases.copy() for layer in self.layers if hasattr(layer, "biases")
        ]

    def initialize_new_layers(self):
        """Initializes the layers of the neural network.

        Each layer is created with the specified number of neurons and activation function.
        """
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(
                JITDenseLayer(
                    self.layer_sizes[i], self.layer_sizes[i + 1], self.activations[i]
                )
            )
            weight = (
                np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            )
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
        self.dWs_cache = [np.zeros_like(w) for w in self.weights]
        self.dbs_cache = [np.zeros_like(b) for b in self.biases]

    def forward(self, X, training=True):
        """Performs forward propagation through the neural network.

        Args:
            X (ndarray): Input data of shape (batch_size, input_size).
            training (bool): Whether the network is in training mode (applies dropout).

        Returns:
            ndarray: Output predictions of shape (batch_size, output_size).
        """
        # Convert input to float64 for Numba compatibility
        self.layer_outputs = [X]
        A = X.astype(np.float64)

        output = None
        for i, layer in enumerate(self.layers):
            A = layer.forward(A)

            # If last layer, store output for return
            if i == len(self.layers) - 1:
                output = A

            if training and self.dropout_rate > 0 and isinstance(layer, JITDenseLayer):
                A = apply_dropout_jit(A, self.dropout_rate)
            self.layer_outputs.append(A)

        return output

    def backward(self, y):
        """Performs backward propagation to calculate the gradients.

        Args:
            y (ndarray): Target labels of shape (m, output_size).
        """
        # Convert target labels to int32 for Numba compatibility
        y = y.astype(np.int32)
        m = y.shape[0]
        outputs = self.layer_outputs[-1]

        if self.is_binary:
            y = y.reshape(-1, 1).astype(np.float64)
            dA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15))
        else:
            dA = outputs.copy()
            for i in range(m):
                dA[i, y[i]] -= 1

        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backward(dA, self.reg_lambda)

    @staticmethod
    def is_not_instance_of_classes(obj, classes):
        """Checks if an object is not an instance of any class in a list of classes.

        Args:
            obj: The object to check.
            classes: A list of classes.

        Returns:
            bool: True if the object is not an instance of any class in the list of classes, False otherwise.
        """
        return not isinstance(obj, tuple(classes))

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        optimizer=None,
        epochs=100,
        batch_size=32,
        early_stopping_threshold=10,
        lr_scheduler=None,
        p=True,
        use_tqdm=True,
        n_jobs=1,
        track_metrics=False,
        track_adv_metrics=False,
        save_animation=False,
        save_path="training_animation.mp4",
        fps=1,
        dpi=100,
        frame_every=1,
    ):
        """Trains the neural network model.

        Args:
            X_train: (ndarray) - Training data features.
            y_train: (ndarray) - Training data labels.
            X_val: (ndarray) - Validation data features, optional.
            y_val: (ndarray) - Validation data labels, optional.
            optimizer: (Optimizer) - Optimizer for updating parameters (default: JITAdam, lr=0.0001).
            epochs: (int) - Number of training epochs (default: 100).
            batch_size: (int) - Batch size for mini-batch gradient descent (default: 32).
            early_stopping_threshold: (int) - Patience for early stopping (default: 10).
            lr_scheduler: (Scheduler) - Learning rate scheduler (default: None).
            p: (bool) - Whether to print training progress (default: True).
            use_tqdm: (bool) - Whether to use tqdm for progress bar (default: True).
            n_jobs: (int) - Number of jobs for parallel processing (default: 1).
            track_metrics: (bool) - Whether to track training metrics (default: False).
            track_adv_metrics: (bool) - Whether to track advanced metrics (default: False).
            save_animation: (bool) - Whether to save the animation of metrics (default: False).
            save_path: (str) - Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
            fps: (int) - Frames per second for the saved animation (default: 1).
            dpi: (int) - DPI for the saved animation (default: 100).
            frame_every: (int) - Capture frame every N epochs (to reduce file size) (default: 1).
        """
        if use_tqdm and not TQDM_AVAILABLE:
            warnings.warn(
                "TQDM is not available. Disabling progress bar.",
                UserWarning,
                stacklevel=2,
            )
            use_tqdm = False

        # If track_adv_metrics is True, X_val and y_val must be provided
        if track_adv_metrics and (X_val is None or y_val is None):
            track_adv_metrics = False
            warnings.warn(
                "track_adv_metrics is True but X_val and y_val are not provided. Disabling track_adv_metrics.",
                UserWarning,
                stacklevel=2,
            )

        # If track_adv_metrics is True, set track_metrics to True
        if track_adv_metrics:
            track_metrics = True

        # If save_animation is True but track_metrics is False, set track_metrics to True, try to set track_adv_metrics to True
        if save_animation and not track_metrics:
            track_metrics = True
            if (X_val is not None) and (y_val is not None):
                track_adv_metrics = True

        if save_animation:
            import os

            from .animation import TrainingAnimator

            # Animation metrics to track
            metrics = ["loss", "accuracy", "precision", "recall", "f1"]
            if lr_scheduler:
                metrics.append("learning_rate")

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
            optimizer = JITAdamOptimizer(learning_rate=0.0001)

        # If optimizer is not a JIT optimizer, convert it to a JIT optimizer
        jit_optimizer_classes = [
            JITAdamOptimizer,
            JITSGDOptimizer,
            JITAdadeltaOptimizer,
        ]
        if NumbaBackendNeuralNetwork.is_not_instance_of_classes(
            optimizer, jit_optimizer_classes
        ):
            warnings.warn(
                "Attempting to use a non-JIT optimizer. Converting to a JIT optimizer.",
                UserWarning,
                stacklevel=2,
            )
            try:
                if optimizer.__class__.__name__ == "AdamOptimizer":
                    optimizer = JITAdamOptimizer(
                        learning_rate=optimizer.learning_rate,
                        beta1=optimizer.beta1,
                        beta2=optimizer.beta2,
                        epsilon=optimizer.epsilon,
                        reg_lambda=optimizer.reg_lambda,
                    )
                elif optimizer.__class__.__name__ == "SGDOptimizer":
                    optimizer = JITSGDOptimizer(
                        learning_rate=optimizer.learning_rate,
                        momentum=optimizer.momentum,
                        reg_lambda=optimizer.reg_lambda,
                    )
                elif optimizer.__class__.__name__ == "AdadeltaOptimizer":
                    optimizer = JITAdadeltaOptimizer(
                        learning_rate=optimizer.learning_rate,
                        rho=optimizer.rho,
                        epsilon=optimizer.epsilon,
                        reg_lambda=optimizer.reg_lambda,
                    )
                else:
                    raise ValueError(
                        f"Unsupported optimizer: {optimizer.__class__.__name__}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Unable to convert optimizer to a JIT optimizer: {str(e)}. Please use a JIT optimizer."
                ) from None

        # Initialize optimizer
        optimizer.initialize(self.trainable_layers)

        # Track best model for early stopping
        best_loss = float("inf")
        patience_counter = 0
        best_weights = [
            layer.weights.copy() for layer in self.layers if hasattr(layer, "weights")
        ]
        best_biases = [
            layer.biases.copy() for layer in self.layers if hasattr(layer, "biases")
        ]

        # Number of threads for parallel processing
        # If n_jobs > 1, use that many threads, otherwise let Numba decide
        if n_jobs > 1:
            import os

            os.environ["NUMBA_NUM_THREADS"] = str(n_jobs)

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

        # Training loop with progress bar
        progress_bar = tqdm(range(epochs)) if use_tqdm else range(epochs)
        for epoch in progress_bar:
            # Reset gradients
            for layer in self.trainable_layers:
                layer.zero_grad()

            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Prepare layer parameters for JIT function
            weights = [layer.weights for layer in self.trainable_layers]
            biases = [layer.biases for layer in self.trainable_layers]

            # Initialize accumulated gradients with zeros
            dWs_zeros = [np.zeros_like(w) for w in weights]
            dbs_zeros = [np.zeros_like(b) for b in biases]

            # Get indices of dropout layers
            dropout_layer_indices = [
                i
                for i, layer in enumerate(self.layers)
                if isinstance(layer, JITDenseLayer)
            ]
            if len(dropout_layer_indices) == 0:
                dropout_layer_indices = [-1]

            # Process batches based on classification type
            if self.is_binary:
                dWs_acc, dbs_acc, train_loss, train_accuracy = process_batches_binary(
                    X_shuffled,
                    y_shuffled,
                    batch_size,
                    self.layers,
                    self.dropout_rate,
                    dropout_layer_indices,
                    self.reg_lambda,
                    dWs_zeros,
                    dbs_zeros,
                )
            else:
                dWs_acc, dbs_acc, train_loss, train_accuracy = process_batches_multi(
                    X_shuffled,
                    y_shuffled,
                    batch_size,
                    self.layers,
                    self.dropout_rate,
                    dropout_layer_indices,
                    self.reg_lambda,
                    dWs_zeros,
                    dbs_zeros,
                )

            # Update weights and biases using the optimizer
            optimizer.update_layers(self.trainable_layers, dWs_acc, dbs_acc)

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
                    best_weights = [
                        layer.weights.copy()
                        for layer in self.layers
                        if hasattr(layer, "weights")
                    ]
                    best_biases = [
                        layer.biases.copy()
                        for layer in self.layers
                        if hasattr(layer, "biases")
                    ]
                else:
                    patience_counter += 1
            else:
                # Use training loss for early stopping if no validation set
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                    best_weights = [
                        layer.weights.copy()
                        for layer in self.layers
                        if hasattr(layer, "weights")
                    ]
                    best_biases = [
                        layer.biases.copy()
                        for layer in self.layers
                        if hasattr(layer, "biases")
                    ]
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
                train_precision, train_recall, train_f1 = (
                    self.calculate_precision_recall_f1(X_train, y_train)
                )
                self.train_precision.append(train_precision)
                self.train_recall.append(train_recall)
                self.train_f1.append(train_f1)
                if X_val is not None:
                    val_precision, val_recall, val_f1 = (
                        self.calculate_precision_recall_f1(X_val, y_val)
                    )
                    self.val_precision.append(val_precision)
                    self.val_recall.append(val_recall)
                    self.val_f1.append(val_f1)

            # Update progress bar or print metrics
            if p:
                if use_tqdm and isinstance(progress_bar, tqdm):
                    progress_bar.set_description(
                        f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}{val_metrics}"
                    )
                else:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}{val_metrics}"
                    )

            # Learning rate scheduler step
            if lr_scheduler:
                if isinstance(lr_scheduler, lr_scheduler_plateau):
                    msg = lr_scheduler.step(
                        epoch, train_loss if X_val is None else val_loss
                    )
                    lr = optimizer.learning_rate
                    if p and msg:
                        tqdm.write(msg)
                else:
                    msg = lr_scheduler.step(epoch)
                    lr = optimizer.learning_rate
                    if p and msg:
                        tqdm.write(msg)

            # Update animator with metrics
            if save_animation:
                train_metrics = {
                    "loss": train_loss,
                    "accuracy": train_accuracy,
                    "precision": train_precision,
                    "recall": train_recall,
                    "f1": train_f1,
                }
                if lr:
                    train_metrics["learning_rate"] = lr
                animator.update_metrics(train_metrics, validation=False)
                val_metrics = {
                    "loss": val_loss,
                    "accuracy": val_accuracy,
                    "precision": val_precision,
                    "recall": val_recall,
                    "f1": val_f1,
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
                if p and use_tqdm:
                    tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best weights
        trainable_layers = [layer for layer in self.layers if hasattr(layer, "weights")]
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
                    static_plot_path = save_path.rsplit(".", 1)[0] + ".png"
                    self.plot_metrics(save_dir=static_plot_path)
                    print(f"Static metrics plot saved to {static_plot_path} instead")
                except Exception:
                    pass

        # Return the animator for further use if needed
        return animator if save_animation else None

    def evaluate(self, X, y):
        """Evaluates the neural network on the given data.

        Args:
            X (ndarray): Input data.
            y (ndarray): Target labels.

        Returns:
            tuple: Accuracy and predicted labels.
        """
        # Get predictions (forward pass w/o dropout)
        y_hat = self.forward(X, training=False)

        # Evaluate predictions
        accuracy, predicted = evaluate_jit(y_hat, y, self.is_binary)
        predicted = predicted.reshape(y.shape)
        return accuracy, predicted

    def predict(self, X):
        """Predicts the output for the given input data.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Predicted outputs.
        """
        # Get predictions (forward pass w/o dropout)
        outputs = self.forward(X, training=False)
        return outputs if self.is_binary else np.argmax(outputs, axis=1)

    def calculate_loss(self, X, y):
        """Calculates the loss with L2 regularization.

        Args:
            X (ndarray): Input data.
            y (ndarray): Target labels.

        Returns:
            float: The calculated loss value.
        """
        # Get predictions (forward pass w/o dropout)
        outputs = self.forward(X, training=False)
        # If binary classification use BCE loss
        if self.is_binary:
            loss_fn = JITBCEWithLogitsLoss()
            loss = loss_fn.calculate_loss(outputs, y.reshape(-1, 1))
        # If multi-class classification use Cross-Entropy loss
        else:
            loss_fn = JITCrossEntropyLoss()
            y_ohe = np.eye(self.layer_sizes[-1])[y]
            loss = loss_fn.calculate_loss(outputs, y_ohe)

        # Add L2 regularization term
        weights = [layer.weights for layer in self.layers if hasattr(layer, "weights")]
        l2_reg = self.reg_lambda * compute_l2_reg(weights)
        loss += l2_reg
        return float(loss)

    def _create_optimizer(self, optimizer_type, learning_rate, JIT=False):
        """Helper method to create optimizer instances."""
        if optimizer_type == "Adam":
            return JITAdamOptimizer(learning_rate)
        elif optimizer_type == "SGD":
            return JITSGDOptimizer(learning_rate)
        elif optimizer_type == "Adadelta":
            return JITAdadeltaOptimizer(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def tune_hyperparameters(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        param_grid,
        layer_configs=None,
        optimizer_types=None,
        lr_range=(0.0001, 0.01, 5),
        epochs=30,
        batch_size=32,
    ):
        """Performs hyperparameter tuning using grid search.

        Args:
            X_train: (np.ndarray) - Training feature data.
            y_train: (np.ndarray) - Training target data.
            X_val: (np.ndarray) - Validation feature data.
            y_val: (np.ndarray) - Validation target data.
            param_grid: (dict) - Dictionary of parameters to try.
            layer_configs: (list), optional - List of layer configurations (default is None).
            optimizer_types: (list), optional - List of optimizer types (default is None).
            lr_range: (tuple), optional - (min_lr, max_lr, num_steps) for learning rates (default is (0.0001, 0.01, 5)).
            epochs: (int), optional - Max epochs for each trial (default is 30).
            batch_size: (int), optional - Batch size for training (default is 32).

        Returns:
            best_params: (dict) - Best hyperparameters found.
            best_accuracy: (float) - Best validation accuracy.
        """
        if not TQDM_AVAILABLE:
            # TODO: Make tqdm optional for hyperparameter tuning
            raise ImportError(
                "TQDM is currently required for hyperparameter tuning. Please install it using 'pip install tqdm'."
            )

        import warnings
        from itertools import product

        warnings.filterwarnings("ignore")

        # Default values if not provided
        if layer_configs is None:
            layer_configs = [[64], [128], [64, 32]]

        if optimizer_types is None:
            optimizer_types = ["Adam", "SGD"]

        # Output size based on target data
        output_size = (
            1 if len(y_train.shape) == 1 or y_train.shape[1] == 1 else y_train.max() + 1
        )

        # Generate learning rates
        min_lr, max_lr, num_steps = lr_range
        lr_options = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps).tolist()

        # Extract parameter combinations
        keys, values = zip(*param_grid.items(), strict=False)

        # Calculate total iterations for progress tracking
        total_iterations = (
            len(layer_configs)
            * len(lr_options)
            * len(optimizer_types)
            * np.prod([len(value) for value in values])
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
                    full_layer_structure = (
                        [X_train.shape[1]] + layer_structure + [int(output_size)]
                    )

                    for combination in product(*values):
                        params = dict(zip(keys, combination, strict=False))

                        for lr in lr_options:
                            # Create model with current hyperparameters
                            nn = NumbaBackendNeuralNetwork(
                                full_layer_structure,
                                dropout_rate=params["dropout_rate"],
                                reg_lambda=params["reg_lambda"],
                                compile_numba=False,
                            )

                            # Create optimizer
                            optimizer = self._create_optimizer(
                                optimizer_type, lr, JIT=True
                            )

                            # Train model (with early stopping for efficiency)
                            nn.train(
                                X_train,
                                y_train,
                                X_val,
                                y_val,
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
                                    "layers": full_layer_structure,
                                    "learning_rate": lr,
                                }
                                best_optimizer_type = optimizer_type

                                tqdm.write(
                                    f"New best: {best_accuracy:.4f} with {optimizer_type}, "
                                    f"lr={lr}, layers={full_layer_structure}, params={params}"
                                )

                            # Update progress
                            pbar.update(1)

        print(
            f"\nBest configuration: {best_optimizer_type} optimizer with lr={best_params['learning_rate']}"
        )
        print(f"Layers: {best_params['layers']}")
        print(
            f"Parameters: dropout={best_params['dropout_rate']}, reg_lambda={best_params['reg_lambda']}"
        )
        print(f"Validation accuracy: {best_accuracy:.4f}")

        # Add best optimizer type to best_params
        best_params["optimizer"] = best_optimizer_type

        return best_params, best_accuracy

    def compile_numba_functions(self, progress_bar=True):
        """Compiles all Numba JIT functions to improve performance.

        Args:
            progress_bar (bool): Whether to display a progress bar.
        """
        if progress_bar:
            progress_bar = tqdm(total=31, desc="Compiling Numba functions")
        else:
            progress_bar = None
        # Neural network functions
        # --------------------------------------------------------------------
        if progress_bar:
            progress_bar.set_description("Compiling Neural Network Functions")
        apply_dropout_jit(np.random.randn(10, 10), self.dropout_rate)
        if progress_bar:
            progress_bar.update(1)
        compute_l2_reg(self.weights)
        if progress_bar:
            progress_bar.update(1)
        # Compile the appropriate batch processing function based on classification type
        if self.is_binary:
            process_batches_binary(
                X_shuffled=np.random.randn(10, self.layer_sizes[0]),
                y_shuffled=np.random.randint(0, 2, (10, 1)),
                batch_size=32,
                layers=self.layers,
                dropout_rate=self.dropout_rate,
                dropout_layer_indices=[1],
                reg_lambda=self.reg_lambda,
                dWs_acc=self.dWs_cache,
                dbs_acc=self.dbs_cache,
            )
            if progress_bar:
                progress_bar.update(1)
        else:
            process_batches_multi(
                X_shuffled=np.random.randn(10, self.layer_sizes[0]),
                y_shuffled=np.random.randint(0, 2, 10),
                batch_size=32,
                layers=self.layers,
                dropout_rate=self.dropout_rate,
                dropout_layer_indices=[1],
                reg_lambda=self.reg_lambda,
                dWs_acc=self.dWs_cache,
                dbs_acc=self.dbs_cache,
            )
            if progress_bar:
                progress_bar.update(1)

        evaluate_jit(
            np.random.randn(10, self.layer_sizes[-1]),
            np.random.randint(0, 2, 10),
            self.is_binary,
        )
        if progress_bar:
            progress_bar.update(1)

        # Initialize dummy layer outputs for backward pass
        self.layer_outputs = [np.random.randn(10, size) for size in self.layer_sizes]

        # Numba Utils functions
        # --------------------------------------------------------------------
        if progress_bar:
            progress_bar.set_description("Compiling Numba Utils Functions")
        # Loss functions and evaluation
        calculate_loss_from_outputs_binary(
            np.random.randn(10, 1),
            np.random.randint(0, 2, 10).astype(np.float64),
            self.reg_lambda,
            self.weights,
        )
        if progress_bar:
            progress_bar.update(1)
        calculate_loss_from_outputs_multi(
            np.random.randn(10, self.layer_sizes[-1]),
            np.eye(self.layer_sizes[-1])[
                np.random.randint(0, self.layer_sizes[-1], 10)
            ],
            self.reg_lambda,
            self.weights,
        )
        if progress_bar:
            progress_bar.update(1)
        evaluate_batch(
            np.random.randn(10, self.layer_sizes[-1]),
            np.random.randint(0, 2, 10),
            self.is_binary,
        )
        if progress_bar:
            progress_bar.update(1)
        # Activation functions
        relu(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        relu_derivative(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        leaky_relu(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        leaky_relu_derivative(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        tanh(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        tanh_derivative(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        sigmoid(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        sigmoid_derivative(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        softmax(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        # Other utility functions
        sum_reduce(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)
        sum_axis0(np.random.randn(10, 10))
        if progress_bar:
            progress_bar.update(1)

        # Optimizers
        # --------------------------------------------------------------------
        if progress_bar:
            progress_bar.set_description("Compiling Optimizer Functions")
        # Adam
        _adam = JITAdamOptimizer()
        if progress_bar:
            progress_bar.update(1)
        _adam.initialize(self.layers)
        if progress_bar:
            progress_bar.update(1)
        _adam.update_layers(self.layers, self.dWs_cache, self.dbs_cache)
        if progress_bar:
            progress_bar.update(1)
        # SGD
        _sgd = JITSGDOptimizer()
        if progress_bar:
            progress_bar.update(1)
        _sgd.initialize(self.layers)
        if progress_bar:
            progress_bar.update(1)
        _sgd.update_layers(self.layers, self.dWs_cache, self.dbs_cache)
        if progress_bar:
            progress_bar.update(1)
        # Adadelta
        _adadelta = JITAdadeltaOptimizer()
        if progress_bar:
            progress_bar.update(1)
        _adadelta.initialize(self.layers)
        if progress_bar:
            progress_bar.update(1)
        _adadelta.update_layers(self.layers, self.dWs_cache, self.dbs_cache)
        if progress_bar:
            progress_bar.update(1)

        # Loss Modules
        # --------------------------------------------------------------------
        if progress_bar:
            progress_bar.set_description("Compiling Loss Functions")
        _cross_entropy = JITCrossEntropyLoss()
        if progress_bar:
            progress_bar.update(1)
        _cross_entropy.calculate_loss(
            np.random.randn(10, self.layer_sizes[-1]),
            np.eye(self.layer_sizes[-1])[
                np.random.randint(0, self.layer_sizes[-1], 10)
            ],
        )
        if progress_bar:
            progress_bar.update(1)
        _bce = JITBCEWithLogitsLoss()
        if progress_bar:
            progress_bar.update(1)
        _bce.calculate_loss(
            np.random.randn(10, 1),
            np.random.randint(0, 2, 10).astype(np.float64).reshape(-1, 1),
        )
        if progress_bar:
            progress_bar.update(1)
        if progress_bar:
            progress_bar.close()
