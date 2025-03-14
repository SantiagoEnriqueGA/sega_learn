import numpy as np
import warnings

from .schedulers import *
from .layers import *
try:
    from .layers_jit import *
except:
    JITDenseLayer = None


class NeuralNetworkBase:
    def __init__(self, layers, dropout_rate=0.0, reg_lambda=0.0, activations=None):
        _layers = [DenseLayer, FlattenLayer, ConvLayer, RNNLayer]
        _layers_jit = [JITDenseLayer, JITFlattenLayer, JITConvLayer, JITRNNLayer]
        available_layers = tuple(_layers + _layers_jit)
        
        # iF all layers are integers, initialize the layers as DenseLayers
        if all(isinstance(layer, int) for layer in layers):
            self.layer_sizes = layers
            self.dropout_rate = dropout_rate
            self.reg_lambda = reg_lambda
            self.activations = activations if activations else ['relu'] * (len(layers) - 2) + ['softmax']
            self.layers = []
            self.weights = []
            self.biases = []
            self.layer_outputs = None
            self.is_binary = layers[-1] == 1

        # Else if all layers are Layer objects, use them directly
        elif all(isinstance(layer, available_layers) for layer in layers):
            self.layers = layers
            self.layer_sizes = [layer.input_size for layer in layers] + [layers[-1].output_size]
            self.dropout_rate = dropout_rate
            self.reg_lambda = reg_lambda
            self.is_binary = layers[-1].output_size == 1
        else:
            raise ValueError("layers must be a list of integers or a list of Layer objects.")
        
    def initialize_layers(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def forward(self, X, training=True):
        raise NotImplementedError("This method should be implemented by subclasses")

    def backward(self, y):
        raise NotImplementedError("This method should be implemented by subclasses")

    def train(self, X_train, y_train, X_val=None, y_val=None, optimizer=None, epochs=100, batch_size=32, early_stopping_threshold=10, lr_scheduler=None, p=True, use_tqdm=True, n_jobs=1, track_metrics=False, track_adv_metrics=False):
        raise NotImplementedError("This method should be implemented by subclasses")

    def evaluate(self, X, y):
        raise NotImplementedError("This method should be implemented by subclasses")

    def predict(self, X):
        raise NotImplementedError("This method should be implemented by subclasses")

    def calculate_loss(self, X, y):
        raise NotImplementedError("This method should be implemented by subclasses")

    def apply_dropout(self, X):
        """
        Applies dropout to the activation X.
        Args:
            X (ndarray): Activation values.
        Returns:
            ndarray: Activation values after applying dropout.
        """
        mask = np.random.rand(*X.shape) < (1 - self.dropout_rate)
        return np.multiply(X, mask) / (1 - self.dropout_rate)

    def compute_l2_reg(self, weights):
        """
        Computes the L2 regularization term.
        Args:
            weights (list): List of weight matrices.
        Returns:
            float: L2 regularization term.
        """
        total = 0.0
        for i in range(len(weights)):
            total += np.sum(weights[i] ** 2)
        return total
    
    def calculate_precision_recall_f1(self, X, y):
        """
        Calculates precision, recall, and F1 score.
        Parameters:
            - X (ndarray): Input data
            - y (ndarray): Target labels
        Returns:
            - precision (float): Precision score
            - recall (float): Recall score
            - f1 (float): F1 score
        """
        _, predicted = self.evaluate(X, y)
        true_positive = np.sum((predicted == 1) & (y == 1))
        false_positive = np.sum((predicted == 1) & (y == 0))
        false_negative = np.sum((predicted == 0) & (y == 1))
        
        precision = true_positive / (true_positive + false_positive + 1e-15)
        recall = true_positive / (true_positive + false_negative + 1e-15)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
        
        return precision, recall, f1
    
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
            
    def plot_metrics(self, save_dir=None):
        """
        Plots the training and validation metrics.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib first.")
        
        if not hasattr(self, 'train_loss'):
            raise ValueError("No training history available. Please set track_metrics=True during training.")
        
        # Different number of plots for metrics vs metrics/adv_metrics
        
        # If ONLY metrics are tracked OR ONLY adv_metrics are tracked
        if (hasattr(self, 'train_loss') + hasattr(self, 'train_precision')) == 1:
            cnt = 1
            plt.figure(figsize=(16, 5))  # Adjust the figure size to accommodate three plots
            
        elif (hasattr(self, 'train_loss') + hasattr(self, 'train_precision')) == 2:
            cnt = 2
            plt.figure(figsize=(16, 8))
        
        # Plot Loss
        if cnt == 1: plt.subplot(1, 3, 1)
        if cnt == 2: plt.subplot(2, 3, 1)
        plt.plot(self.train_loss, label='Train Loss')
        if hasattr(self, 'val_loss'):
            plt.plot(self.val_loss, label='Val Loss')
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot Accuracy
        if cnt == 1: plt.subplot(1, 3, 2)
        if cnt == 2: plt.subplot(2, 3, 2)
        plt.plot(self.train_accuracy, label='Train Accuracy')
        if hasattr(self, 'val_accuracy'):
            plt.plot(self.val_accuracy, label='Val Accuracy')
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Plot Learning Rate
        if cnt == 1: plt.subplot(1, 3, 3)
        if cnt == 2: plt.subplot(2, 3, 3)
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        
        
        if cnt == 2:
            # Plot Precision
            plt.subplot(2, 3, 4)
            plt.plot(self.train_precision, label='Train Precision')
            if hasattr(self, 'val_precision'):
                plt.plot(self.val_precision, label='Val Precision')
            plt.title("Precision")
            plt.xlabel("Epoch")
            plt.ylabel("Precision")
            plt.legend()
            
            # Plot Recall
            plt.subplot(2, 3, 5)
            plt.plot(self.train_recall, label='Train Recall')
            if hasattr(self, 'val_recall'):
                plt.plot(self.val_recall, label='Val Recall')
            plt.title("Recall")
            plt.xlabel("Epoch")
            plt.ylabel("Recall")
            plt.legend()
            
            # Plot F1 Score
            plt.subplot(2, 3, 6)
            plt.plot(self.train_f1, label='Train F1 Score')
            if hasattr(self, 'val_f1'):
                plt.plot(self.val_f1, label='Val F1 Score')
            plt.title("F1 Score")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.legend()       
        
        plt.tight_layout()         
        
        if save_dir:
            plt.savefig(save_dir, dpi=600)
        else:
            plt.show()

    def animate_training_metrics(self, interval=100, figsize=(18, 10), save_path=None, writer='ffmpeg', dpi=100, fps=15):
        """
        Creates an animated visualization of training metrics during model training.
        Uses matplotlib's animation with blitting for efficiency.
        
        Parameters:
            interval (int): Update interval in milliseconds during display
            figsize (tuple): Figure size (width, height)
            save_path (str): Path to save the animation (e.g., 'training_animation.mp4')
            writer (str): Animation writer ('ffmpeg', 'pillow', 'html', etc.)
            dpi (int): DPI for saved animation 
            fps (int): Frames per second for saved animation
            
        Returns:
            matplotlib.animation.FuncAnimation: Animation object
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            import matplotlib.lines as lines
        except ImportError:
            raise ImportError("Matplotlib is required for animation. Please install matplotlib first.")
        
        # Check for ffmpeg if requested
        if writer == 'ffmpeg' and save_path:
            try:
                from matplotlib.animation import FFMpegWriter
            except ImportError:
                raise ImportError("FFmpeg writer not available. Install ffmpeg or choose a different writer.")
        
        # Check if metrics are being tracked
        if not hasattr(self, 'train_loss'):
            raise ValueError("No training metrics available. Set track_metrics=True during training.")
        
        # Setup figure based on available metrics
        has_adv_metrics = hasattr(self, 'train_precision')
        n_rows = 2 if has_adv_metrics else 1
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        axes = []
        # Basic metrics: loss, accuracy, learning rate
        axes.append(fig.add_subplot(n_rows, 3, 1))
        axes.append(fig.add_subplot(n_rows, 3, 2))
        axes.append(fig.add_subplot(n_rows, 3, 3))
        
        # Advanced metrics if available
        if has_adv_metrics:
            axes.append(fig.add_subplot(n_rows, 3, 4))
            axes.append(fig.add_subplot(n_rows, 3, 5))
            axes.append(fig.add_subplot(n_rows, 3, 6))
        
        # Set titles
        axes[0].set_title("Loss")
        axes[1].set_title("Accuracy")
        axes[2].set_title("Learning Rate")
        
        if has_adv_metrics:
            axes[3].set_title("Precision")
            axes[4].set_title("Recall")
            axes[5].set_title("F1 Score")
        
        # Setup x and y labels
        for i in range(3):
            axes[i].set_xlabel("Epoch")
        
        if has_adv_metrics:
            for i in range(3, 6):
                axes[i].set_xlabel("Epoch")
        
        axes[0].set_ylabel("Loss")
        axes[1].set_ylabel("Accuracy")
        axes[2].set_ylabel("Learning Rate")
        
        if has_adv_metrics:
            axes[3].set_ylabel("Precision")
            axes[4].set_ylabel("Recall")
            axes[5].set_ylabel("F1 Score")
        
        # Create empty line objects for blitting
        train_lines = []
        val_lines = []
        
        # Create lines for train metrics
        train_loss_line, = axes[0].plot([], [], 'b-', label='Train Loss')
        train_acc_line, = axes[1].plot([], [], 'b-', label='Train Accuracy')
        lr_line, = axes[2].plot([], [], 'g-', label='Learning Rate')
        
        train_lines.extend([train_loss_line, train_acc_line, lr_line])
        
        # Create lines for validation metrics if available
        if hasattr(self, 'val_loss'):
            val_loss_line, = axes[0].plot([], [], 'r-', label='Val Loss')
            val_acc_line, = axes[1].plot([], [], 'r-', label='Val Accuracy')
            val_lines.extend([val_loss_line, val_acc_line])
        
        # Create lines for advanced metrics if available
        if has_adv_metrics:
            train_prec_line, = axes[3].plot([], [], 'b-', label='Train Precision')
            train_recall_line, = axes[4].plot([], [], 'b-', label='Train Recall')
            train_f1_line, = axes[5].plot([], [], 'b-', label='Train F1')
            
            train_lines.extend([train_prec_line, train_recall_line, train_f1_line])
            
            if hasattr(self, 'val_precision'):
                val_prec_line, = axes[3].plot([], [], 'r-', label='Val Precision')
                val_recall_line, = axes[4].plot([], [], 'r-', label='Val Recall')
                val_f1_line, = axes[5].plot([], [], 'r-', label='Val F1')
                
                val_lines.extend([val_prec_line, val_recall_line, val_f1_line])
        
        # Add legends
        for ax in axes:
            ax.legend()
        
        # Prepare for animation
        epochs_data = list(range(len(self.train_loss)))
        max_frames = len(epochs_data)
        
        # Pre-calculate axis limits for better appearance and to avoid autoscaling during animation
        y_min_loss, y_max_loss = min(self.train_loss) * 0.95, max(self.train_loss) * 1.05
        y_min_acc, y_max_acc = 0, 1.05  # Accuracy is between 0 and 1
        y_min_lr, y_max_lr = min(self.learning_rates) * 0.95, max(self.learning_rates) * 1.05
        
        # Handle validation metrics if available
        if hasattr(self, 'val_loss'):
            y_min_loss = min(y_min_loss, min(self.val_loss) * 0.95)
            y_max_loss = max(y_max_loss, max(self.val_loss) * 1.05)
        
        # Set axis limits
        axes[0].set_xlim(0, max(len(epochs_data) - 1, 1))
        axes[0].set_ylim(y_min_loss, y_max_loss)
        
        axes[1].set_xlim(0, max(len(epochs_data) - 1, 1))
        axes[1].set_ylim(y_min_acc, y_max_acc)
        
        axes[2].set_xlim(0, max(len(epochs_data) - 1, 1))
        axes[2].set_ylim(y_min_lr, y_max_lr)
        
        # Set limits for advanced metrics
        if has_adv_metrics:
            y_min_prec, y_max_prec = 0, 1.05
            y_min_recall, y_max_recall = 0, 1.05
            y_min_f1, y_max_f1 = 0, 1.05
            
            axes[3].set_xlim(0, max(len(epochs_data) - 1, 1))
            axes[3].set_ylim(y_min_prec, y_max_prec)
            
            axes[4].set_xlim(0, max(len(epochs_data) - 1, 1))
            axes[4].set_ylim(y_min_recall, y_max_recall)
            
            axes[5].set_xlim(0, max(len(epochs_data) - 1, 1))
            axes[5].set_ylim(y_min_f1, y_max_f1)
        
        # Create empty lists to store static background artists
        static_artists = []
        for ax in axes:
            static_artists.extend(ax.get_xticklabels())
            static_artists.extend(ax.get_yticklabels())
            static_artists.append(ax.title)
            static_artists.append(ax.xaxis.label)
            static_artists.append(ax.yaxis.label)
            if ax.legend_ is not None:
                static_artists.append(ax.legend_)
        
        # Tight layout for better appearance
        plt.tight_layout()
        
        # Initialize function for blitting
        def init():
            # Initialize all lines with empty data
            for line in train_lines + val_lines:
                line.set_data([], [])
            
            # Return all artists that need to be redrawn
            return train_lines + val_lines
        
        # Animation update function
        def update(frame):
            # Calculate current epoch to show (use frame as index)
            current_epoch = min(frame, len(epochs_data) - 1)
            
            # Update basic metrics with data up to current epoch
            x_data = epochs_data[:current_epoch+1]
            
            train_loss_line.set_data(x_data, self.train_loss[:current_epoch+1])
            train_acc_line.set_data(x_data, self.train_accuracy[:current_epoch+1])
            lr_line.set_data(x_data, self.learning_rates[:current_epoch+1])
            
            # Update validation metrics if available
            if hasattr(self, 'val_loss'):
                val_loss_line.set_data(x_data, self.val_loss[:current_epoch+1])
                val_acc_line.set_data(x_data, self.val_accuracy[:current_epoch+1])
            
            # Update advanced metrics if available
            if has_adv_metrics:
                train_prec_line.set_data(x_data, self.train_precision[:current_epoch+1])
                train_recall_line.set_data(x_data, self.train_recall[:current_epoch+1])
                train_f1_line.set_data(x_data, self.train_f1[:current_epoch+1])
                
                if hasattr(self, 'val_precision'):
                    val_prec_line.set_data(x_data, self.val_precision[:current_epoch+1])
                    val_recall_line.set_data(x_data, self.val_recall[:current_epoch+1])
                    val_f1_line.set_data(x_data, self.val_f1[:current_epoch+1])
            
            # Return all artists that have been updated
            return train_lines + val_lines
        
        # Create animation with fixed number of frames for saving
        anim = FuncAnimation(
            fig, update, frames=max_frames, init_func=init, 
            interval=interval, blit=True, cache_frame_data=False
        )
        
        # Save animation if path is provided
        if save_path:
            print(f"Saving animation to {save_path}...")
            
            # Configure writer
            if writer == 'ffmpeg':
                writer_obj = FFMpegWriter(fps=fps)
                anim.save(save_path, writer=writer_obj, dpi=dpi)
            elif writer == 'pillow':
                from matplotlib.animation import PillowWriter
                writer_obj = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer_obj, dpi=dpi)
            elif writer == 'html':
                from matplotlib.animation import HTMLWriter
                writer_obj = HTMLWriter(fps=fps)
                anim.save(save_path, writer=writer_obj, dpi=dpi)
            else:
                # Use default writer
                anim.save(save_path, fps=fps, dpi=dpi)
            
            print(f"Animation saved successfully to {save_path}")
        
        # Display plot
        plt.show()
        
        return anim