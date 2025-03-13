from sklearn.metrics import classification_report
import numpy as np
import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils import train_test_split
from sega_learn.neural_networks import *

# CIFAR-10 Data Loading and Preprocessing
# ---------------------------------------------------------------------------------------
def load_cifar10(path, subset=None):
    """Loads the CIFAR-10 dataset. If subset is provided, only randomly selects a subset percentage of the data."""
    import pickle

    train_batches = []
    test_batches = []

    # Load training batches
    for i in range(1, 6):  # CIFAR-10 has 5 training batches
        with open(os.path.join(path, f'data_batch_{i}'), 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
            train_batches.append(batch)

    # Load test batch
    with open(os.path.join(path, 'test_batch'), 'rb') as fo:
        test_batch = pickle.load(fo, encoding='bytes')
        test_batches.append(test_batch)

    # Combine training batches
    train_data = np.concatenate([batch[b'data'] for batch in train_batches])
    train_labels = np.concatenate([batch[b'labels'] for batch in train_batches])

    # Extract test data and labels
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])  # Convert to NumPy array

    if subset:
        # Select a subset of training data
        num_samples_train = train_data.shape[0]
        subset_size_train = int(num_samples_train * subset)
        train_indices = np.random.choice(num_samples_train, subset_size_train, replace=False)
        train_data = train_data[train_indices]
        train_labels = train_labels[train_indices]
        
        # Select a subset of test data
        num_samples_test = test_data.shape[0]
        subset_size_test = int(num_samples_test * subset)
        test_indices = np.random.choice(num_samples_test, subset_size_test, replace=False)
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]

    return train_data, train_labels, test_data, test_labels

def preprocess_cifar10(train_data, test_data):
    """Preprocesses CIFAR-10 data."""

    # Reshape data to (num_samples, channels, height, width)
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32)
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32)

    # Normalize pixel values to [0, 1]
    train_data /= 255.0
    test_data /= 255.0

    return train_data, test_data


# Hyperparameters
# ---------------------------------------------------------------------------------------
dropout = 0.1
reg_lambda = 0.0
lr = 0.0001
output_size = 10  # CIFAR-10 has 10 classes
batch_size = 32
epochs = 100  # Reduced for demonstration, increase for better results.
input_channels = 3  # RGB images
image_height = 32
image_width = 32
# Download cifar-10 dataset and put it in example_datasets folder
# https://www.cs.toronto.edu/~kriz/cifar.html
cifar10_path = 'example_datasets/cifar-10-batches-py'
subset = 0.10  # Select 10% of the data


# Load and Preprocess Data
# ---------------------------------------------------------------------------------------
train_data, train_labels, test_data, test_labels = load_cifar10(cifar10_path, subset=subset)
X_train, X_test = preprocess_cifar10(train_data, test_data)
y_train, y_test = train_labels, test_labels

if subset: print(f"Training on CIFAR-10 subset with {subset*100:.2f}% of the data.\n\tTraining size: {X_train.shape[0]}\n\tTest size: {X_test.shape[0]}")
else: print(f"Training on full CIFAR-10 dataset.\n\tTraining size: {X_train.shape[0]}\n\tTest size: {X_test.shape[0]}")

# Calculate the output dimensions of the convolutional layers
# Initial input: 32x32
# After first conv layer (3x3 kernel, no padding): 30x30
# After second conv layer (3x3 kernel, no padding): 28x28

# Define Optimizer, Scheduler, and Layers
# ---------------------------------------------------------------------------------------
# Optimizer and Scheduler 
optimizer1 = AdamOptimizer(learning_rate=lr)
sub_scheduler1 = lr_scheduler_step(optimizer1, lr_decay=0.1, lr_decay_epoch=10)
scheduler1 = lr_scheduler_plateau(sub_scheduler1, patience=5, threshold=0.001)

# Define Layers
layers = [
    ConvLayer(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=0, activation="relu"),
    ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, activation="relu"),
    FlattenLayer(),
    DenseLayer(64 * 28 * 28, 256, activation="relu"),
    DenseLayer(256, output_size, activation="softmax"),
]

# Initialize and Train Neural Network
# ---------------------------------------------------------------------------------------
cnn = BaseBackendNeuralNetwork(layers=layers, dropout_rate=dropout, reg_lambda=reg_lambda)

cnn.train(X_train, y_train, X_test, y_test, optimizer=optimizer1, lr_scheduler=scheduler1,
         epochs=epochs, batch_size=batch_size, early_stopping_threshold=10,
         track_metrics=True, track_adv_metrics=True
         )


# Evaluate the Model, plot metrics
# ---------------------------------------------------------------------------------------
test_accuracy, y_pred = cnn.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot metrics
cnn.plot_metrics()
cnn.plot_metrics(save_dir="examples/neural_networks/plots/neuralNetwork_layers_conv_cifar_metrics.png")