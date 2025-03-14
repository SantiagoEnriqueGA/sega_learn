from sklearn.metrics import classification_report
import numpy as np
import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils import train_test_split
from sega_learn.neural_networks import *

# Hyperparameters
# ---------------------------------------------------------------------------------------
dropout = 0.1
reg_lambda = 0.0
lr = 0.00001
output_size = 3  # Number of classes
batch_size = 32
epochs = 10 # Reduced for demonstration, can increase.
input_channels = 1  # Grayscale images
image_height = 28
image_width = 28
num_samples = 100  # Total number of dummy images

# Generate Sample Data 
# ---------------------------------------------------------------------------------------
X = np.random.rand(num_samples, input_channels, image_height, image_width)
y = np.random.randint(0, output_size, num_samples)  # Labels 0, 1, or 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimizer and Scheduler
# ---------------------------------------------------------------------------------------
optimizer1 = AdamOptimizer(learning_rate=lr)
sub_scheduler1 = lr_scheduler_step(optimizer1, lr_decay=0.1, lr_decay_epoch=10)
scheduler1 = lr_scheduler_plateau(sub_scheduler1, patience=5, threshold=0.001)

# Layers
# ---------------------------------------------------------------------------------------
# Each ConvLayer will reduce the spatial dimensions by 4 pixels (2 pixels on each side)
# This is because the kernel size is 3 and padding is 0, so the output size is (input_size - kernel_size + 1)
#   For example, if input size is 28x28, after two ConvLayers with kernel_size=3 and padding=0,
#   the output size will be (28 - 3 + 1) = 26, and then (26 - 3 + 1) = 24.
#   After two ConvLayers, the output size will be 24x24.
layers = [
    ConvLayer(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=0, activation="relu"),
    ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, activation="relu"),
    FlattenLayer(),
    DenseLayer(64 * (image_height - 4) * (image_width - 4), 256, activation="relu"),  # Adjusted input size
    DenseLayer(256, output_size, activation="softmax"),  # Added output layer for 3 classes
]

# Initialize and Train Neural Network 
# ---------------------------------------------------------------------------------------
cnn = BaseBackendNeuralNetwork(layers=layers, dropout_rate=dropout, reg_lambda=reg_lambda)

cnn.train(X_train, y_train, X_test, y_test, optimizer=optimizer1, lr_scheduler=scheduler1,
         epochs=epochs, batch_size=batch_size, early_stopping_threshold=10,
         track_metrics=True, track_adv_metrics=True
         )

test_accuracy, y_pred = cnn.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

