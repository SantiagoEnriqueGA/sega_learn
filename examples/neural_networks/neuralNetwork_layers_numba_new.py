from sklearn.metrics import classification_report
import numpy as np
import random

# from numba import config
# config.DISABLE_JIT = True

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils import train_test_split
from sega_learn.utils import make_classification
from sega_learn.neural_networks_numba_dev import *
from sega_learn.neural_networks_numba_dev.layers_jit_unified import JITLayer
from sega_learn.neural_networks_numba_dev.optimizers_jit import JITAdamOptimizer

# Define parameter grid and tuning ranges
dropout = 0.1
reg_lambda=  0.0
lr = 0.0001
layers = [250, 50, 25]
output_size = 3

# Get training and test data
X, y = make_classification(n_samples=3000, n_features=20, n_classes=3, n_informative=18,random_state=42, class_sep=.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_size = X_train.shape[1]

# Create optimizers and learning rate schedulers
# --------------------------------------------------------------------------------------------------------------------------
# Select optimizers
optimizer3 = JITAdamOptimizer(learning_rate=lr)

# Select learning rate schedulers
sub_scheduler3 = lr_scheduler_step(optimizer3, lr_decay=0.1, lr_decay_epoch=10)
scheduler3 = lr_scheduler_plateau(sub_scheduler3, patience=5, threshold=0.001)


# Layer creation method #2: Provide a list of JITLayer (unified) objects
# --------------------------------------------------------------------------------------------------------------------------
layers = [
        JITLayer("dense", input_size, layers[0], activation="relu"),
        JITLayer("dense", layers[0], layers[1], activation="relu"),
        JITLayer("dense", layers[1], layers[2], activation="relu"),
        JITLayer("dense", layers[2], output_size, activation="softmax"),
]

# Initialize Neural Network
nn3 = NumbaBackendNeuralNetwork(layers=layers, dropout_rate=dropout, reg_lambda=reg_lambda, compile_numba=True)

# Call the train method
nn3.train(X_train, y_train, X_test, y_test, optimizer=optimizer3, lr_scheduler=scheduler3, 
         epochs=100, batch_size=32, early_stopping_threshold=10, 
         track_metrics=True, track_adv_metrics=True,
         )

# Evaluate the Model
test_accuracy, y_pred = nn3.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

nn3.plot_metrics()