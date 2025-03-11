from sklearn.metrics import classification_report
import numpy as np
import random


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils import train_test_split
from sega_learn.neural_networks import *
from sega_learn.utils import make_classification


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

# Select optimizers
optimizer1 = AdamOptimizer(learning_rate=lr)
optimizer2 = AdamOptimizer(learning_rate=lr)

# Select learning rate schedulers
sub_scheduler1 = lr_scheduler_step(optimizer1, lr_decay=0.1, lr_decay_epoch=10)  
scheduler1 = lr_scheduler_plateau(sub_scheduler1, patience=5, threshold=0.001)  
sub_scheduler2 = lr_scheduler_step(optimizer2, lr_decay=0.1, lr_decay_epoch=10)  
scheduler2 = lr_scheduler_plateau(sub_scheduler2, patience=5, threshold=0.001)


# Layer creation method #1: Provide a list of layer sizes and activation functions
# --------------------------------------------------------------------------------------------------------------------------
activations = ['relu'] * len(layers) + ['softmax']

# Initialize Neural Network
nn1 = BaseBackendNeuralNetwork(layers = [input_size] + layers + [output_size], activations=activations,
                              dropout_rate=dropout, reg_lambda=reg_lambda)


# Call the train method
nn1.train(X_train, y_train, X_test, y_test, optimizer=optimizer1, lr_scheduler=scheduler1, 
         epochs=100, batch_size=32, early_stopping_threshold=10, 
         track_metrics=True, track_adv_metrics=True,
         )

# Evaluate the Model
test_accuracy, y_pred = nn1.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# Layer creation method #2: Provide a list of Layer objects
# --------------------------------------------------------------------------------------------------------------------------
# Ensure no cary-over from previous method, delete the previous neural network

layers = [
        Layer(input_size, layers[0], activation="relu"),
        Layer(layers[0], layers[1], activation="relu"),
        Layer(layers[1], layers[2], activation="relu"),
        Layer(layers[2], output_size, activation="softmax"),
]

# Initialize Neural Network
nn2 = BaseBackendNeuralNetwork(layers=layers, dropout_rate=dropout, reg_lambda=reg_lambda)

# Call the train method
nn2.train(X_train, y_train, X_test, y_test, optimizer=optimizer2, lr_scheduler=scheduler2, 
         epochs=100, batch_size=32, early_stopping_threshold=10, 
         track_metrics=True, track_adv_metrics=True,
         )

# Evaluate the Model
test_accuracy, y_pred = nn2.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
