from sklearn.metrics import classification_report
import numpy as np
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils import train_test_split
from sega_learn.utils import make_classification
from sega_learn.neural_networks import *

# Define parameter grid and tuning ranges
dropout = 0.1
reg_lambda=  0.0
lr = 0.0001
layers = [250, 50, 25]

# Get training and test data
X, y = make_classification(n_samples=3000, n_features=20, n_classes=3, n_informative=18,random_state=42, class_sep=.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

print(f"Input size: {input_size}, Output size: {output_size}")

# Function to train and evaluate a model with a given optimizer
def train_and_evaluate(optimizer, lr_scheduler, optimizer_name):
    print(f"\n--- Training with {optimizer_name} ---")

    layers_list = [
        DenseLayer(input_size, layers[0], activation="relu"),
        DenseLayer(layers[0], layers[1], activation="relu"),
        DenseLayer(layers[1], layers[2], activation="relu"),
        DenseLayer(layers[2], output_size, activation="softmax" if output_size > 2 else "sigmoid"),
    ]

    nn = BaseBackendNeuralNetwork(layers=layers_list, dropout_rate=dropout, reg_lambda=reg_lambda)

    nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, lr_scheduler=lr_scheduler, 
             epochs=100, batch_size=32, early_stopping_threshold=10, 
             track_metrics=True, track_adv_metrics=True,
             save_animation=True, save_path=f"examples/neural_networks/plots/neuralNetwork_classifier_{optimizer_name}.mp4",
             fps=1, dpi=100, frame_every=1,
             )

    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def train_and_evaluate_numba(optimizer, lr_scheduler, optimizer_name):
    print(f"\n--- Training Numba NN with {optimizer_name} ---")

    layers_list = [
        JITDenseLayer(input_size, layers[0], activation="relu"),
        JITDenseLayer(layers[0], layers[1], activation="relu"),
        JITDenseLayer(layers[1], layers[2], activation="relu"),
        JITDenseLayer(layers[2], output_size, activation="softmax" if output_size > 2 else "sigmoid"),
    ]

    nn = NumbaBackendNeuralNetwork(layers=layers_list, dropout_rate=dropout, reg_lambda=reg_lambda, compile_numba=True)

    nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, lr_scheduler=lr_scheduler, 
             epochs=100, batch_size=32, early_stopping_threshold=10, 
             track_metrics=True, track_adv_metrics=True,
             save_animation=True, save_path=f"examples/neural_networks/plots/neuralNetwork_classifier_{optimizer_name}_numba.mp4",
             fps=1, dpi=100, frame_every=1,
             )

    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# AdamOptimizer
# --------------------------------------------------------------------------------------------------------------------------
# Base backend
optimizer_adam = AdamOptimizer(learning_rate=lr)
sub_scheduler_adam = lr_scheduler_step(optimizer_adam, lr_decay=0.1, lr_decay_epoch=10)
scheduler_adam = lr_scheduler_plateau(sub_scheduler_adam, patience=5, threshold=0.001)
train_and_evaluate(optimizer_adam, scheduler_adam, "AdamOptimizer")

# Numba backend
# nb_optimizer_adam = JITAdamOptimizer(learning_rate=lr)
# nb_sub_scheduler_adam = lr_scheduler_step(nb_optimizer_adam, lr_decay=0.1, lr_decay_epoch=10)
# nb_scheduler_adam = lr_scheduler_plateau(nb_sub_scheduler_adam, patience=5, threshold=0.001)
# train_and_evaluate_numba(nb_optimizer_adam, nb_scheduler_adam, "AdamOptimizer")

# # SGDOptimizer
# # --------------------------------------------------------------------------------------------------------------------------
# optimizer_sgd = SGDOptimizer(learning_rate=lr)
# sub_scheduler_sgd = lr_scheduler_step(optimizer_sgd, lr_decay=0.1, lr_decay_epoch=10)
# scheduler_sgd = lr_scheduler_plateau(sub_scheduler_sgd, patience=5, threshold=0.001)
# train_and_evaluate(optimizer_sgd, scheduler_sgd, "SGDOptimizer")

# Numba backend
# nb_optimizer_sgd = JITSGDOptimizer(learning_rate=lr)
# nb_sub_scheduler_sgd = lr_scheduler_step(nb_optimizer_sgd, lr_decay=0.1, lr_decay_epoch=10)
# nb_scheduler_sgd = lr_scheduler_plateau(nb_sub_scheduler_sgd, patience=5, threshold=0.001)
# train_and_evaluate_numba(nb_optimizer_sgd, nb_scheduler_sgd, "SGDOptimizer")

# # AdadeltaOptimizer
# # --------------------------------------------------------------------------------------------------------------------------
# optimizer_adadelta = AdadeltaOptimizer(learning_rate=lr)
# sub_scheduler_adadelta = lr_scheduler_step(optimizer_adadelta, lr_decay=0.1, lr_decay_epoch=10)
# scheduler_adadelta = lr_scheduler_plateau(sub_scheduler_adadelta, patience=5, threshold=0.001)
# train_and_evaluate(optimizer_adadelta, scheduler_adadelta, "AdadeltaOptimizer")

# Numba backend
# nb_optimizer_adadelta = JITAdadeltaOptimizer(learning_rate=lr)
# nb_sub_scheduler_adadelta = lr_scheduler_step(nb_optimizer_adadelta, lr_decay=0.1, lr_decay_epoch=10)
# nb_scheduler_adadelta = lr_scheduler_plateau(nb_sub_scheduler_adadelta, patience=5, threshold=0.001)
# train_and_evaluate_numba(nb_optimizer_adadelta, nb_scheduler_adadelta, "AdadeltaOptimizer")