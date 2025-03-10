import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import random
from sklearn.metrics import classification_report

from sega_learn.utils import train_test_split
from sega_learn.utils import make_classification
from sega_learn.neural_networks import *

np.random.seed(41)
random.seed(41)

# Define parameter grid and tuning ranges
dropout = 0.1
reg_lambda=  0.0
lr = 0.0001
layers = [250, 50, 25]
output_size = 3

X, y = make_classification(n_samples=3000, n_features=20, n_classes=3, n_informative=18,random_state=42, class_sep=.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
activations = ['relu'] * len(layers) + ['softmax']

# Initialize Neural Network
# nn = BaseBackendNeuralNetwork([input_size] + layers + [output_size], dropout_rate=dropout, reg_lambda=reg_lambda, activations=activations)

layers = [
        Layer(input_size, layers[0], activation="relu"),
        Layer(layers[0], layers[1], activation="relu"),
        Layer(layers[1], layers[2], activation="relu"),
        Layer(layers[2], output_size, activation="softmax"),
]
nn = BaseBackendNeuralNetwork(layers=layers, dropout_rate=dropout, reg_lambda=reg_lambda)

# Select optimizer
optimizer = AdamOptimizer(learning_rate=lr)

# Select learning rate scheduler
sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)  
scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)  

# Call the train method
nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, lr_scheduler=scheduler, 
         epochs=100, batch_size=32, early_stopping_threshold=10, 
         track_metrics=True, track_adv_metrics=True,
        )

# Evaluate the Model
test_accuracy, y_pred = nn.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Plot metrics
# nn.plot_metrics()
# nn.plot_metrics(save_dir="examples/neural_networks/neuralNetwork_classifier_vanilla_metrics.png")
