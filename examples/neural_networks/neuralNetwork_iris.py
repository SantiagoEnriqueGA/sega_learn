
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.neural_networks import *


def test_model(load_data_func, nn_layers, dropout_rate, reg_lambda, test_size=0.2):
    """Generic function to test the neural network model on a given dataset."""
    import random
    np.random.seed(42)
    random.seed(42)
    
    from sega_learn.utils import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    # Load the dataset
    data = load_data_func()
    X = data.data
    y = data.target

    print(f"\nTesting on {data.DESCR.splitlines()[0]} dataset:")
    print(f"--------------------------------------------------------------------------")
    print(f"X shape: {X.shape}, Y shape: {y.shape}")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    activations = ['relu'] * len(nn_layers) + ['sigmoid']

    # Initialize the neural network and optimizer
    nn = NeuralNetwork(nn_layers, dropout_rate=dropout_rate, reg_lambda=reg_lambda, activations=activations)
    optimizer = AdamOptimizer(learning_rate=0.0001)
    sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=5)  # Learning rate scheduler
    scheduler = lr_scheduler_plateau(sub_scheduler, patience=10, threshold=0.001)  # Learning rate scheduler
    
    # Train the neural network
    nn.train(X_train, y_train, X_test, y_test, optimizer, epochs=100, batch_size=32, early_stopping_threshold=100, lr_scheduler=scheduler)

    # Evaluate the neural network
    accuracy, predicted = nn.evaluate(X_test, y_test)
    print(f"Final Accuracy: {accuracy}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predicted, zero_division=0))
    
    print(f"End Neural Network State: \n{str(nn)}")

from sklearn.datasets import load_iris
test_model(load_iris, [4, 100, 25, 3], dropout_rate=0.1, reg_lambda=0.0, test_size=0.1)

