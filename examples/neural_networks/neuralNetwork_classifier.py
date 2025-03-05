
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.neural_networks import *

def train_and_evaluate_model(X_train, X_test, y_train, y_test, 
                             layers, output_size, lr, dropout, reg_lambda, 
                             hidden_activation='relu', output_activation='softmax',
                             epochs=100, batch_size=32):
    """Function to train and evaluate the Neural Network"""
    
    input_size = X_train.shape[1]
    
    activations = [hidden_activation] * len(layers) + [output_activation]

    # Initialize Neural Network
    nn = NeuralNetwork([input_size] + layers + [output_size], dropout_rate=dropout, reg_lambda=reg_lambda, activations=activations)
    
    # Select optimizer
    optimizer = AdamOptimizer(learning_rate=lr)
    # optimizer = SGDOptimizer(learning_rate=lr, momentum=0.25, reg_lambda=0.1)
    # optimizer = AdadeltaOptimizer(learning_rate=lr, rho=0.95, epsilon=1e-6, reg_lambda=0.0)
    
    # Select learning rate scheduler
    sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)  
    scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)  
    # scheduler = lr_scheduler_exp(optimizer, lr_decay=0.1, lr_decay_epoch=10)  
    # scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10) 

    # Call the train method
    nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, lr_scheduler=scheduler, epochs=epochs, 
             batch_size=batch_size, early_stopping_threshold=10, 
             track_metrics=True,
             track_adv_metrics=True,
             )

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot metrics
    # nn.plot_metrics()
    # nn.plot_metrics(save_dir="examples/neural_networks/neuralNetwork_classifier_vanilla_metrics.png")


def train_and_evaluate_model_numba(X_train, X_test, y_train, y_test, 
                             layers, output_size, lr, dropout, reg_lambda, 
                             hidden_activation='relu', output_activation='softmax',
                             epochs=100, batch_size=32):
    """Function to train and evaluate the Neural Network"""
    
    input_size = X_train.shape[1]
    
    activations = [hidden_activation] * len(layers) + [output_activation]

    # Initialize Neural Network
    nn = NeuralNetwork(
        [input_size] + layers + [output_size], 
        dropout_rate=dropout, 
        reg_lambda=reg_lambda, 
        activations=activations,
        use_numba=True,
        compile_numba=True,  
        progress_bar=True,
    )
    
    # Select optimizer
    optimizer = JITAdamOptimizer(learning_rate=lr)
    # optimizer = JITSGDOptimizer(learning_rate=lr, momentum=0.25, reg_lambda=0.1)
    # optimizer = JITAdadeltaOptimizer(learning_rate=lr, rho=0.95, epsilon=1e-6, reg_lambda=0.0)
    
    # Select learning rate scheduler
    sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=25)  
    scheduler = lr_scheduler_plateau(sub_scheduler, patience=10, threshold=0.001)  
    # scheduler = lr_scheduler_exp(optimizer, lr_decay=0.1, lr_decay_epoch=10)  
    # scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10) 

    # Call the train method
    nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, lr_scheduler=scheduler, epochs=epochs, 
             batch_size=batch_size, early_stopping_threshold=10, 
             track_metrics=True,
             track_adv_metrics=True,
             )

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Plot metrics
    # nn.plot_metrics()
    # nn.plot_metrics(save_dir="examples/neural_networks/neuralNetwork_classifier_numba_metrics.png")

def main():
    import random
    np.random.seed(41)
    random.seed(41)
    
    # Define parameter grid and tuning ranges
    dropout = 0.1
    reg_lambda=  0.0
    lr = 0.0001
    layers = [250, 50, 25]
    output_size = 3

    X, y = make_classification(n_samples=3000, n_features=20, n_classes=3, n_informative=18,
                                random_state=42, class_sep=1.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    train_and_evaluate_model(X_train, X_test, y_train, y_test, 
                                layers, output_size, lr, dropout, reg_lambda, 
                                hidden_activation='relu', output_activation='softmax',
                                epochs=1000, batch_size=32)
    
    # To use the Numba backend:
    # train_and_evaluate_model_numba(X_train, X_test, y_train, y_test, 
    #                               layers, output_size, lr, dropout, reg_lambda, 
    #                               hidden_activation='relu', output_activation='softmax',
    #                               epochs=1000, batch_size=32)

if __name__ == "__main__":
    main()
