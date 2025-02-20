
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.neural_networks import NeuralNetwork, AdamOptimizer, SGDOptimizer, AdadeltaOptimizer
from sega_learn.neural_networks import lr_scheduler_exp, lr_scheduler_step, lr_scheduler_plateau

def load_pima_diabetes_data(file_path):
    """Function to load and preprocess Pima Indians Diabetes dataset"""
    df = pd.read_csv(file_path)
    X = df.drop('y', axis=1).to_numpy()
    y = df['y'].to_numpy().reshape(-1, 1)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X, y, X_train, X_test, y_train, y_test

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
    nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, lr_scheduler=scheduler,epochs=epochs, batch_size=batch_size, early_stopping_threshold=10)

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def run_diabetes():
    import random
    np.random.seed(41)
    random.seed(41)
    
    # Define parameter grid and tuning ranges
    dropout = 0.1
    reg_lambda=  0.0
    lr = 0.0001
    layers = [100, 50, 25] 
    output_size = 1

    # Train and evaluate on Pima Indians Diabetes dataset
    print("\n--- Training on Pima Indians Diabetes Dataset ---")
    
    X, y, X_train, X_test, y_train, y_test = load_pima_diabetes_data("example_datasets/pima-indians-diabetes_prepared.csv")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, 
                                layers, output_size, lr, dropout, reg_lambda, 
                                hidden_activation='tanh', output_activation='softmax',
                                epochs=1000, batch_size=32)

if __name__ == "__main__":
    run_diabetes()
