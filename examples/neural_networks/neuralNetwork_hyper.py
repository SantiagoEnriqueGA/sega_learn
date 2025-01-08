import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.neural_networks import NeuralNetwork, AdamOptimizer

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

def load_breast_prognostic_data(file_path):
    """Function to load and preprocess Wisconsin Breast Prognostic dataset"""
    df = pd.read_csv(file_path)
    X = df.drop('diagnosis', axis=1).to_numpy()
    y = df['diagnosis'].to_numpy().reshape(-1, 1)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X, y, X_train, X_test, y_train, y_test

def hyper_train_and_evaluate_model(X, y ,X_train, X_test, y_train, y_test, 
                                   param_grid, num_layers_range, layer_size_range, 
                                   lr_range, epochs=100, batch_size=32):
    """Function to train and evaluate the Neural Network with hyperparameter tuning"""
    
    input_size = X_train.shape[1]
    output_size = 1

    # Initialize Neural Network
    nn = NeuralNetwork([input_size] + [100, 50, 25] + [output_size], dropout_rate=0.5, reg_lambda=0.0)
    optimizer = AdamOptimizer(learning_rate=0.0001)

    # Hyperparameter tuning with Adam optimizer
    best_params, best_accuracy = nn.tune_hyperparameters(
        param_grid,
        num_layers_range,
        layer_size_range,
        output_size,
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer_type='Adam',
        lr_range=lr_range,
        epochs=epochs,
        batch_size=batch_size
    )

    print(f"Best parameters: {best_params} with accuracy: {best_accuracy:.4f}")

    # Train the final model with best parameters
    nn = NeuralNetwork([input_size] + [best_params['layer_size']] * best_params['num_layers'] + [output_size], 
                       dropout_rate=best_params['dropout_rate'], 
                       reg_lambda=best_params['reg_lambda'])
    nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, epochs=epochs, batch_size=batch_size)

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def main(diabetes=True, cancer=True, test_case=False):
    # Define parameter grid and tuning ranges
    param_grid = {
        'dropout_rate': [0.1, 0.2, 0.3],
        'reg_lambda': [0.0, 0.01]
    }
    num_layers_range = (3, 5, 1)        # min, max, step
    layer_size_range = (25, 100, 25)    # min, max, step
    lr_range = (1e-5, 0.01, 3)          # (min_lr, max_lr, num_steps)

    if test_case:
        param_grid = {
            'dropout_rate': [0.1],
            'reg_lambda': [0.0]
        }
        num_layers_range = (3, 4, 1)
        layer_size_range = (25, 50, 25)
        lr_range = (1e-5, 1e-5, 1)

    if diabetes:
        # Train and evaluate on Pima Indians Diabetes dataset
        print("\n--- Training on Pima Indians Diabetes Dataset ---")
        
        X, y, X_train, X_test, y_train, y_test = load_pima_diabetes_data("example_datasets/pima-indians-diabetes_prepared.csv")
        hyper_train_and_evaluate_model(X, y, X_train, X_test, y_train, y_test, 
                                       param_grid, num_layers_range, 
                                       layer_size_range, lr_range)

    if cancer:
        # Train and evaluate on Wisconsin Breast Prognostic dataset
        print("\n--- Training on Wisconsin Breast Prognostic Dataset ---")
        
        X, y, X_train, X_test, y_train, y_test = load_breast_prognostic_data("example_datasets/Wisconsin_breast_prognostic.csv")
        hyper_train_and_evaluate_model(X, y, X_train, X_test, y_train, y_test, 
                                       param_grid, num_layers_range, 
                                       layer_size_range, lr_range)

    if not diabetes and not cancer:
        print("Please select at least one dataset to train and evaluate.")

if __name__ == "__main__":
    main(diabetes=True, cancer=True)
