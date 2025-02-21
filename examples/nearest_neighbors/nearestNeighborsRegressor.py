import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.nearest_neighbors import *
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error

X, y = make_regression(n_samples=1000, n_features=5, n_informative=3, noise=0.1, random_state=42)

# Add a categorical feature to the dataset
X_categorical = np.random.choice(['A', 'B', 'C'], size=(X.shape[0], 1))
X = np.hstack((X, X_categorical))

# Instantiate the KNNRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=5, one_hot_encode=True)

# Fit the model
knn_regressor.fit(X, y)

# Make predictions
predictions = knn_regressor.predict(X)

# Print the regression metrics
print(f"R^2 Score: {r2_score(y, predictions):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y, predictions):.2f}")

