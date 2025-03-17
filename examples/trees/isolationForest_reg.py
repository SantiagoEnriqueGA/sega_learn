
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.trees import *

from sega_learn.utils import make_regression
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

# Generate synthetic regression data for testing
X, y, = make_regression(n_samples=1000, n_features=1, noise=0.5, random_state=42)

# Initialize Isolation Forest object
n_trees = 100
max_samples = 256
max_depth = 10
force_true_length = False
isolation_forest = IsolationForest(n_trees=n_trees, max_samples=max_samples, max_depth=max_depth, force_true_length=force_true_length)

# Fit the Isolation Forest model to the data
isolation_forest.fit(X)

# Calculate the anomaly scores for the data points
anomaly_scores = np.array([isolation_forest.anomaly_score(x) for x in X])

# Predict anomalies based on the anomaly scores
threshold = 0.5
predictions = np.array([isolation_forest.predict(x, threshold) for x in X])

# Plot the Isolation Forest anomaly results
plt.figure(figsize=(8,8))

# Scatter plot of the data points colored by their anomaly score
plt.scatter(X, y, c=anomaly_scores, s=50, cmap='viridis')
plt.title(f'Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Target Variable')
plt.colorbar(label='Anomaly Score')
# plt.show()
plt.savefig('examples/trees/plots/isolation_forest_reg.png', dpi=300)









