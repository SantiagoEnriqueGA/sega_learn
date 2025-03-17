
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.trees import *

from sega_learn.utils import make_blobs
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

# Generate synthetic data for testing
true_k = 5
X, y, _ = make_blobs(n_samples=1000, n_features=2, centers=true_k, cluster_std=0.60, random_state=1)

# Initialize Isolation Forest object
n_trees = 100
max_samples = 256
max_depth = 20
force_true_length = False
isolation_forest = IsolationForest(n_trees=n_trees, max_samples=max_samples, max_depth=max_depth, force_true_length=force_true_length)

# Fit the Isolation Forest model to the data
isolation_forest.fit(X)

# Calculate the anomaly scores for the data points
anomaly_scores = np.array([isolation_forest.anomaly_score(x) for x in X])

# Predict anomalies based on the anomaly scores
threshold = 0.5
predictions = np.array([isolation_forest.predict(x, threshold) for x in X])

# Plot the Isolation Forest results
plt.figure(figsize=(8, 8))

# Scatter plot of the data points colored by their anomaly score
plt.scatter(X[:, 0], X[:, 1], c=anomaly_scores, s=50, cmap='viridis')
plt.title(f'Isolation Forest Anomaly Detection')  
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Anomaly Score')
# plt.show()
plt.savefig('examples/trees/plots/isolation_forest_blob.png', dpi=300)















