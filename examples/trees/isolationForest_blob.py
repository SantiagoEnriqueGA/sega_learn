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
threshold = 0.6
predictions = np.array([isolation_forest.predict(x, threshold) for x in X])

# Plot the Isolation Forest results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# Scatter plot of the data points colored by their anomaly score
scatter = ax1.scatter(X[:, 0], X[:, 1], c=anomaly_scores, s=50, cmap='viridis')
ax1.set_title('Isolation Forest Anomaly Detection')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
fig.colorbar(scatter, ax=ax1, label='Anomaly Score')

# Histogram of the anomaly scores with colored bins
n_bins = 50
hist, bins = np.histogram(anomaly_scores, bins=n_bins)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
colormap = plt.cm.viridis
norm = plt.Normalize(vmin=min(anomaly_scores), vmax=max(anomaly_scores))
colors = colormap(norm(bin_centers))

ax2.bar(bin_centers, hist, width=(bins[1] - bins[0]), color=colors, alpha=0.7)
ax2.set_title('Anomaly Scores Distribution')
ax2.set_xlabel('Anomaly Score')
ax2.set_ylabel('Frequency')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label='Threshold')
ax2.legend()

plt.tight_layout()
# plt.show()
plt.savefig('examples/trees/plots/isolation_forest_blob.png', dpi=300)















