
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.clustering import *

from sega_learn.utils import make_blobs
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

# Generate synthetic data for testing
true_k = 8
X, y, _ = make_blobs(n_samples=1000, n_features=2, centers=true_k, cluster_std=0.60, random_state=1)

# Initialize DBSCAN object
eps = 0.5
min_samples = 10
dbscan = DBSCAN(X, eps=eps, min_samples=min_samples)

# Fit the DBSCAN model to the data
labels = dbscan.fit_predict()

# Calculate the silhouette score for evaluation
silhouette_score = dbscan.silhouette_score()
print(f'Silhouette Score: {silhouette_score}')

# # Can also use numba for faster computation (incurs overhead for compilation)
# dbscan = DBSCAN(X, eps=eps, min_samples=min_samples, compile_numba=True)

# # Fit the DBSCAN model to the data
# labels = dbscan.fit_predict()

# silhouette_score = dbscan.silhouette_score()
# print(f'Silhouette Score: {silhouette_score}')

# Plot the DBSCAN results
plt.figure(figsize=(8, 8))

# Scatter plot of the data points colored by their cluster label
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting core points and noise points
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.labels != -1] = True

# Plot core points
core_points = X[core_samples_mask]
plt.scatter(core_points[:, 0], core_points[:, 1], c=labels[core_samples_mask], s=50, cmap='viridis', edgecolors='k')

# Plot noise points
noise_points = X[~core_samples_mask]
plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=50, label='Noise')

plt.legend()
# plt.show()
plt.savefig('examples/clustering/plots/dbscan.png', dpi=300)