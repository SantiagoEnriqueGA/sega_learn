import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.svm import *
from sega_learn.utils import make_classification
from sega_learn.utils import Scaler

# Set random seed for reproducibility
np.random.seed(42)

# Generate a binary classification dataset
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                         random_state=42, n_clusters_per_class=1, class_sep=1.5)

# Convert labels to -1 and 1
y = np.where(y == 0, -1, 1)

# Scale features
scaler = Scaler()
X_scaled = scaler.fit_transform(X)

# Define kernel parameters
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_params = {
    'linear': {},
    'poly': {'degree': 3, 'gamma': 0.1, 'coef0': 1.0},
    'rbf': {'gamma': 0.5},
    'sigmoid': {'gamma': 0.1, 'coef0': 1.0}
}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Train models and plot decision boundaries
for i, kernel in enumerate(kernels):
    params = kernel_params[kernel]
    svc = GeneralizedSVC(C=1.0, tol=1e-4, max_iter=2000, kernel=kernel, **params)
    svc.fit(X_scaled, y)
    
    # Evaluate the model
    accuracy = svc.score(X_scaled, y)

    # Create mesh to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Scale mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Predict on mesh points
    Z = svc.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax = axes[i // 2, i % 2]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    ax.set_title(f'{kernel} Kernel: Accuracy={accuracy:.4f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
# plt.show()
plt.savefig('examples/svm/plots/generalizedSVC_binary.png', dpi=300)