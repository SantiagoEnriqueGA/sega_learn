import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.svm import *
from sega_learn.utils.makeData import make_blobs

# Random seed for reproducibility
np.random.seed(42)

# Generate classification data
X, y, _ = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=3,
    random_state=42,
    cluster_std=3.0,
    center_box=(-15.0, 15.0),
)

# Define kernel parameters
kernels = ["linear", "poly", "rbf", "sigmoid"]
colors = ["red", "purple", "orange", "blue"]
kernel_params = {
    "linear": {},
    "poly": {"degree": 3, "gamma": 0.1, "coef0": 1.0},
    "rbf": {"gamma": 0.5},
    "sigmoid": {"gamma": 0.1, "coef0": 1.0},
}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
cmap = ListedColormap(["blue", "green", "yellow"])

# Train models and plot decision boundaries
for i, kernel in enumerate(kernels):
    params = kernel_params[kernel]
    svc = GeneralizedSVC(C=1.0, tol=1e-4, max_iter=1000, kernel=kernel, **params)
    svc.fit(X, y)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict on mesh grid
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax = axes[i // 2, i % 2]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=cmap)
    ax.set_title(f"{kernel} Kernel")
    ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
# plt.show()
plt.savefig("examples/svm/plots/generalizedSVC_kernels_multi.png", dpi=300)
