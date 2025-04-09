import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import *
from sega_learn.utils import Scaler, make_classification
from sega_learn.utils.metrics import Metrics

# Set random seed for reproducibility
np.random.seed(42)

# Generate a binary classification dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=2,
    random_state=42,
    n_clusters_per_class=1,
    class_sep=1,
)

# Scale features
scaler = Scaler()
X_scaled = scaler.fit_transform(X)

# Train Perceptron model
log_reg = Perceptron(learning_rate=0.01, max_iter=1_000)
log_reg.fit(X_scaled, y)

# Evaluate the model
y_pred = log_reg.predict(X_scaled)
accuracy = Metrics.accuracy(y, y_pred)

# Create mesh to plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Scale mesh points
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)

# Predict on mesh points
Z = log_reg.predict(mesh_points_scaled)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors="k")
plt.title(f"Perceptron: Accuracy={accuracy:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)

# Save the plot
plt.tight_layout()
plt.savefig("examples/linear_models/plots/perceptron_binary.png", dpi=300)
# plt.show()


# Generate a multi-class classification dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=3,
    random_state=42,
    n_clusters_per_class=1,
    class_sep=2,
)

# Scale features
scaler = Scaler()
X_scaled = scaler.fit_transform(X)

# Train Perceptron model
log_reg = Perceptron(learning_rate=0.01, max_iter=1_000)
log_reg.fit(X_scaled, y)

# Evaluate the model
y_pred = log_reg.predict(X_scaled)
accuracy = Metrics.accuracy(y, y_pred)

# Create mesh to plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Scale mesh points
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)

# Predict on mesh points
Z = log_reg.predict(mesh_points_scaled)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors="k")
plt.title(f"Perceptron: Accuracy={accuracy:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, alpha=0.3)

# Save the plot
plt.tight_layout()
plt.savefig("examples/linear_models/plots/perceptron_multiclass.png", dpi=300)
# plt.show()
