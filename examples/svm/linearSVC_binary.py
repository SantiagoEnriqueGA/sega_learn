import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.svm import *
from sega_learn.utils import Scaler, make_classification

# Set random seed for reproducibility
np.random.seed(42)

try:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
    )
except:
    from sega_learn.utils import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=10,
        class_sep=1.5,
    )

# Convert labels to -1 and 1
y = np.where(y == 0, -1, 1)

# Scale features
scaler = Scaler()
X_scaled = scaler.fit_transform(X)

# Create and fit our LinearSVC model
svc = LinearSVC(C=0.0, tol=1e-4, max_iter=2000, learning_rate=0.01)
svc.fit(X_scaled, y)

# Evaluate the model
accuracy = svc.score(X_scaled, y)
# print(f"LinearSVC - Accuracy: {accuracy:.4f}")

# Create a mesh to plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Scale mesh points
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_scaled = scaler.transform(mesh_points)

# Predict on mesh points to create decision boundary
Z = svc.predict(mesh_points_scaled)
Z = Z.reshape(xx.shape)

# Create plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors="k")

# Plot the SVM decision boundary (the line where w·x + b = 0)
w_original = svc.w
b_original = svc.b

# Transform w back to the original feature space
w = scaler.std * w_original
b = b_original - np.dot(scaler.mean, w_original)

# Plot the decision boundary line
slope = -w[0] / w[1]
intercept = -b / w[1]
x_support = np.linspace(x_min, x_max, 2)
y_support = slope * x_support + intercept
plt.plot(x_support, y_support, "k-", linewidth=2)

# Plot the margins
margin_width = 1 / np.linalg.norm(w)
margin_top = y_support + margin_width
margin_bottom = y_support - margin_width
plt.plot(x_support, margin_top, "k--", linewidth=1)
plt.plot(x_support, margin_bottom, "k--", linewidth=1)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title(f"LinearSVC Decision Boundary - Accuracy: {accuracy:.4f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
# plt.show()
plt.savefig("examples/svm/plots/linear_svc_binary.png", dpi=300)
