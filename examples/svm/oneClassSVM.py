import os
import sys
from math import ceil, floor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import make_sample_data
from sega_learn.svm import OneClassSVM


def plot_ellipse(mean, cov, color, ax, label):
    """Plot an ellipse representing the class conditional density."""
    v, w = linalg.eigh(
        cov
    )  # Compute eigenvalues and eigenvectors of the covariance matrix
    u = w[0] / linalg.norm(w[0])  # Normalize the eigenvector
    angle = np.arctan(u[1] / u[0])  # Compute the angle of the ellipse
    angle = 180 * angle / np.pi  # Convert to degrees

    ell = mpl.patches.Ellipse(
        xy=mean,
        width=2 * np.sqrt(np.abs(v[0])),
        height=2 * np.sqrt(np.abs(v[1])),
        angle=180 + angle,
        facecolor=color,
        edgecolor="black",
        label=label,
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.75)
    ax.add_artist(ell)


def fig_add_one_class_svm(X, y, ocsvm, ax):
    """Add a OneClassSVM plot to a figure."""
    ocsvm.fit(X)

    # Predict the labels using decision function
    decision_values = ocsvm.decision_function(X)
    y_pred = np.where(decision_values >= 0, 1, -1)  # Inliers (1), outliers (-1)

    # Plot the data points
    ax.scatter(X[:, 0], X[:, 1], color="blue", label="", alpha=0.2)

    # Plot the ellipse for class 1 (training class)
    mean_combined = np.mean(X, axis=0)
    cov_combined = np.cov(X, rowvar=False)
    plot_ellipse(mean_combined, cov_combined, "blue", ax, "Cond. Density")

    # Create a grid for plotting the decision boundary
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Compute decision function on the grid
    Z = ocsvm.decision_function(grid)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, levels=[-1e5, 0, 1e5], colors=["red", "blue"], alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")

    # Set axis limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()


# ---------------------------------------------------------------------
# OneClassSVM with Different Kernels
# ---------------------------------------------------------------------

cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0  # Covariance matrix for class 1
cov_class_2 = cov_class_1.T  # Covariance matrix for class 2

# Generate data
X, y = make_sample_data(
    n_samples=1000,
    n_features=2,
    cov_class_1=cov_class_1,
    cov_class_2=cov_class_2,
    shift=[4, 1],
    seed=1,
)

x_min, x_max = floor(np.min(X[:, 0])), ceil(np.max(X[:, 0]))
y_min, y_max = floor(np.min(X[:, 1])), ceil(np.max(X[:, 1]))

# Create subplots for different kernels
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Fit and plot OneClassSVM with RBF kernel
ocsvm_rbf = OneClassSVM(kernel="rbf", gamma="scale", learning_rate=1e-4)
fig_add_one_class_svm(X, y, ocsvm_rbf, axes[0])
axes[0].set_title("RBF Kernel")

# Fit and plot OneClassSVM with Polynomial kernel
ocsvm_poly = OneClassSVM(kernel="poly", degree=7, gamma="scale")
poly_acc = fig_add_one_class_svm(X, y, ocsvm_poly, axes[1])
axes[1].set_title("Polynomial Kernel")

# Fit and plot OneClassSVM with Sigmoid kernel
ocsvm_linear = OneClassSVM(kernel="sigmoid")
linear_acc = fig_add_one_class_svm(X, y, ocsvm_linear, axes[2])
axes[2].set_title("Sigmoid Kernel")

# Add overall title
plt.suptitle("OneClassSVM Decision Boundary Comparison per Kernel", fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig("examples/svm/plots/oneclasssvm_comparison.png", dpi=300)
