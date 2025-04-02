import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from math import ceil, floor

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sega_learn.linear_models import *
from sega_learn.linear_models import make_sample_data


def plot_ellipse(mean, cov, color, ax, label):
    """Plot an ellipse representing the class conditional density."""
    v, w = linalg.eigh(
        cov
    )  # Compute the eigenvalues and eigenvectors of the covariance matrix
    u = w[0] / linalg.norm(
        w[0]
    )  # Compute the normalized eigenvector corresponding to the largest eigenvalue
    angle = np.arctan(u[1] / u[0])  # Compute the angle of the ellipse
    angle = 180 * angle / np.pi  # Convert the angle to degrees (from radians)

    ell = mpl.patches.Ellipse(
        xy=mean,
        width=2 * np.sqrt(np.abs(v[0])),
        height=2 * np.sqrt(np.abs(v[1])),
        angle=180 + angle,
        facecolor=color,
        edgecolor="black",
        label=label,
    )
    ell.set_clip_box(ax.bbox)  # Set the clipping box
    ell.set_alpha(0.75)
    ax.add_artist(ell)


def fig_add_discriminant_analysis(X, y, da, ax):
    """Add a discriminant analysis plot to a figure."""
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.metrics import accuracy_score

    da.fit(X, y)

    # Predict the labels
    y_pred = da.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)

    # Plot the data points
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="C1", alpha=0.2)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="C2", alpha=0.2)

    # Plot the ellipses representing the class conditional densities
    plot_ellipse(np.mean(X[y == 0], axis=0), cov_class_1, "red", ax, "C1 Cond. Density")
    plot_ellipse(
        np.mean(X[y == 1], axis=0), cov_class_2, "blue", ax, "C2 Cond. Density"
    )

    # Create the decision boundary, using the predict method to get the class labels
    # First, plot the decision boundary using pcolormesh
    DecisionBoundaryDisplay.from_estimator(
        da,
        X,
        response_method="predict",
        plot_method="pcolormesh",
        ax=ax,
        cmap="RdBu",
        alpha=0.3,
    )
    # Next, plot the decision boundary using contour, with the decision boundary at 0.5 (i.e., the point where the two classes are equally likely)
    DecisionBoundaryDisplay.from_estimator(
        da,
        X,
        response_method="predict",
        plot_method="contour",
        ax=ax,
        alpha=1.0,
        levels=[0.5],
    )
    # Set the axis limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()

    return accuracy


# ---------------------------------------------------------------------
# LDA SVD vs LSQR vs EIGEN
# ---------------------------------------------------------------------

cov_class_1 = (
    np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
)  # Covariance matrix for class 1, scaled by 2.0
cov_class_2 = (
    cov_class_1.T
)  # Covariance matrix for class 2, same as class 1 but transposed

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

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Fit and plot LDA model with SVD solver
lda_svd = LinearDiscriminantAnalysis(solver="svd")
lda_svd_acc = fig_add_discriminant_analysis(X, y, lda_svd, axes[0])
axes[0].set_title(f"SVD Decision Boundary (Accuracy: {lda_svd_acc:.2f})")

# Fit and plot LDA model with LSQR solver
lda_lsqr = LinearDiscriminantAnalysis(solver="lsqr")
lda_lsqr_acc = fig_add_discriminant_analysis(X, y, lda_lsqr, axes[1])
axes[1].set_title(f"LSQR Decision Boundary (Accuracy: {lda_lsqr_acc:.2f})")

# Fit and plot LDA model with EIGEN solver
lda_eigen = LinearDiscriminantAnalysis(solver="eigen")
lda_eigen_acc = fig_add_discriminant_analysis(X, y, lda_eigen, axes[2])
axes[2].set_title(f"EIGEN Decision Boundary (Accuracy: {lda_eigen_acc:.2f})")

# Add Overall Title
plt.suptitle("LDA Decision Boundary Comparison per Solver", fontsize=16)
plt.tight_layout()
plt.savefig(r"examples\linear_models\plots\lda_comparison.png")
# plt.show()
