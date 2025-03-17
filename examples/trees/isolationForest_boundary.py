import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.trees import *
from sega_learn.linear_models import make_sample_data

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
import numpy as np
from math import log, floor, ceil
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

def fig_add_iso_forest(X, y, threshold, isolation_forest, ax):
    """Add a fitted isolation forest plot to a figure by threshold."""

    # Predict anomalies based on the anomaly scores
    y_pred = isolation_forest.predict(X, threshold)

    # Plot the data points
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='C1', alpha=0.2)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='C2', alpha=0.2)
    
    # Create the decision boundary, using the predict method to get the class labels
    # First, plot the decision boundary using pcolormesh
    
    DecisionBoundaryDisplay.from_estimator(isolation_forest,X,response_method="predict",plot_method="pcolormesh",
                                           ax=ax,cmap="RdBu",alpha=0.3,)
    # Next, plot the decision boundary using contour, with the decision boundary at the threshold
    DecisionBoundaryDisplay.from_estimator(isolation_forest,X,response_method="predict",plot_method="contour",
                                           ax=ax,alpha=1.0,levels=[threshold],)
    # Set the axis limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0     # Covariance matrix for class 1, scaled by 2.0
cov_class_2 = cov_class_1.T                                 # Covariance matrix for class 2, same as class 1 but transposed

# Generate data
X, y = make_sample_data(n_samples=1000, n_features=2, cov_class_1=cov_class_1, cov_class_2=cov_class_2, shift=[4,1], seed=1)
# Reshape y to be 1-dimensional
y = y.ravel()

x_min, x_max = floor(np.min(X[:, 0])), ceil(np.max(X[:, 0]))
y_min, y_max = floor(np.min(X[:, 1])), ceil(np.max(X[:, 1]))

# Initialize Isolation Forest object
n_trees = 100
max_samples = 256
max_depth = 20
threshold = 0.5
force_true_length = False
isolation_forest = IsolationForest(n_trees=n_trees, max_samples=max_samples, max_depth=max_depth, force_true_length=force_true_length)

# Fit the Isolation Forest model to the data
isolation_forest.fit(X)

# Plot just the threshold of 0.5
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
fig_add_iso_forest(X, y, threshold, isolation_forest, ax)
ax.set_title(f'Isolation Forest Decision Boundary') 
plt.tight_layout()
# plt.show()
plt.savefig('examples/trees/plots/isolationForest_boundary.png', dpi=300)














