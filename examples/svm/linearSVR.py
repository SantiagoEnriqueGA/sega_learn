import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.svm import *
from sega_learn.utils import make_regression

# Set random seed for reproducibility
np.random.seed(42)

# Generate a regression dataset
X, y = make_regression(n_samples=200, n_features=1, n_informative=1, noise=.1, random_state=42)

# # Add some outliers
outlier_indices = np.random.choice(len(X), size=10, replace=False)
y[outlier_indices] += np.random.normal(-3, 3, size=10)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Create and fit our LinearSVR model
svr = LinearSVR(C=0.01, epsilon=0.2, max_iter=1000, learning_rate=0.01)

# Can also use GeneralizedSVR with linear kernel
# svr = GeneralizedSVR(C=0.01, epsilon=0.2, max_iter=1000, learning_rate=0.01 ,kernel='linear')

# Fit the model
svr.fit(X_scaled, y_scaled)

# Evaluate the model
r2 = svr.score(X_scaled, y_scaled)
# print(f"LinearSVR - R²: {r2:.4f}")

# Create a mesh to plot regression line
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x_mesh = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
x_mesh_scaled = scaler_X.transform(x_mesh)

# Get predictions
y_mesh_scaled = svr.predict(x_mesh_scaled)
y_mesh = scaler_y.inverse_transform(y_mesh_scaled.reshape(-1, 1)).ravel()

# Get epsilon bounds
epsilon = svr.epsilon * scaler_y.scale_[0]  # Convert scaled epsilon back to original scale

# Create plot
plt.figure(figsize=(10, 8))
plt.scatter(X, y, c='b', label='Data')
plt.plot(x_mesh, y_mesh, color='g', linewidth=2, label='SVR model')

# Plot epsilon tube
plt.fill_between(
    x_mesh.ravel(),
    y_mesh - epsilon,
    y_mesh + epsilon,
    alpha=0.3,
    color='g',
    label=f'ε-bounds (ε={svr.epsilon:.2f})'
)

# Highlight outliers
plt.scatter(X[outlier_indices], y[outlier_indices],
            c='purple', s=80, marker='*', label='Added Outliers')

plt.xlim(x_min, x_max)
plt.title('LinearSVR Regression with ε-bounds')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend(loc='best')
plt.tight_layout()
# plt.show()
plt.savefig('examples/svm/plots/linear_svr.png', dpi=300)