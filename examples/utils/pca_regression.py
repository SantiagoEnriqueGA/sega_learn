import numpy as np

np.random.seed(0)

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_regression
from sega_learn.utils.decomposition import PCA

# Example data
X, y = make_regression(n_samples=10_000, n_features=10, noise=25, random_state=0)

# Initialize PCA object
pca = PCA(n_components=8)

# Fit and transform the data
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")

print(f"Explained variance: {pca.get_explained_variance_ratio()}")


# Fit linear regression both on the original and reduced data
from sega_learn.linear_models import OrdinaryLeastSquares
from sega_learn.utils import Metrics

# Fit OLS on the original data
reg = OrdinaryLeastSquares(fit_intercept=True)
reg.fit(X, y)

r2 = round(Metrics.r_squared(y, reg.predict(X)), 2)
mse = round(Metrics.mean_squared_error(y, reg.predict(X)), 2)
print("\nOriginal data:\n" + "-" * 20)
print(f"R^2 Score: {r2:.2f}")
print(f"MSE:       {mse:.2f}")

# Fit OLS on the reduced data
reg_reduced = OrdinaryLeastSquares(fit_intercept=True)
reg_reduced.fit(X_reduced, y)
r2_reduced = round(Metrics.r_squared(y, reg_reduced.predict(X_reduced)), 2)
mse_reduced = round(Metrics.mean_squared_error(y, reg_reduced.predict(X_reduced)), 2)

print("\nReduced data:\n" + "-" * 20)
print(f"R^2 Score: {r2_reduced:.2f}")
print(f"MSE:       {mse_reduced:.2f}")
