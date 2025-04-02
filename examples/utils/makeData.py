import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_blobs, make_classification, make_regression

# Example: Generate regression data
X, y = make_regression(n_samples=100, n_features=3, noise=0.5)
print("Regression data:\n", "-" * 75)
print(f"X: {X.shape}, y: {y.shape}")
print(f"X: {X.dtype}, y: {y.dtype}")
print(f"First 5 rows of X: \n{X[:5]}")
print(f"First 5 rows of y: \n{y[:5]}")

# Example: Generate classification data
X, y = make_classification(n_samples=200, n_classes=3, n_features=5)
print("\n\nClassification data:\n", "-" * 75)
print(f"X: {X.shape}, y: {y.shape}")
print(f"X: {X.dtype}, y: {y.dtype}")
print(f"First 5 rows of X: \n{X[:5]}")
print(f"First 5 rows of y: \n{y[:5]}")

# # Example: Generate clustering data
X, y, centers = make_blobs(n_samples=300, n_features=2, centers=4)
print("\n\nClustering data:\n", "-" * 75)
print(f"X: {X.shape}, y: {y.shape}")
print(f"X: {X.dtype}, y: {y.dtype}")
print(f"First 5 rows of X: \n{X[:5]}")
print(f"First 5 rows of y: \n{y[:5]}")
print(f"First 5 rows of centers: \n{centers[:5]}")
