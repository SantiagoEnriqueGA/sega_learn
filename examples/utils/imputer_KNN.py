import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_regression
from sega_learn.utils.imputation import KNNImputer

# Example dataset with missing values (NaN)
X, y = make_regression(n_samples=10, n_features=2, noise=0.5)

# TODO: fix the issue with categorical features
# X_class = np.random.randint(0, 2, size=(10, 1))  # Binary classification labels
# # Convert to string for categorical feature
# X_class = np.array([str(i) for i in X_class.flatten()]).reshape(-1, 1)

# # Combine the two datasets
# X = np.hstack((X, X_class))

# Introduce missing values randomly
nan_indices = np.random.choice(X.size, size=int(X.size * 0.2), replace=False)
X.ravel()[nan_indices] = np.nan

print("Original Data:")
print(X)

# Example with KNN Imputer
imputer_knn = KNNImputer(n_neighbors=3)
transformed_data_knn = imputer_knn.fit_transform(X)
print("\nTransformed Data with KNN Imputer:")
print(transformed_data_knn)
