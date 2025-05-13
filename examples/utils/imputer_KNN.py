import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_regression
from sega_learn.utils.imputation import KNNImputer

np.random.seed(42)
X_num, y = make_regression(n_samples=10, n_features=2, noise=0.5, random_state=42)

# Create categorical feature as strings
X_cat_val = np.random.randint(0, 2, size=(10, 1))
X_cat = np.array(["feat1" if i == 0 else "feat2" for i in X_cat_val.flatten()]).reshape(
    -1, 1
)

# Combine using pandas DataFrame first to preserve types, then convert to object array
df = pd.DataFrame(X_num, columns=["num_feat1", "num_feat2"])
df["cat_feat"] = X_cat
X = df.to_numpy(dtype=object)  # Convert to NumPy array

# Introduce np.nan (float NaN) randomly
nan_indices = np.random.choice(X.size, size=int(X.size * 0.2), replace=False)
row_idx, col_idx = np.unravel_index(nan_indices, X.shape)
X[row_idx, col_idx] = np.nan

print("Original Data (Corrected Types):")
print(X)

# Example with KNN Imputer
imputer_knn = KNNImputer(n_neighbors=3)

print("\nAttempting KNN Imputation...")
transformed_data_knn = imputer_knn.fit_transform(X)
print("\nTransformed Data with KNN Imputer:")
print(transformed_data_knn)
