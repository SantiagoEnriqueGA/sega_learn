import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_time_series
from sega_learn.utils.imputation import DirectionalImputer

# Example dataset with missing values (NaN)
X = make_time_series(
    n_samples=1,
    n_timestamps=30,
    n_features=1,
    trend="linear",
    seasonality="sine",
    seasonality_period=30,
    noise=0.1,
    random_state=1,
)[0]
nan_indices = np.random.choice(X.size, size=int(X.size * 0.2), replace=False)
X.ravel()[nan_indices] = np.nan
# Add nan to first and last rows
X[0, 0] = np.nan
X[-1, 0] = np.nan

print("Original Data:")
print(X)

# Example with forward fill
imputer_forward = DirectionalImputer(direction="forward")
imputer_forward.fit(X)
transformed_data_forward = imputer_forward.transform(X)
print("\nTransformed Data with Forward Fill:")
print(transformed_data_forward)

# Example with backward fill
imputer_backward = DirectionalImputer(direction="backward")
imputer_backward.fit(X)
transformed_data_backward = imputer_backward.transform(X)
print("\nTransformed Data with Backward Fill:")
print(transformed_data_backward)
