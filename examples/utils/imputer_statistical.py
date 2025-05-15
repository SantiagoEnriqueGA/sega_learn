import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_regression
from sega_learn.utils.imputation import StatisticalImputer

# Example dataset with missing values (NaN)
X, y = make_regression(n_samples=10, n_features=1, noise=0.5)
nan_indices = np.random.choice(X.size, size=int(X.size * 0.2), replace=False)
X.ravel()[nan_indices] = np.nan

print("Original Data:")
print(X)

# Example with mean strategy
imputer_mean = StatisticalImputer(strategy="mean")
imputer_mean.fit(X)
transformed_data_mean = imputer_mean.transform(X)
print(f"\nTransformed Data with Mean Strategy, statistic_: {imputer_mean.statistic_}:")
print(transformed_data_mean)

# Example with median strategy
imputer_median = StatisticalImputer(strategy="median")
imputer_median.fit(X)
transformed_data_median = imputer_median.transform(X)

print(
    f"\nTransformed Data with Median Strategy, statistic_: {imputer_median.statistic_}:"
)
print(transformed_data_median)

# Example with mode strategy
imputer_mode = StatisticalImputer(strategy="mode")
imputer_mode.fit(X)
transformed_data_mode = imputer_mode.transform(X)

print(f"\nTransformed Data with Mode Strategy, statistic_: {imputer_mode.statistic_}:")
print(transformed_data_mode)
