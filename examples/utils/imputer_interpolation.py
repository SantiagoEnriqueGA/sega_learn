import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_regression
from sega_learn.utils.imputation import InterpolationImputer

# Example dataset with missing values (NaN)
X, y = make_regression(n_samples=10, n_features=1, noise=0.5)
nan_indices = np.random.choice(X.size, size=int(X.size * 0.2), replace=False)
X.ravel()[nan_indices] = np.nan

print("Original Data:")
print(X)

# Example with Linear Interpolation
imputer_linear = InterpolationImputer(method="linear")
imputer_linear.fit(X)
transformed_data_linear = imputer_linear.transform(X)
print("\nTransformed Data with Linear Interpolation:")
print(transformed_data_linear)

# Example with Polynomial Interpolation
imputer_polynomial = InterpolationImputer(method="polynomial", degree=2)
imputer_polynomial.fit(X)
transformed_data_polynomial = imputer_polynomial.transform(X)
print("\nTransformed Data with Polynomial Interpolation:")
print(transformed_data_polynomial)

# Example with Polynomial Interpolation with degree 3
imputer_polynomial_degree3 = InterpolationImputer(method="polynomial", degree=3)
imputer_polynomial_degree3.fit(X)
transformed_data_polynomial_degree3 = imputer_polynomial_degree3.transform(X)
print("\nTransformed Data with Polynomial Interpolation with degree 3:")
print(transformed_data_polynomial_degree3)
