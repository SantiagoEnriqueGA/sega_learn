import numpy as np

np.random.seed(0)

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.utils import make_classification
from sega_learn.utils.decomposition import SVD

# Example data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    n_informative=9,
    n_redundant=1,
    random_state=0,
)

# Initialize SVD object
svd = SVD(n_components=5)

# Fit and transform the data
X_reduced = svd.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")

# Fit quadratic discriminant analysis both on the original and reduced data
from sega_learn.linear_models import QuadraticDiscriminantAnalysis
from sega_learn.utils import Metrics

# Fit QDA on the original data
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)
y_pred = qda.predict(X)

print("\nOriginal data:\n" + "-" * 20)
print(f"Accuracy: {Metrics.accuracy(y, y_pred):.2f}")
print(f"Precision: {Metrics.precision(y, y_pred):.2f}")

# Fit QDA on the reduced data
qda_reduced = QuadraticDiscriminantAnalysis()
qda_reduced.fit(X_reduced, y)
y_pred_reduced = qda_reduced.predict(X_reduced)

print("\nReduced data:\n" + "-" * 20)
print(f"Accuracy: {Metrics.accuracy(y, y_pred_reduced):.2f}")
print(f"Precision: {Metrics.precision(y, y_pred_reduced):.2f}")
