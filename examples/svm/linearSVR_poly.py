import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils import PolynomialTransform
from sega_learn.svm import *
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)  # Random seed for reproducibility

pnts = 300
X = np.linspace(0, 5, pnts)
y = 10 * np.sin(X) + np.random.normal(0, 1, pnts)


# Normalize the data
X_norm = (X - np.mean(X)) / np.std(X)
y_norm = (y - np.mean(y)) / np.std(y)


# Create and fit SVR model
svr = GeneralizedSVR(C=1.0, tol=1e-3, max_iter=1000, learning_rate=0.001, kernel='poly', degree=3)
# svr = GeneralizedSVR(C=1.0, tol=1e-3, max_iter=1000, learning_rate=0.001, kernel='rbf')
# svr = GeneralizedSVR(C=1.0, tol=1e-3, max_iter=1000, learning_rate=0.001, kernel='sigmoid')
svr.fit(X_norm.reshape(-1, 1), y_norm)
y_pred = svr.predict(X_norm.reshape(-1, 1))

# print(f"y_pred: {y_pred}")


# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, color='blue', label='Data Points', alpha=0.5)
# plt.plot(X, y_pred, color='red', label='SVR Prediction', linewidth=2)
# plt.plot(X, 10 * np.sin(X), color='green', label='True Function', linewidth=2)
# plt.xlabel('Feature 0')
# plt.ylabel('Target')
# plt.title('Sin Curve with Noise')
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig('examples/utils/plots/polynomialTransform.png', dpi=300)
