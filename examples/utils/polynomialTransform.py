import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import numpy as np
from sega_learn.linear_models import OrdinaryLeastSquares
from sega_learn.utils import Metrics, PolynomialTransform

r2_score = Metrics.r_squared

np.random.seed(42)  # Random seed for reproducibility

pnts = 300
X = np.linspace(0, 5, pnts)
y = 10 * np.sin(X) + np.random.normal(0, 1, pnts)

# Fit OLS Regression
reg = OrdinaryLeastSquares(fit_intercept=False)
reg.fit(X.reshape(-1, 1), y)

# Fit OLS Regression with Polynomial Features (degree=2)
poly2 = PolynomialTransform(degree=2)
X_poly2 = poly2.fit_transform(X.reshape(-1, 1))
reg_poly2 = OrdinaryLeastSquares(fit_intercept=False)
reg_poly2.fit(X_poly2, y)

# Fit OLS Regression with Polynomial Features (degree=3)
poly3 = PolynomialTransform(degree=3)
X_poly3 = poly3.fit_transform(X.reshape(-1, 1))
reg_poly3 = OrdinaryLeastSquares(fit_intercept=False)
reg_poly3.fit(X_poly3, y)

print("\nExample Usage Polynomial Transform of Features")
print("Base Model: y = 10 * sin(x) + noise")
print(f"OLS R^2 Score: {r2_score(y, reg.predict(X.reshape(-1, 1))):.2}")
print(
    f"OLS Polynomial Degree 2 R^2 Score: {r2_score(y, reg_poly2.predict(X_poly2)):.2}"
)
print(
    f"OLS Polynomial Degree 3 R^2 Score: {r2_score(y, reg_poly3.predict(X_poly3)):.2}"
)


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data Points", alpha=0.5)
plt.plot(
    X, reg.predict(X.reshape(-1, 1)), color="red", label="OLS Regression", linewidth=2
)
plt.plot(
    X,
    reg_poly2.predict(X_poly2),
    color="green",
    label="OLS Regression Polynomial Degree 2",
    linewidth=2,
)
plt.plot(
    X,
    reg_poly3.predict(X_poly3),
    color="orange",
    label="OLS Regression Polynomial Degree 3",
    linewidth=2,
)
plt.xlabel("Feature 0")
plt.ylabel("Target")
plt.title("Sin Curve with Noise")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("examples/utils/plots/polynomialTransform.png", dpi=300)
