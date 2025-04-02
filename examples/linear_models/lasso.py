import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import Lasso
from sega_learn.utils import Metrics, make_regression

r2_score = Metrics.r_squared

X, y = make_regression(n_samples=1000, n_features=5, noise=0.5, random_state=42)

reg = Lasso(alpha=1.0, fit_intercept=True)
reg.fit(X, y)

print("Lasso Regression Results: (without numba)")
print(f"R^2 Score: {r2_score(y, reg.predict(X)):.2f}")
print(f"Regression Coefficients: {[round(coef, 2) for coef in reg.coef_]}")
print(f"Regression Intercept: {reg.intercept_:.2f}")
print(f"Regression Formula: {reg.get_formula()}")

reg_nb = Lasso(alpha=1.0, fit_intercept=True)
reg_nb.fit(X, y, numba=True)

r2 = round(r2_score(y, reg_nb.predict(X)), 2)
coef = [round(c, 2) for c in reg_nb.coef_]
intercept = round(reg_nb.intercept_, 2)
formula = reg_nb.get_formula()

print("\nLasso Regression Results: (with numba)")
print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")
