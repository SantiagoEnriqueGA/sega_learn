import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.linear_models import Ridge
from sega_learn.utils import make_regression
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

X, y = make_regression(n_samples=1000, n_features=5, noise=.5, random_state=42)

reg = Ridge(alpha=1.0, fit_intercept=True)
reg.fit(X, y, numba=False)

r2 = round(r2_score(y, reg.predict(X)), 2)
coef = [round(c, 2) for c in reg.coef_]
intercept = round(reg.intercept_, 2)
formula = reg.get_formula()

print("Ridge Regression Results (without numba):")
print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")

reg_nb = Ridge(alpha=1.0, fit_intercept=True)
reg_nb.fit(X, y, numba=True)

r2 = round(r2_score(y, reg_nb.predict(X)), 2)
coef = [round(c, 2) for c in reg_nb.coef_]
intercept = round(reg_nb.intercept_, 2)
formula = reg_nb.get_formula()

print("\nRidge Regression Results (with numba):")
print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")
