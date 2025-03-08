import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.linear_models import PassiveAggressiveRegressor
from sega_learn.utils import make_regression
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)

reg = PassiveAggressiveRegressor(C=.001, max_iter=1000, tol=1e-4)
reg.fit(X, y, save_steps=True)

r2 = round(r2_score(y, reg.predict(X)), 2)
coef = [round(c, 2) for c in reg.coef_]
intercept = round(reg.intercept_, 2)
formula = reg.get_formula()

print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")

