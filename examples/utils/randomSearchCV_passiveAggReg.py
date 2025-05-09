import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sega_learn.utils.modelSelection as ms
from sega_learn.linear_models import PassiveAggressiveRegressor
from sega_learn.utils import Metrics, make_regression

r2_score = Metrics.r_squared

X, y = make_regression(n_samples=100, n_features=5, noise=25, random_state=42)


grid = [
    {"max_iter": [100, 500, 1000]},
    {"tol": [1e-3, 1e-5, 1e-7]},
    {"C": [1e-5, 1e-3, 1e-1]},
]

# grid_search = ms.RandomSearchCV(PassiveAggressiveRegressor, grid, iter=5, cv=3, metric='r2', direction='maximize')
grid_search = ms.RandomSearchCV(
    PassiveAggressiveRegressor, grid, iter=5, cv=3, metric="mse", direction="minimize"
)
reg = grid_search.fit(X, y, verbose=True)

print(f"Best Score: {grid_search.best_score_}")
print(f"Best Params: {grid_search.best_params_}")

print("\nResults from Best Model: ")
print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")
