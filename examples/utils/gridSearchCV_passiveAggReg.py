
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.linear_models import PassiveAggressiveRegressor
import sega_learn.utils.modelSelection as ms
from sega_learn.utils import make_regression
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

X, y = make_regression(n_samples=100, n_features=5, noise=25, random_state=42)


grid = [
    {'max_iter': [100, 1000]},
    {'tol': [1e-3, 1e-5]},
    {'C': [1e-6, 1e-8]},
]

# grid_search = ms.GridSearchCV(PassiveAggressiveRegressor, grid, cv=3, metric='r2', direction='maximize')
grid_search = ms.GridSearchCV(PassiveAggressiveRegressor, grid, cv=3, metric='mse', direction='minimize')
reg = grid_search.fit(X, y, verbose=True)

print(f"Best Score: {grid_search.best_score_}")
print(f"Best Params: {grid_search.best_params_}")

print("\nResults from Best Model: ")
print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")

