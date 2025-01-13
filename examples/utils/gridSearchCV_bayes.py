
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.linear_models import Bayesian
import sega_learn.utils.model_selection as ms
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)


grid = [
    {'max_iter': [100, 500, 1000]},
    {'tol': [1e-3, 1e-4, 1e-5]},
    {'alpha_1': [1e-6, 1e-7, 1e-8]},
    {'alpha_2': [1e-6, 1e-7, 1e-8]},
    {'lambda_1': [1e-6, 1e-7, 1e-8]},
    {'lambda_2': [1e-6, 1e-7, 1e-8]},
    {'fit_intercept' : [True, False]},
]

grid_search = ms.GridSearchCV(Bayesian, grid, cv=3, metric='mse', direction='minimize')
# grid_search = ms.GridSearchCV(Bayesian, grid, cv=3, metric='r2', direction='maximize')
reg = grid_search.fit(X, y, verbose=False)

print(f"Best Score: {grid_search.best_score_}")
print(f"Best Params: {grid_search.best_params_}")

print("\nResults from Best Model: ")
print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")

