# Import Custom Classes
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sega_learn.trees.randomForestRegressor as rfr
import sega_learn.utils.modelSelection as ms
from sega_learn.utils import make_regression

X, y = make_regression(n_samples=100, n_features=3, noise=5, random_state=42)

grid = [
    {"n_estimators": [5, 10, 15, 20]},
    {"max_depth": [5, 10, 15, 20]},
]
# grid_search = ms.RandomSearchCV(rfr.RandomForestRegressor, grid, iter=5, cv=3, metric='mse', direction='minimize')
grid_search = ms.RandomSearchCV(
    rfr.RandomForestRegressor, grid, iter=5, cv=3, metric="r2", direction="maximize"
)
model = grid_search.fit(X, y, verbose=True)

print(f"Best Score: {grid_search.best_score_}")
print(f"Best Params: {grid_search.best_params_}")
