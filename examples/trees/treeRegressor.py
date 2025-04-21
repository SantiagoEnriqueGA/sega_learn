import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.trees import treeRegressor
from sega_learn.utils import Metrics, make_regression

X, y = make_regression(n_samples=1000, n_features=5, noise=0.25, random_state=42)

# Train random forest model
tree = treeRegressor.RegressorTree(max_depth=5)
tree.fit(X, y)

preds = tree.predict(X)
print(f"R^2:  {Metrics.r_squared(y, preds):.4f}")
print(f"MAE:  {Metrics.mean_absolute_error(y, preds):.4f}")
print(f"RMSE: {Metrics.root_mean_squared_error(y, preds):.4f}")
print(f"MAPE: {Metrics.mean_absolute_percentage_error(y, preds):.4f}")
