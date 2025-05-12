import os
import sys
import warnings

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Pipeline and necessary components
from sega_learn.linear_models import Ridge
from sega_learn.pipelines import Pipeline
from sega_learn.utils import (
    GridSearchCV,
    Metrics,
    PolynomialTransform,
    Scaler,
    make_regression,
    train_test_split,
)

# Suppress warnings for cleaner output during testing/example run
warnings.filterwarnings("ignore", category=UserWarning)

print("\n--- Pipeline with GridSearchCV Example ---")

# --- Regression Tuning ---
print("\nTuning Regression Pipeline...")
X_reg, y_reg = make_regression(n_samples=250, n_features=3, noise=0.15, random_state=43)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=43
)

tune_reg_pipe = Pipeline(
    [
        ("poly", PolynomialTransform(degree=2)),  # Degree will be tuned
        ("scaler", Scaler()),
        ("ridge", Ridge(alpha=1.0)),  # Alpha will be tuned
    ]
)

param_grid_reg = [{"poly__degree": [1, 2], "ridge__alpha": [0.1, 1.0]}]

grid_search_reg = GridSearchCV(
    model=tune_reg_pipe,
    param_grid=param_grid_reg,
    cv=2,
    metric="r2",  # Optimize for R-squared
    direction="maximize",
)

print("Running GridSearchCV for regression...")
grid_search_reg.fit(X_reg_train, y_reg_train, verbose=False)
print("GridSearchCV fit complete.")

print(f"\nBest Regression Score (R2): {grid_search_reg.best_score_:.4f}")
print(f"Best Regression Parameters: {grid_search_reg.best_params_}")

best_reg_pipe = grid_search_reg.best_model
y_reg_pred_best = best_reg_pipe.predict(X_reg_test)
best_r2 = Metrics.r_squared(y_reg_test, y_reg_pred_best)
print(f"R-squared of best regression pipeline on test set: {best_r2:.4f}")
