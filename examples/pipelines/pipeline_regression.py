import os
import sys
import warnings

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Pipeline and necessary components
from sega_learn.linear_models import Ridge
from sega_learn.pipelines import Pipeline
from sega_learn.utils import (
    Metrics,
    PolynomialTransform,
    Scaler,
    make_regression,
    train_test_split,
)

# Suppress warnings for cleaner output during testing/example run
warnings.filterwarnings("ignore", category=UserWarning)


print("\n--- Regression Pipeline Example ---")

# 1. Generate Data
X, y = make_regression(n_samples=200, n_features=3, noise=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Define Pipeline Steps
reg_steps = [
    ("poly", PolynomialTransform(degree=2)),
    ("scaler", Scaler(method="standard")),
    ("ridge", Ridge(alpha=0.5, max_iter=1000)),
]

# 3. Create and Fit Pipeline
reg_pipe = Pipeline(steps=reg_steps)
print("Fitting regression pipeline...")
reg_pipe.fit(X_train, y_train, **{"ridge__numba": True})
print("Fit complete.")

# 4. Predict and Evaluate
y_pred = reg_pipe.predict(X_test)
r2 = Metrics.r_squared(y_test, y_pred)
mse = Metrics.mean_squared_error(y_test, y_pred)

print("\nRegression Results (Default Pipeline):")
print(f"  R-squared: {r2:.4f}")
print(f"  Mean Squared Error: {mse:.4f}")
