import os
import sys

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.trees import AdaBoostRegressor
from sega_learn.utils import make_regression, train_test_split

# --- Data Generation ---
print("--- Generating Regression Data ---")
X, y = make_regression(n_samples=1000, n_features=3, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")

# --- Model Initialization ---
# Initialize with default parameters (RegressorTree(max_depth=3) base estimator, loss='linear')
print("\n--- Initializing AdaBoost Regressor ---")
ada_regressor = AdaBoostRegressor(
    n_estimators=100,  # Number of weak learners (trees)
    learning_rate=0.1,  # Lower learning rate often beneficial for regression
    loss="linear",  # Options: 'linear', 'square', 'exponential'
    random_state=42,
)

# Example with different loss function
# ada_regressor_square = AdaBoostRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     loss='square',
#     random_state=42
# )

# --- Model Training ---
print("\n--- Training AdaBoost Regressor ---")
ada_regressor.fit(X_train, y_train)
print("Training complete.")

# print("\n--- Training AdaBoost Regressor (Square Loss) ---")
# ada_regressor_square.fit(X_train, y_train)
# print("Training complete.")

# --- Prediction ---
print("\n--- Making Predictions ---")
y_pred = ada_regressor.predict(X_test)
# y_pred_square = ada_regressor_square.predict(X_test)

print("True values for first 5 test samples:", y_test[:5].round(2))
print("Predictions for first 5 test samples (Linear Loss):", y_pred[:5].round(2))
# print("Predictions for first 5 test samples (Square Loss):", y_pred_square[:5].round(2))

# --- Evaluation ---
print("\n--- Evaluating AdaBoost Regressor (Linear Loss) ---")
stats = ada_regressor.get_stats(y_test, y_pred=y_pred, verbose=True)

# print("\n--- Evaluating AdaBoost Regressor (Square Loss) ---")
# stats_square = ada_regressor_square.get_stats(y_test, y_pred=y_pred_square, verbose=True)

# Optional: Direct metric calculation
# r2 = Metrics.r_squared(y_test, y_pred)
# mse = Metrics.mean_squared_error(y_test, y_pred)
# mae = Metrics.mean_absolute_error(y_test, y_pred)
# print(f"\nDirect Metrics (Linear Loss): R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
