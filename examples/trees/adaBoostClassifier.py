import os
import sys

import numpy as np

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.trees import AdaBoostClassifier
from sega_learn.utils import make_classification, train_test_split

# --- Data Generation ---
print("--- Generating Classification Data ---")
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,  # Number of informative features
    n_redundant=2,  # Number of redundant features (linear combinations)
    n_classes=2,  # Binary classification
    class_sep=0.5,  # How separated the classes are
    flip_y=0.05,  # Add some noise to labels
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
print(f"Class distribution in training data: {np.bincount(y_train)}")

# --- Model Initialization ---
# Initialize with default parameters (Decision Stump base estimator)
print("\n--- Initializing AdaBoost Classifier (Default: Decision Stumps) ---")
ada_classifier_default = AdaBoostClassifier(
    n_estimators=50,  # Number of weak learners (trees)
    learning_rate=1.0,  # Contribution of each learner
    random_state=42,
)

# Initialize with custom base estimator
from sega_learn.trees import RandomForestClassifier  # noqa: E402, I001

ada_classifier_custom = AdaBoostClassifier(
    base_estimator=RandomForestClassifier(n_jobs=1, n_estimators=10),
    n_estimators=10,
    learning_rate=0.5,
    random_state=42,
)

# --- Model Training ---
print("\n--- Training Default AdaBoost Classifier ---")
ada_classifier_default.fit(X_train, y_train)
print("Training complete.")

print("\n--- Training AdaBoost Classifier with Custom Base Estimator ---")
ada_classifier_custom.fit(X_train, y_train)
print("Training complete.")

# --- Prediction ---
print("\n--- Making Predictions ---")
y_pred_default = ada_classifier_default.predict(X_test)
y_pred_deeper = ada_classifier_custom.predict(X_test)

print("Predictions made for first 5 test samples (Default Model):", y_pred_default[:5])
print("Predictions made for first 5 test samples (Custom Model):", y_pred_deeper[:5])

# --- Evaluation ---
print("\n--- Evaluating Default AdaBoost Classifier ---")
stats_default = ada_classifier_default.get_stats(
    y_test, y_pred=y_pred_default, verbose=True
)

print("\n--- Evaluating AdaBoost Classifier with Custom Base Estimator ---")
stats_custom = ada_classifier_custom.get_stats(
    y_test, y_pred=y_pred_deeper, verbose=True
)

# Optional: Direct metric calculation
# accuracy = Metrics.accuracy(y_test, y_pred_default)
# precision = Metrics.precision(y_test, y_pred_default)
# recall = Metrics.recall(y_test, y_pred_default)
# f1 = Metrics.f1_score(y_test, y_pred_default)
# print(f"\nDirect Metrics (Default Model): Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
