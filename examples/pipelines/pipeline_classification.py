import os
import sys
import warnings

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Pipeline and necessary components
from sega_learn.linear_models import LogisticRegression
from sega_learn.pipelines import Pipeline
from sega_learn.utils import (
    Metrics,
    Scaler,
    make_classification,
    train_test_split,
)
from sega_learn.utils.decomposition import PCA

# Suppress warnings for cleaner output during testing/example run
warnings.filterwarnings("ignore", category=UserWarning)

print("\n--- Classification Pipeline Example ---")

# 1. Generate Data
X, y = make_classification(
    n_samples=200, n_features=10, n_informative=5, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Define Pipeline Steps
cls_steps = [
    ("scaler", Scaler(method="standard")),
    ("pca", PCA(n_components=3)),
    ("logistic", LogisticRegression(max_iter=100, learning_rate=0.01)),
]

# 3. Create and Fit Pipeline
cls_pipe = Pipeline(steps=cls_steps)
print("Fitting classification pipeline...")
cls_pipe.fit(X_train, y_train)
print("Fit complete.")

# 4. Predict and Evaluate
y_pred = cls_pipe.predict(X_test)
accuracy = Metrics.accuracy(y_test, y_pred)
precision = Metrics.precision(y_test, y_pred)
recall = Metrics.recall(y_test, y_pred)

print("\nClassification Results (Default Pipeline):")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")

# 5. Access individual steps (optional)
print(
    f"\nPCA components shape from fitted pipeline: {cls_pipe.named_steps['pca'].components_.shape}"
)
