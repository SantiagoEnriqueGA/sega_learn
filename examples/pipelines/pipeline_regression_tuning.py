import os
import sys
import warnings

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Pipeline and necessary components
from sega_learn.pipelines import Pipeline
from sega_learn.trees import AdaBoostClassifier
from sega_learn.utils import (
    GridSearchCV,
    Metrics,
    Scaler,
    make_classification,
    train_test_split,
)
from sega_learn.utils.decomposition import PCA

# Suppress warnings for cleaner output during testing/example run
warnings.filterwarnings("ignore", category=UserWarning)

print("\n--- Pipeline with GridSearchCV Example ---")

# --- Classification Tuning ---
print("\nTuning Classification Pipeline...")
X_cls, y_cls = make_classification(
    n_samples=150, n_features=8, n_informative=4, n_classes=2, random_state=43
)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=43
)

# Pipeline to tune
tune_cls_pipe = Pipeline(
    [
        ("scaler", Scaler()),
        ("pca", PCA(n_components=5)),  # n_components will be tuned
        (
            "ada",
            AdaBoostClassifier(n_estimators=10, random_state=43, learning_rate=0.1),
        ),
    ]
)

# Parameter grid targeting steps in the pipeline
# Use `step_name__parameter_name` syntax
param_grid_cls = [
    {
        "pca__n_components": [2, 3, 4],  # Tune PCA components
        "ada__n_estimators": [10, 20],  # Tune AdaBoost estimators
        "ada__learning_rate": [0.01, 0.1],  # Tune AdaBoost learning rate
    }
]

# Note: GridSearchCV uses its own internal cross-validation scorer.
# Ensure your chosen 'metric' in GridSearchCV matches Metrics keys if needed elsewhere,
# or use standard sklearn scoring strings if using sklearn's GridSearchCV.
# Using sega_learn's GridSearchCV here.
grid_search_cls = GridSearchCV(
    model=tune_cls_pipe,  # Pass the pipeline as the estimator
    param_grid=param_grid_cls,
    cv=2,  # Use fewer folds for faster example
    metric="accuracy",  # Metric to optimize
    direction="maximize",
)

print("Running GridSearchCV for classification...")
grid_search_cls.fit(X_cls_train, y_cls_train, verbose=False)  # Fit the search
print("GridSearchCV fit complete.")

print(f"\nBest Classification Score (Accuracy): {grid_search_cls.best_score_:.4f}")
print(f"Best Classification Parameters: {grid_search_cls.best_params_}")

# Evaluate the best model found by GridSearchCV
best_cls_pipe = grid_search_cls.best_model
y_cls_pred_best = best_cls_pipe.predict(X_cls_test)
best_accuracy = Metrics.accuracy(y_cls_test, y_cls_pred_best)
print(f"Accuracy of best classification pipeline on test set: {best_accuracy:.4f}")
