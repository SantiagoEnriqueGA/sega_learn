import os
import sys
import warnings

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import Pipeline and necessary components
from sega_learn.linear_models import LogisticRegression, Ridge
from sega_learn.pipelines import Pipeline
from sega_learn.trees import RandomForestClassifier
from sega_learn.utils import (
    GridSearchCV,
    Metrics,
    PolynomialTransform,
    Scaler,
    make_classification,
    make_regression,
    train_test_split,
)
from sega_learn.utils.decomposition import PCA

# Suppress warnings for cleaner output during testing/example run
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_classification_example():
    """Demonstrates the Pipeline for a classification task."""
    print("\n--- Classification Pipeline Example ---")

    # 1. Generate Data
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 2. Define Pipeline Steps
    # Example: Scale -> PCA -> Logistic Regression
    cls_steps = [
        ("scaler", Scaler(method="standard")),
        ("pca", PCA(n_components=3)),
        (
            "logistic",
            LogisticRegression(max_iter=100, learning_rate=0.01),
        ),  # Use LogisticRegression from sega_learn
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


def run_regression_example():
    """Demonstrates the Pipeline for a regression task."""
    print("\n--- Regression Pipeline Example ---")

    # 1. Generate Data
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 2. Define Pipeline Steps
    # Example: Polynomial Features -> Scale -> Ridge Regression
    reg_steps = [
        ("poly", PolynomialTransform(degree=2)),
        ("scaler", Scaler(method="standard")),
        ("ridge", Ridge(alpha=1.0, max_iter=500)),  # Use Ridge from sega_learn
    ]

    # 3. Create and Fit Pipeline
    reg_pipe = Pipeline(steps=reg_steps)
    print("Fitting regression pipeline...")
    reg_pipe.fit(X_train, y_train)
    print("Fit complete.")

    # 4. Predict and Evaluate
    y_pred = reg_pipe.predict(X_test)
    r2 = Metrics.r_squared(y_test, y_pred)
    mse = Metrics.mean_squared_error(y_test, y_pred)

    print("\nRegression Results (Default Pipeline):")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Mean Squared Error: {mse:.4f}")


def run_tuning_example():
    """Demonstrates hyperparameter tuning with GridSearchCV and Pipeline."""
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
                "rf",
                RandomForestClassifier(n_estimators=10, random_state=43, n_jobs=1),
            ),  # n_estimators will be tuned
        ]
    )

    # Parameter grid targeting steps in the pipeline
    # Use `step_name__parameter_name` syntax
    param_grid_cls = {
        "pca__n_components": [2, 3, 4],  # Tune PCA components
        "rf__n_estimators": [10, 20],  # Tune RandomForest estimators
        "rf__max_depth": [3, 5],  # Tune RandomForest max_depth
    }

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

    # --- Regression Tuning ---
    print("\nTuning Regression Pipeline...")
    X_reg, y_reg = make_regression(
        n_samples=150, n_features=3, noise=15, random_state=43
    )
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

    param_grid_reg = {"poly__degree": [1, 2, 3], "ridge__alpha": [0.1, 1.0, 10.0]}

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


if __name__ == "__main__":
    run_classification_example()
    run_regression_example()
    run_tuning_example()
