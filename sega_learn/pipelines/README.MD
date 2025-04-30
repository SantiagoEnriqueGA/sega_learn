# Sega Learn - Pipelines Module (`sega_learn.pipelines`)

The `pipelines` module provides tools for chaining multiple machine learning steps together into a single workflow. This includes preprocessing, feature transformation, dimensionality reduction, and final model estimation. Pipelines simplify building complex models, enhance reproducibility, and prevent common pitfalls like data leakage during cross-validation.

## Purpose

Using pipelines allows you to encapsulate your entire machine learning process, from raw data transformation to final prediction, within a single object. This is particularly useful for:

*   **Consistency:** Ensuring the same steps are applied to both training and testing data.
*   **Reproducibility:** Making it easier to replicate and share complex workflows.
*   **Preventing Data Leakage:** Ensuring that transformations (like scaling or PCA) are fit *only* on the training data, even within cross-validation loops.
*   **Simplifying Hyperparameter Tuning:** Enabling tools like `GridSearchCV` or `RandomSearchCV` to tune parameters across multiple steps simultaneously.
*   **Code Organization:** Creating cleaner, more modular, and maintainable machine learning code.

## Core Components

*   **`Pipeline`**:
    *   The standard pipeline class for general machine learning tasks (classification, regression).
    *   Sequentially applies a list of transformers and a final estimator.
    *   Each step is defined as a `(name, transform_or_estimator)` tuple.
    *   Provides `fit`, `predict`, and `predict_proba` (if the final estimator supports it) methods.
    *   Allows access to individual steps via the `named_steps` attribute.
    *   Integrates seamlessly with `sega_learn.utils.GridSearchCV` and `RandomSearchCV` for hyperparameter tuning across steps using the `step_name__parameter_name` syntax.

*   **`ForecastingPipeline`**:
    *   A specialized pipeline designed specifically for time series forecasting tasks.
    *   Separates steps into `preprocessors`, `model` (forecasting model), and `evaluators`.
    *   Handles the sequential nature of time series data appropriately during fitting and prediction.
    *   Provides `fit`, `predict` (often aliased or used similarly to `forecast`), `evaluate`, and `summary` methods.

## Key Features & Benefits

*   **Workflow Simplification**: Chain multiple data processing and modeling steps into a single estimator.
*   **Data Leakage Prevention**: Correctly handles fitting transformers only on training data during cross-validation and tuning.
*   **Reproducibility**: Encapsulates the entire sequence of operations.
*   **Hyperparameter Tuning**: Easily tune parameters of any step within the pipeline using grid search or random search.
*   **Code Organization**: Promotes cleaner and more structured ML code.

## Usage Examples

### 1. Basic Classification Pipeline

This example demonstrates setting up a pipeline for classification with scaling, PCA, and logistic regression.

```python
import os
import sys
# Adjust path if necessary
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import LogisticRegression
from sega_learn.pipelines import Pipeline
from sega_learn.utils import (
    Metrics,
    Scaler,
    make_classification,
    train_test_split,
)
from sega_learn.utils.decomposition import PCA

# 1. Generate Data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define Pipeline Steps
cls_steps = [
    ("scaler", Scaler(method="standard")),  # Scale features
    ("pca", PCA(n_components=3)),           # Reduce dimensionality
    ("logistic", LogisticRegression()),     # Final estimator
]

# 3. Create and Fit Pipeline
cls_pipe = Pipeline(steps=cls_steps)
cls_pipe.fit(X_train, y_train)

# 4. Predict and Evaluate
y_pred = cls_pipe.predict(X_test)
accuracy = Metrics.accuracy(y_test, y_pred)
print(f"Pipeline Accuracy: {accuracy:.4f}")

# Access fitted step attributes (optional)
print(f"PCA components shape: {cls_pipe.named_steps['pca'].components_.shape}")
```
*(Based on `examples/pipelines/pipeline_classification.py`)*

### 2. Basic Regression Pipeline

This example sets up a pipeline for regression involving polynomial feature transformation, scaling, and Ridge regression.

```python
# Imports similar to classification example...
from sega_learn.linear_models import Ridge
from sega_learn.pipelines import Pipeline
from sega_learn.utils import PolynomialTransform, Scaler, Metrics, make_regression, train_test_split

# 1. Generate Data
X, y = make_regression(n_samples=200, n_features=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define Pipeline Steps
reg_steps = [
    ("poly", PolynomialTransform(degree=2)), # Create polynomial features
    ("scaler", Scaler(method="standard")),   # Scale features
    ("ridge", Ridge(alpha=0.5)),             # Final estimator
]

# 3. Create and Fit Pipeline
reg_pipe = Pipeline(steps=reg_steps)
reg_pipe.fit(X_train, y_train)

# 4. Predict and Evaluate
y_pred = reg_pipe.predict(X_test)
r2 = Metrics.r_squared(y_test, y_pred)
print(f"Pipeline R-squared: {r2:.4f}")
```
*(Based on `examples/pipelines/pipeline_regression.py`)*

### 3. Pipeline with Hyperparameter Tuning (`GridSearchCV`)

Pipelines can be used directly with `GridSearchCV` to tune parameters from different steps.

```python
# Imports similar to above, plus GridSearchCV
from sega_learn.utils import GridSearchCV
# ... other necessary imports like AdaBoostClassifier, PCA etc.

# Assume X_cls_train, y_cls_train are defined

# Define pipeline to tune
tune_cls_pipe = Pipeline([
    ("scaler", Scaler()),
    ("pca", PCA()), # n_components will be tuned
    ("ada", AdaBoostClassifier(random_state=43)) # n_estimators, learning_rate tuned
])

# Define parameter grid using 'step_name__parameter_name'
param_grid_cls = [{
    "pca__n_components": [2, 3, 4],
    "ada__n_estimators": [10, 20],
    "ada__learning_rate": [0.01, 0.1],
}]

# Set up GridSearchCV
grid_search_cls = GridSearchCV(
    model=tune_cls_pipe,
    param_grid=param_grid_cls,
    cv=3,
    metric="accuracy",
    direction="maximize",
)

# Fit the search
grid_search_cls.fit(X_cls_train, y_cls_train)

print(f"Best Score: {grid_search_cls.best_score_:.4f}")
print(f"Best Params: {grid_search_cls.best_params_}")

# Get the best pipeline instance
best_pipeline = grid_search_cls.best_model
# Use best_pipeline for predictions...
```
*(Based on `examples/pipelines/pipeline_classification_tuning.py` and `pipeline_regression_tuning.py`)*

### 4. Forecasting Pipeline

This example shows the specialized `ForecastingPipeline` for time series tasks.

```python
# Imports similar to above, plus ForecastingPipeline and time series components
from sega_learn.pipelines import ForecastingPipeline
from sega_learn.time_series import SARIMA, WeightedMovingAverage
from sega_learn.utils import Metrics, make_time_series

# 1. Generate Time Series Data
time_series = make_time_series(n_timestamps=300, random_state=1).flatten()
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# 2. Define Pipeline Components
pipeline = ForecastingPipeline(
    preprocessors=[
        WeightedMovingAverage(window=5),
        # Add more preprocessors if needed
    ],
    model=SARIMA(order=(2, 1, 2), seasonal_order=(1, 1, 1, 100)),
    evaluators=[
        Metrics.mean_squared_error,
        Metrics.mean_absolute_error,
    ]
)

# 3. Fit the pipeline
pipeline.fit(train_series)

# 4. Predict (Forecast)
predictions = pipeline.predict(test_series, steps=len(test_series))

# 5. Evaluate
results = pipeline.evaluate(predictions, test_series)
print("\nPipeline Evaluation Results:")
for metric, value in results.items():
    print(f"\t{metric:30}: {value:.4f}")

# 6. Show Summary
pipeline.summary()
```
*(Based on `examples/pipelines/pipeline_forecasting.py`)*

## Examples Directory

See the `examples/pipelines/` directory for more detailed, runnable examples:
*   [`pipeline_classification.py`](../../examples/pipelines/pipeline_classification.py)
*   [`pipeline_regression.py`](../../examples/pipelines/pipeline_regression.py)
*   [`pipeline_classification_tuning.py`](../../examples/pipelines/pipeline_classification_tuning.py)
*   [`pipeline_regression_tuning.py`](../../examples/pipelines/pipeline_regression_tuning.py)
*   [`pipeline_forecasting.py`](../../examples/pipelines/pipeline_forecasting.py)
