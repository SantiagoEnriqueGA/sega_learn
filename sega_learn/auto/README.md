
# Sega Learn - Auto Module (`sega_learn.auto`)

The `auto` module provides high-level wrappers for automated machine learning tasks within the Sega Learn library. It aims to simplify the process of selecting, training, and evaluating models for standard classification and regression problems by trying a predefined set of algorithms and configurations.

## Purpose

The primary goal of this module is to offer a quick and easy way to get a baseline model working on a dataset with minimal code. It automates parts of the model selection process, leveraging various estimators available in other Sega Learn modules (`linear_models`, `trees`, `svm`, `nearest_neighbors`, etc.). It is best used for quick prototyping and experimentation, but not for production-level applications. It provide insights into the baseline performance of different models, but does not optimize the hyperparameters of the models.

## Core Components

*   **`AutoClassifier`**:
    *   Performs automated classification.
    *   Trains and evaluates a selection of classification algorithms from Sega Learn (e.g., Logistic Regression, SVM, Random Forest Classifier, K-Nearest Neighbors).
    *   Selects the best-performing model based on a specified metric (e.g., accuracy, F1-score).
    *   Provides a standard `fit`, `predict`, `predict_proba`, and `score` interface.

*   **`AutoRegressor`**:
    *   Performs automated regression.
    *   Trains and evaluates a selection of regression algorithms from Sega Learn (e.g., Linear Regression, Ridge, Lasso, Random Forest Regressor, K-Neighbors Regressor).
    *   Selects the best-performing model based on a specified metric (e.g., R², Mean Squared Error).
    *   Provides a standard `fit`, `predict`, and `score` interface.

## Key Features

*   **Simplicity**: Reduces the boilerplate code needed to try multiple models.
*   **Automation**: Handles the iteration through different model types.
*   **Integration**: Built on top of existing Sega Learn estimators.
*   **Baseline**: Useful for quickly establishing a baseline performance on a new dataset.

## Usage Example

```python
from sega_learn.auto import AutoClassifier
from sega_learn.utils import make_classification

# Optional: Define custom metrics
from sega_learn.utils.metrics import Metrics
accuracy = Metrics.accuracy
precision = Metrics.precision
recall = Metrics.recall
f1 = Metrics.f1_score

# Get regression data
X, y = make_classification(n_samples=1_000, n_features=5, random_state=42)

# Create and fit the AutoClassifier
reg = AutoClassifier()
reg.fit(X, y, verbose=verbose)

# Or can use custom metrics {str: callable}
# reg.fit(X, y, custom_metrics={"r_squared": r_squared, "rmse": root_mean_squared_error})

# Print the summary of all models
reg.summary()

# Predict using all models or a specific model
predictions = reg.predict(X[:3])
ols_pred = reg.predict(X[:3], model="ClassifierTree")

print("\nAll Predictions:")
for model, pred in predictions.items():
    print(f"\t{model}: {pred}")
print(f"Classifier Tree Predictions: {ols_pred}")

# Evaluate all or a specific model, can also use custom metrics {str: callable}
results = reg.evaluate(y)
results_classifier_tree = reg.evaluate(y, model="ClassifierTree")

print("\nAll Evaluation Results:")
for model, result in results.items():
    print(f"\t{model}: {result}")
print(f"Classifier Tree Results: {results_classifier_tree}")
```

## Limitations & Future Work

*   The current selection of models is limited to the ones available in the `sega_learn` library.
*   Preprocessing steps (scaling, encoding) are typically expected to be done *before* using the auto-estimators, although future versions might incorporate basic automated preprocessing.
*   Advanced techniques like ensemble building (beyond Random Forests) or complex pipeline construction are not the primary focus of this basic module.

See the examples directory for practical usage: [`examples/auto/`](../../examples/auto/)
