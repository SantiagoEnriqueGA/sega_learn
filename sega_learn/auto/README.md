# Sega Learn - Auto Module (`sega_learn.auto`)

The `auto` module provides high-level wrappers for automated machine learning tasks within the Sega Learn library. It simplifies the process of selecting, training, tuning, and evaluating models for classification and regression problems by leveraging a predefined set of algorithms and optional hyperparameter optimization.

## Purpose

The `auto` module is designed to streamline model prototyping and experimentation. It automates model selection and now includes optional hyperparameter tuning to improve performance beyond default settings. While ideal for establishing baselines and exploring datasets, it is not intended for production-level applications without further customization.

## Core Components

*   **`AutoClassifier`**:
    *   Performs automated classification.
    *   Trains and evaluates a diverse set of classifiers from Sega Learn (e.g., Logistic Regression, SVM with multiple kernels, Random Forest Classifier, K-Nearest Neighbors, One-Class SVM for binary tasks, and Neural Networks).
    *   Selects the best-performing model based on a specified metric (e.g., accuracy, F1-score).
    *   Provides a standard `fit`, `predict`, `predict_proba`, and `evaluate` interface.

*   **`AutoRegressor`**:
    *   Performs automated regression.
    *   Trains and evaluates a variety of regressors from Sega Learn (e.g., Linear Regression, Ridge, Lasso, Random Forest Regressor, K-Neighbors Regressor, and Neural Networks).
    *   Selects the best-performing model based on a specified metric (e.g., R², Mean Squared Error).
    *   Provides a standard `fit`, `predict`, and `evaluate` interface.

## Key Features

*   **Simplicity**: Minimal code required to test multiple models.
*   **Automation**: Iterates through model types and optionally tunes hyperparameters.
*   **Hyperparameter Tuning**: Supports grid search or random search (`GridSearchCV`, `RandomSearchCV`) to optimize model performance.
    *   Configurable via `tune_hyperparameters=True`, `tuning_method` ("random" or "grid"), `tuning_iterations`, and `cv`.
    *   Predefined parameter grids for most models (excluding neural networks).
*   **Neural Network Support**: Automatically scales data and trains neural networks with default architectures (e.g., [128, 64, 32] for classification).
*   **Integration**: Built on Sega Learn estimators with enhanced model management.
*   **Comprehensive Evaluation**: Detailed `summary` output includes tuning results, metrics, and runtime.

## Usage Example

```python
from sega_learn.auto import AutoClassifier
from sega_learn.utils import make_classification

# Optional: Define custom metrics
from sega_learn.utils.metrics import Metrics
accuracy = Metrics.accuracy
f1 = Metrics.f1_score

# Generate classification data
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

# Create and fit the AutoClassifier with hyperparameter tuning
clf = AutoClassifier(tune_hyperparameters=True, tuning_method="random", tuning_iterations=10, cv=3, tuning_metric="f1")
clf.fit(X, y, verbose=True)

# Print a summary of all models, including tuning results
clf.summary()

# Predict using all models or a specific model
predictions = clf.predict(X[:3])
rf_pred = clf.predict(X[:3], model="RandomForestClassifier")

print("\nAll Predictions:")
for model, pred in predictions.items():
    print(f"\t{model}: {pred}")
print(f"Random Forest Predictions: {rf_pred}")

# Evaluate all models or a specific model with custom metrics
results = clf.evaluate(X, y, custom_metrics={"accuracy": accuracy, "f1": f1})
rf_results = clf.evaluate(X, y, model="RandomForestClassifier")

print("\nAll Evaluation Results:")
for model, result in results.items():
    print(f"\t{model}: {result}")
print(f"Random Forest Results: {rf_results}")
```

## Example Model Summaries
### Example 1: AutoClassifier with Default Parameters

```
╭───────────────────┬──────────────────────────┬────────────┬─────────────┬──────────┬────────────┬──────────────┬─────────╮
│ Model Class       │ Model                    │   Accuracy │   Precision │   Recall │   F1 Score │   Time Taken │ Tuned   │
├───────────────────┼──────────────────────────┼────────────┼─────────────┼──────────┼────────────┼──────────────┼─────────┤
│ SVM               │ OneClassSVM - Linear     │       0.27 │      1      │   1      │     1      │       0.019  │ False   │
│ Trees             │ RandomForestClassifier   │       1    │      1      │   1      │     1      │       4.2018 │ False   │
│ SVM               │ LinearSVC                │       0.52 │      0.8525 │   1      │     0.9204 │       0.093  │ False   │
│ Trees             │ ClassifierTree           │       0.9  │      0.9245 │   0.8909 │     0.9074 │       0.006  │ False   │
│ Nearest Neighbors │ KNeighborsClassifier     │       0.89 │      0.8793 │   0.9273 │     0.9027 │       0.001  │ False   │
│ SVM               │ GeneralizedSVC - Linear  │       0.85 │      0.8571 │   0.8727 │     0.8649 │       0.039  │ False   │
│ Neural Networks   │ BaseBackendNeuralNetwork │       0.55 │      0.55   │   1      │     0.7097 │       0.061  │ False   │
╰───────────────────┴──────────────────────────┴────────────┴─────────────┴──────────┴────────────┴──────────────┴─────────╯
```

### Example 2: AutoClassifier with Hyperparameter Tuning
```
╭───────────────────┬──────────────────────────┬────────────┬─────────────┬──────────┬────────────┬──────────────┬─────────┬───────────────────────┬────────────────────────────────────────────────────╮
│ Model Class       │ Model                    │   Accuracy │   Precision │   Recall │   F1 Score │   Time Taken │ Tuned   │ Best Score (Tuning)   │ Best Params (Tuning)                               │
├───────────────────┼──────────────────────────┼────────────┼─────────────┼──────────┼────────────┼──────────────┼─────────┼───────────────────────┼────────────────────────────────────────────────────┤
│ Trees             │ ClassifierTree           │       1    │      1      │   1      │     1      │       0.091  │ True    │ 0.8927                │ {'max_depth': 15}                                  │
│ SVM               │ OneClassSVM - Linear     │       0.21 │      1      │   1      │     1      │       0.4201 │ True    │ 1.0000                │ {'C': 100, 'tol': 0.001, 'max_iter': 3000, 'lea... │
│ Trees             │ RandomForestClassifier   │       1    │      1      │   1      │     1      │       1.3242 │ True    │ 0.8934                │ {'n_estimators': 50, 'max_depth': 10}               │
│ SVM               │ LinearSVC                │       0.46 │      0.8679 │   1      │     0.9293 │       1.0683 │ True    │ 0.9292                │ {'C': 0.1, 'tol': 0.0001, 'max_iter': 1000, 'le... │
│ SVM               │ GeneralizedSVC - Linear  │       0.84 │      0.8679 │   0.8364 │     0.8519 │       0.9749 │ True    │ 0.8683                │ {'C': 0.1, 'tol': 0.0001, 'max_iter': 3000, 'le... │
│ Nearest Neighbors │ KNeighborsClassifier     │       0.81 │      0.8103 │   0.8545 │     0.8319 │       0.005  │ True    │ 0.8766                │ {'n_neighbors': 9, 'distance_metric': 'manhattan'} │
│ Neural Networks   │ BaseBackendNeuralNetwork │       0.55 │      0.55   │   1      │     0.7097 │       0.0312 │ False   │ N/A                   │ N/A                                                │
╰───────────────────┴──────────────────────────┴────────────┴─────────────┴──────────┴────────────┴──────────────┴─────────┴───────────────────────┴────────────────────────────────────────────────────╯
```


## Advanced Configuration

*   **Tuning Options**:
    *   `tune_hyperparameters=True`: Enables tuning.
    *   `tuning_method="random"`: Random search (default) or `"grid"` for exhaustive search.
    *   `tuning_iterations=10`: Number of parameter sets to try (random search only).
    *   `cv=3`: Cross-validation folds for tuning.
*   **Neural Networks**:
    *   Automatically included with default architectures and training (50 epochs, batch size 32).
    *   Data is scaled using `Scaler(method="standard")` before training.
*   **One-Class SVM**:
    *   Added for binary classification tasks with multiple kernel options.

## Limitations & Future Work

*   **Model Scope**: Limited to Sega Learn’s built-in estimators; external models require manual integration.
*   **Preprocessing**: Scaling is included for neural networks, but other preprocessing (e.g., encoding) must be done beforehand. Future versions may automate this.
*   **Tuning Overhead**: Hyperparameter tuning increases computation time, especially with grid search or large datasets.
*   **Neural Network Flexibility**: Current implementation uses fixed training parameters; advanced customization requires separate use of the `neural_network` module.
*   **Ensemble Methods**: Limited to Random Forests; broader ensemble techniques may be added later.

See the examples directory for practical usage: [`examples/auto/`](../../examples/auto/)
