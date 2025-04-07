import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.auto import AutoClassifier
from sega_learn.utils import make_classification
from sega_learn.utils.metrics import Metrics

accuracy = Metrics.accuracy
precision = Metrics.precision
recall = Metrics.recall
f1 = Metrics.f1_score


def run_example(verbose=False):
    """Runs the example."""
    X, y = make_classification(n_samples=1_000, n_features=5, random_state=42)

    # Create the AutoClassifier
    # reg = AutoClassifier()
    reg = AutoClassifier(all_kernels=True)  # Include all kernels in the model list

    # Fit all models
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


if __name__ == "__main__":
    run_example(verbose=True)
