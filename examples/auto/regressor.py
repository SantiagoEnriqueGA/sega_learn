import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.auto import AutoRegressor
from sega_learn.utils import make_regression
from sega_learn.utils.metrics import Metrics

r_squared = Metrics.r_squared
root_mean_squared_error = Metrics.root_mean_squared_error


def run_example(verbose=False):
    """Runs the example."""
    X, y = make_regression(n_samples=1_000, n_features=5, noise=0.5, random_state=42)

    # Create and fit the AutoRegressor
    reg = AutoRegressor()
    reg.fit(X, y, verbose=verbose)
    # Or can use custom metrics {str: callable}
    # reg.fit(X, y, custom_metrics={"r_squared": r_squared, "rmse": root_mean_squared_error})

    # Print the summary of all models
    reg.summary()

    # Predict using all models or a specific model
    predictions = reg.predict(X[:3])
    ols_pred = reg.predict(X[:3], model="OrdinaryLeastSquares")

    print("\nAll Predictions:")
    for model, pred in predictions.items():
        print(f"\t{model}: {pred}")
    print(f"Ordinary Least Squares Predictions: {ols_pred}")

    # Evaluate all or a specific model, can also use custom metrics {str: callable}
    results = reg.evaluate(y)
    results_ols = reg.evaluate(y, model="OrdinaryLeastSquares")

    print("\nAll Evaluation Results:")
    for model, result in results.items():
        print(f"\t{model}: {result}")
    print(f"Ordinary Least Squares Results: {results_ols}")


if __name__ == "__main__":
    run_example(verbose=True)
