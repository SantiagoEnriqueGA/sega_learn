import os
import sys
import time

from tests_performance._utils import suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.linear_models import *
from sklearn import linear_model
from tests_performance._utils import synthetic_data_regression


class TestTime:
    def __init__(self, X, y, X_test, num_samples, num_runs):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.num_samples = num_samples
        self.num_runs = num_runs

    def measure_performance(self, model_class_sklearn, model_class_sega):
        def model_performance(model_class):
            start_time = time.time()
            model = model_class()
            model.fit(self.X, self.y)
            y_pred = model.predict(self.X_test)
            end_time = time.time()
            return end_time - start_time

        sklearn_times = [
            model_performance(model_class_sklearn) for _ in range(self.num_runs)
        ]
        sega_times = [model_performance(model_class_sega) for _ in range(self.num_runs)]

        avg_sklearn_time = sum(sklearn_times) / self.num_runs
        avg_sega_time = sum(sega_times) / self.num_runs

        return avg_sklearn_time, avg_sega_time


def test_ols(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    timer = TestTime(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    avg_sklearn_time, avg_sega_time = timer.measure_performance(
        linear_model.LinearRegression, OrdinaryLeastSquares
    )

    print(
        f"\nOLS Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):"
    )
    print(f"\tsklearn Linear Regression: {avg_sklearn_time:7.4f} seconds")
    print(f"\tsega Linear Regression: {avg_sega_time:10.4f} seconds")

    if avg_sklearn_time == 0:
        avg_sklearn_time = 1e-10
    if avg_sega_time == 0:
        avg_sega_time = 1e-10
    if avg_sklearn_time == avg_sega_time:
        print("\tSpeed difference, is negligible")
    elif avg_sklearn_time > avg_sega_time:
        print(
            f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster"
        )
    elif avg_sega_time > avg_sklearn_time:
        print(
            f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster"
        )


def test_ridge(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    original_X = X
    original_y = y
    reduced_sample_size = NUM_SAMPLES // 100
    X = X[:reduced_sample_size]
    y = y[:reduced_sample_size]

    timer = TestTime(X, y, X_test, reduced_sample_size, NUM_RUNS)
    avg_sklearn_time, avg_sega_time = timer.measure_performance(
        linear_model.Ridge, Ridge
    )

    print(
        f"\nRidge Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):"
    )
    print(f"\tsklearn Ridge Regression: {avg_sklearn_time:7.4f} seconds")
    print(f"\tsega Ridge Regression: {avg_sega_time:10.4f} seconds")

    if avg_sklearn_time == 0:
        avg_sklearn_time = 1e-10
    if avg_sega_time == 0:
        avg_sega_time = 1e-10
    if avg_sklearn_time == avg_sega_time:
        print("\tSpeed difference, is negligible")
    elif avg_sklearn_time > avg_sega_time:
        print(
            f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster"
        )
    elif avg_sega_time > avg_sklearn_time:
        print(
            f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster"
        )

    X = original_X
    y = original_y


def test_lasso(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    original_X = X
    original_y = y
    reduced_sample_size = NUM_SAMPLES // 100
    X = X[:reduced_sample_size]
    y = y[:reduced_sample_size]

    timer = TestTime(X, y, X_test, reduced_sample_size, NUM_RUNS)
    avg_sklearn_time, avg_sega_time = timer.measure_performance(
        linear_model.Lasso, Lasso
    )

    print(
        f"\nLasso Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):"
    )
    print(f"\tsklearn Lasso Regression: {avg_sklearn_time:7.4f} seconds")
    print(f"\tsega Lasso Regression: {avg_sega_time:10.4f} seconds")

    if avg_sklearn_time == 0:
        avg_sklearn_time = 1e-10
    if avg_sega_time == 0:
        avg_sega_time = 1e-10
    if avg_sklearn_time == avg_sega_time:
        print("\tSpeed difference, is negligible")
    elif avg_sklearn_time > avg_sega_time:
        print(
            f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster"
        )
    elif avg_sega_time > avg_sklearn_time:
        print(
            f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster"
        )

    X = original_X
    y = original_y


def test_bayesian(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    timer = TestTime(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    avg_sklearn_time, avg_sega_time = timer.measure_performance(
        linear_model.BayesianRidge, Bayesian
    )

    print(
        f"\nBayesian Ridge Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):"
    )
    print(f"\tsklearn Bayesian Ridge Regression: {avg_sklearn_time:7.4f} seconds")
    print(f"\tsega Bayesian Ridge Regression: {avg_sega_time:10.4f} seconds")

    if avg_sklearn_time == 0:
        avg_sklearn_time = 1e-10
    if avg_sega_time == 0:
        avg_sega_time = 1e-10
    if avg_sklearn_time == avg_sega_time:
        print("\tSpeed difference, is negligible")
    elif avg_sklearn_time > avg_sega_time:
        print(
            f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster"
        )
    elif avg_sega_time > avg_sklearn_time:
        print(
            f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster"
        )


def test_ransac(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    timer = TestTime(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    avg_sklearn_time, avg_sega_time = timer.measure_performance(
        linear_model.RANSACRegressor, RANSAC
    )

    print(
        f"\nRANSAC Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):"
    )
    print(f"\tsklearn RANSAC Regression: {avg_sklearn_time:7.4f} seconds")
    print(f"\tsega RANSAC Regression: {avg_sega_time:10.4f} seconds")

    if avg_sklearn_time == 0:
        avg_sklearn_time = 1e-10
    if avg_sega_time == 0:
        avg_sega_time = 1e-10
    if avg_sklearn_time == avg_sega_time:
        print("\tSpeed difference, is negligible")
    elif avg_sklearn_time > avg_sega_time:
        print(
            f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster"
        )
    elif avg_sega_time > avg_sklearn_time:
        print(
            f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster"
        )


def test_passiveAggressiveRegressor(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    original_X = X
    original_y = y
    reduced_sample_size = NUM_SAMPLES // 100
    X = X[:reduced_sample_size]
    y = y[:reduced_sample_size]

    timer = TestTime(X, y, X_test, reduced_sample_size, NUM_RUNS)
    with suppress_print():
        avg_sklearn_time, avg_sega_time = timer.measure_performance(
            linear_model.PassiveAggressiveRegressor, PassiveAggressiveRegressor
        )

    print(
        f"\nPassive Aggressive Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):"
    )
    print(f"\tsklearn Passive Aggressive Regression: {avg_sklearn_time:7.4f} seconds")
    print(f"\tsega Passive Aggressive Regression: {avg_sega_time:10.4f} seconds")

    if avg_sklearn_time == 0:
        avg_sklearn_time = 1e-10
    if avg_sega_time == 0:
        avg_sega_time = 1e-10
    if avg_sklearn_time == avg_sega_time:
        print("\tSpeed difference, is negligible")
    elif avg_sklearn_time > avg_sega_time:
        print(
            f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster"
        )
    elif avg_sega_time > avg_sklearn_time:
        print(
            f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster"
        )

    X = original_X
    y = original_y


def main():
    NUM_SAMPLES = 1_000_000
    NUM_RUNS = 5

    # Generate synthetic data
    X, y = synthetic_data_regression(NUM_SAMPLES)
    X_test = X[:1000]  # Use a subset for testing

    # Run tests
    test_ols(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    test_ridge(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    test_lasso(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    test_bayesian(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    test_ransac(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    test_passiveAggressiveRegressor(X, y, X_test, NUM_SAMPLES, NUM_RUNS)


if __name__ == "__main__":
    main()
