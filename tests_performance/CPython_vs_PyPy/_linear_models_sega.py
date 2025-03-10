import os
import sys
import time
import numpy as np
import pandas as pd

from _utils import synthetic_data_regression, suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.linear_models import *
from sega_learn.linear_models import make_sample_data


class TestTime():
    def __init__(self, X, y, X_test, num_samples, num_runs):
        self.X = X
        self.y = y        
        self.X_test = X_test
        self.num_samples = num_samples
        self.num_runs = num_runs    
        
    def measure_performance(self, model_class):
        def model_performance(model_class):
            start_time = time.time()
            model = model_class()
            model.fit(self.X, self.y)
            y_pred = model.predict(self.X_test)
            end_time = time.time()
            return end_time - start_time

        times = [model_performance(model_class) for _ in range(self.num_runs)]
        avg_time = sum(times) / self.num_runs
        std_time = np.std(times)

        return avg_time, std_time

def test_ols(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    timer = TestTime(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    avg_sega_time, std_sega_time = timer.measure_performance(OrdinaryLeastSquares)
        
    print(f"\nOLS Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
    print(f"\tsega Linear Regression: {avg_sega_time:10.4f} ± {std_sega_time:7.4f} seconds")

    return avg_sega_time, std_sega_time    
    
def test_ridge(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    original_X = X
    original_y = y
    reduced_sample_size = NUM_SAMPLES // 100
    X = X[:reduced_sample_size]
    y = y[:reduced_sample_size]
    
    timer = TestTime(X, y, X_test, reduced_sample_size, NUM_RUNS)
    avg_sega_time, std_sega_time = timer.measure_performance(Ridge)
    
    print(f"\nRidge Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):")
    print(f"\tsega Ridge Regression: {avg_sega_time:10.4f} ± {std_sega_time:7.4f} seconds")
    
    X = original_X
    y = original_y

    return avg_sega_time, std_sega_time

def test_lasso(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    original_X = X
    original_y = y
    reduced_sample_size = NUM_SAMPLES // 100
    X = X[:reduced_sample_size]
    y = y[:reduced_sample_size]
    
    timer = TestTime(X, y, X_test, reduced_sample_size, NUM_RUNS)
    avg_sega_time, std_sega_time = timer.measure_performance(Lasso)
    
    print(f"\nLasso Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):")
    print(f"\tsega Lasso Regression: {avg_sega_time:10.4f} ± {std_sega_time:7.4f} seconds")
        
    X = original_X
    y = original_y

    return avg_sega_time, std_sega_time

def test_bayesian(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    timer = TestTime(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    avg_sega_time, std_sega_time = timer.measure_performance(Bayesian)
    
    print(f"\nBayesian Ridge Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
    print(f"\tsega Bayesian Ridge Regression: {avg_sega_time:10.4f} ± {std_sega_time:7.4f} seconds")
    
    return avg_sega_time, std_sega_time

def test_ransac(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    timer = TestTime(X, y, X_test, NUM_SAMPLES, NUM_RUNS)
    avg_sega_time, std_sega_time = timer.measure_performance(RANSAC)
    
    print(f"\nRANSAC Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
    print(f"\tsega RANSAC Regression: {avg_sega_time:10.4f} ± {std_sega_time:7.4f} seconds")
    
    return avg_sega_time, std_sega_time

def test_passiveAggressiveRegressor(X, y, X_test, NUM_SAMPLES, NUM_RUNS):
    original_X = X
    original_y = y
    reduced_sample_size = NUM_SAMPLES // 100
    X = X[:reduced_sample_size]
    y = y[:reduced_sample_size]
    
    timer = TestTime(X, y, X_test, reduced_sample_size, NUM_RUNS)
    with suppress_print():
        avg_sega_time, std_sega_time = timer.measure_performance(PassiveAggressiveRegressor)
    
    print(f"\nPassive Aggressive Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):")
    print(f"\tsega Passive Aggressive Regression: {avg_sega_time:10.4f} ± {std_sega_time:7.4f} seconds")
    
    X = original_X
    y = original_y

    return avg_sega_time, std_sega_time


def main():  
    os.makedirs(f"tests_performance/CPython_vs_PyPy/logs_linear_models", exist_ok=True)

    # Run all tests for increasing sample size
    num_runs = 10
    num_zeros = 7
    sample_sizes = [10**i for i in range(3, num_zeros)]
    results = pd.DataFrame(columns=["model", "sample_size", "avg_time", "std_time"])

    for sample_size in sample_sizes:
        X, y = synthetic_data_regression(sample_size)
        X_test = X[:1000]  # Use a subset for testing
        
        # Run tests
        avg_sklearn_time, std_sklearn_time = test_ols(X,y, X_test, sample_size, num_runs)
        results.loc[len(results)] = ["OLS", sample_size, avg_sklearn_time, std_sklearn_time]

        avg_sklearn_time, std_sklearn_time = test_ridge(X,y, X_test, sample_size, num_runs)
        results.loc[len(results)] = ["Ridge", sample_size, avg_sklearn_time, std_sklearn_time]

        avg_sklearn_time, std_sklearn_time = test_lasso(X,y, X_test, sample_size, num_runs)
        results.loc[len(results)] = ["Lasso", sample_size, avg_sklearn_time, std_sklearn_time]

        avg_sklearn_time, std_sklearn_time = test_bayesian(X,y, X_test, sample_size, num_runs)
        results.loc[len(results)] = ["Bayesian", sample_size, avg_sklearn_time, std_sklearn_time]

        avg_sklearn_time, std_sklearn_time = test_ransac(X,y, X_test, sample_size, num_runs)
        results.loc[len(results)] = ["RANSAC", sample_size, avg_sklearn_time, std_sklearn_time]

        avg_sklearn_time, std_sklearn_time = test_passiveAggressiveRegressor(X,y, X_test, sample_size, num_runs)
        results.loc[len(results)] = ["PassiveAggressive", sample_size, avg_sklearn_time, std_sklearn_time]

    # Get the current python executable
    python_executable = sys.executable
    python_executable = python_executable.split("envs\\")[1]
    python_executable = python_executable.split("\\python.exe")[0]

    results["version"] = python_executable
    results.to_csv(f"tests_performance/CPython_vs_PyPy/logs_linear_models/linear_models_{python_executable}.csv", index=False)

if __name__ == '__main__':
    main()