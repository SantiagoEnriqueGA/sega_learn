import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sega_learn.linear_models import *
from sega_learn.linear_models import make_sample_data
from utils import synthetic_data_regression, suppress_print

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

class TestTime():
    def __init__(self, X, y, X_test, num_samples, num_runs):
        self.X = X
        self.y = y        
        self.X_test = X_test
        self.num_samples = num_samples
        self.num_runs = num_runs    
        
    def measure_performance(self, model_class_sklearn, model_class_sega):
        def model_performance(model_class):
            start_time = time.time()
            if model_class == RANSAC:
                model = model_class(n=10, k=100, t=0.05, d=self.num_samples//10,
                                    auto_scale_t=True, scale_t_factor=2)
            else:
                model = model_class()
            model.fit(self.X, self.y)
            y_pred = model.predict(self.X_test)
            end_time = time.time()
            return end_time - start_time

        sklearn_times = [model_performance(model_class_sklearn) for _ in range(self.num_runs)]
        sega_times = [model_performance(model_class_sega) for _ in range(self.num_runs)]

        avg_sklearn_time = sum(sklearn_times) / self.num_runs
        std_sklearn_time = np.std(sklearn_times)

        avg_sega_time = sum(sega_times) / self.num_runs
        std_sega_time = np.std(sega_times)

        return avg_sklearn_time, std_sklearn_time, avg_sega_time, std_sega_time

def plot_results(results, title):
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a line plot for the results
    plt.figure(figsize=(12, 6))

    # Plot mean times
    sns.lineplot(x="Sample Size", y="sklearn", markers=True, dashes=False, data=results, label="sklearn", color="blue")
    sns.lineplot(x="Sample Size", y="sega_learn", markers=True, dashes=False, data=results, label="sega_learn", color="orange")

    # Fill standard deviation areas
    plt.fill_between(results["Sample Size"], 
                     results["sklearn"] - results["std_sklearn"], 
                     results["sklearn"] + results["std_sklearn"], 
                     alpha=0.3, color="blue")

    plt.fill_between(results["Sample Size"], 
                     results["sega_learn"] - results["std_sega_learn"], 
                     results["sega_learn"] + results["std_sega_learn"], 
                     alpha=0.3, color="orange")

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Time (s)")
    plt.legend(title="Legend", title_fontproperties={'weight': 'bold'})
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"tests_performance/scalability/plots/linear_models_{title.replace(' ', '_')}.png")
    # plt.show()


def test_compare_libraries(models_to_test, num_runs):
    for models in models_to_test:
        sklearn_model, sega_model, num_zeros = models
        sample_sizes = [10**i for i in range(1, num_zeros)]
        sample_times_sega = []
        sample_times_sklearn = []
        sample_std_sega = []
        sample_std_sklearn = []

        print(f"\n{sega_model.__name__} and {sklearn_model.__name__} Performance comparison:")

        for sample_size in sample_sizes:
            # Generate synthetic data
            X, y = synthetic_data_regression(sample_size)
            X_test = X[:1000]  # Use a subset for testing
            
            timer = TestTime(X, y, X_test, sample_size, num_runs)
            avg_sklearn_time, std_sklearn_time, avg_sega_time, std_sega_time = timer.measure_performance(model_class_sklearn=sklearn_model, model_class_sega=sega_model)
            
            sample_times_sega.append(avg_sega_time)
            sample_std_sega.append(std_sega_time)
            sample_times_sklearn.append(avg_sklearn_time)
            sample_std_sklearn.append(std_sklearn_time)

        # Construct DataFrame
        results = pd.DataFrame({
            "Sample Size": sample_sizes,
            "sklearn": sample_times_sklearn,
            "std_sklearn": sample_std_sklearn,
            "sega_learn": sample_times_sega,
            "std_sega_learn": sample_std_sega
        })

        print(results)
        plot_results(results, f"{sega_model.__name__} Scalability")

def main():   
    NUM_RUNS = 10

    models_to_test = [
        (linear_model.LinearRegression, OrdinaryLeastSquares, 8),
        (linear_model.Ridge, Ridge, 5),
        (linear_model.Lasso, Lasso, 5),
        (linear_model.BayesianRidge, Bayesian, 8),
        (linear_model.RANSACRegressor, RANSAC, 7),
        (linear_model.PassiveAggressiveRegressor, PassiveAggressiveRegressor, 7)
    ]

    test_compare_libraries(models_to_test, NUM_RUNS)


if __name__ == '__main__':
    main()
