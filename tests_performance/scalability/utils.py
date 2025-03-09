import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from _utils import synthetic_data_regression

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sega_learn.utils import *
from sklearn.preprocessing import PolynomialFeatures

NUM_RUNS = 10
NUM_SAMPLES_LIST = [10**i for i in range(3, 7)]

class TestTime():
    def __init__(self, num_runs):
        self.num_runs = num_runs    
        
    def measure_performance(self, transform_class_sklearn, transform_class_sega, X):
        def transform_performance(transform_class):
            start_time = time.time()
            
            if transform_class == PolynomialTransform:
                model = transform_class(degree=2)
                X_poly = model.fit_transform(X)
            elif transform_class == PolynomialFeatures:
                model = transform_class(degree=2)
                X_poly = model.fit_transform(X)
            
            end_time = time.time()
            return end_time - start_time

        sklearn_times = [transform_performance(transform_class_sklearn) for _ in range(self.num_runs)]
        sega_times = [transform_performance(transform_class_sega) for _ in range(self.num_runs)]

        avg_sklearn_time = np.mean(sklearn_times)
        std_sklearn_time = np.std(sklearn_times)

        avg_sega_time = np.mean(sega_times)
        std_sega_time = np.std(sega_times)

        return avg_sklearn_time, std_sklearn_time, avg_sega_time, std_sega_time
    
def plot_results(results, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    sns.lineplot(x="Sample Size", y="sklearn", markers=True, dashes=False, data=results, label="sklearn", color="blue")
    sns.lineplot(x="Sample Size", y="sega_learn", markers=True, dashes=False, data=results, label="sega_learn", color="orange")

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
    plt.savefig(f"tests_performance/scalability/plots/utils_{title.replace(' ', '_')}.png")
    # plt.show()

def test_compare_libraries(num_runs):
    sample_times_sega = []
    sample_times_sklearn = []
    sample_std_sega = []
    sample_std_sklearn = []

    for sample_size in NUM_SAMPLES_LIST:
        X, _ = synthetic_data_regression(n_samples=sample_size, n_features=5, noise=0.1, random_state=42)
        timer = TestTime(num_runs)
        avg_sklearn_time, std_sklearn_time, avg_sega_time, std_sega_time = timer.measure_performance(
            transform_class_sklearn=PolynomialFeatures, transform_class_sega=PolynomialTransform, X=X
        )
        
        sample_times_sega.append(avg_sega_time)
        sample_std_sega.append(std_sega_time)
        sample_times_sklearn.append(avg_sklearn_time)
        sample_std_sklearn.append(std_sklearn_time)
    
    results = pd.DataFrame({
        "Sample Size": NUM_SAMPLES_LIST,
        "sklearn": sample_times_sklearn,
        "std_sklearn": sample_std_sklearn,
        "sega_learn": sample_times_sega,
        "std_sega_learn": sample_std_sega
    })
    
    print(results)
    plot_results(results, "Polynomial Transform Scalability")

def main():
    test_compare_libraries(NUM_RUNS)

if __name__ == '__main__':
    main()
