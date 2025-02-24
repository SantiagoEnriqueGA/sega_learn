import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import warnings
# warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sega_learn.nearest_neighbors import *
from utils import suppress_print

from sklearn.datasets import make_regression, make_classification
from sklearn import neighbors


MAX_ITER = 100

class TestTime():
    def __init__(self, X, y, num_samples, num_runs):
        self.X = X
        self.y = y        
        self.num_samples = num_samples
        self.num_runs = num_runs    
        
    def measure_performance(self, model_class_sklearn, model_class_sega, neighbors_to_test):
        def model_performance(model_class):
            start_time = time.time()

            model = model_class(n_neighbors=neighbors_to_test)
            model.fit(self.X, self.y)
            y_pred = model.predict(self.X)

            end_time = time.time()
            return end_time - start_time

        sklearn_times = [model_performance(model_class_sklearn) for _ in range(self.num_runs)]
        sega_times = [model_performance(model_class_sega) for _ in range(self.num_runs)]

        avg_sklearn_time = sum(sklearn_times) / self.num_runs
        std_sklearn_time = np.std(sklearn_times)

        avg_sega_time = sum(sega_times) / self.num_runs
        std_sega_time = np.std(sega_times)

        return avg_sklearn_time, std_sklearn_time, avg_sega_time, std_sega_time
    
def test_compare_libraries(models_to_test, neighbors_to_test, num_runs):
    for models in models_to_test:
        sklearn_model, sega_model, neighbors_to_test, num_zeros = models
        sample_sizes = [10**i for i in range(1, num_zeros)]

        print(f"\n{sega_model.__name__} and {sklearn_model.__name__} Performance comparison:")

        # Create empty list to store results
        results_data = []

        for n_neighbors in neighbors_to_test:
            for sample_size in sample_sizes:
                # Generate synthetic data
                if sklearn_model == neighbors.KNeighborsRegressor:
                    X, y = make_regression(n_samples=sample_size, n_features=5, n_informative=3, noise=0.1, random_state=1)
                elif sklearn_model == neighbors.KNeighborsClassifier:
                    X, y = make_classification(n_samples=sample_size, n_features=5, n_informative=3, n_classes=2, random_state=1)

                timer = TestTime(X, y, sample_size, num_runs)
                avg_sklearn_time, std_sklearn_time, avg_sega_time, std_sega_time = timer.measure_performance(
                    model_class_sklearn=sklearn_model,
                    model_class_sega=sega_model,
                    neighbors_to_test=n_neighbors
                )

                # Append result as a dictionary
                results_data.append({
                    "Sample Size": sample_size,
                    "N Neighbors": n_neighbors,
                    "sklearn": avg_sklearn_time,
                    "std_sklearn": std_sklearn_time,
                    "sega_learn": avg_sega_time,
                    "std_sega_learn": std_sega_time
                })

        # Create DataFrame from collected results
        results = pd.DataFrame(results_data)
        
        # Sort results by N Neighbors and Sample Size for better visualization
        results = results.sort_values(['N Neighbors', 'Sample Size'])
        
        print(results)
        plot_results(results, f"{sega_model.__name__} Scalability for {neighbors_to_test} Neighbors")

def plot_results(results, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Create color palettes for sklearn and sega
    sklearn_colors = sns.color_palette("Blues", n_colors=len(results['N Neighbors'].unique()))
    sega_colors = sns.color_palette("Oranges", n_colors=len(results['N Neighbors'].unique()))

    # Plot lines for each k value
    legend_added = False
    for idx, k in enumerate(sorted(results['N Neighbors'].unique())):
        subset = results[results['N Neighbors'] == k]
        
        # Plot sklearn line
        sns.lineplot(x="Sample Size", y="sklearn", data=subset, 
                    label="sklearn" if not legend_added else None,
                    color=sklearn_colors[idx])
        
        # Plot sega line
        sns.lineplot(x="Sample Size", y="sega_learn", data=subset, 
                    label="sega_learn" if not legend_added else None,
                    color=sega_colors[idx])
        
        # Add confidence intervals
        plt.fill_between(subset["Sample Size"], 
                        subset["sklearn"] - subset["std_sklearn"], 
                        subset["sklearn"] + subset["std_sklearn"], 
                        alpha=0.2, color=sklearn_colors[idx])
        
        plt.fill_between(subset["Sample Size"], 
                        subset["sega_learn"] - subset["std_sega_learn"], 
                        subset["sega_learn"] + subset["std_sega_learn"], 
                        alpha=0.2, color=sega_colors[idx])
        
        legend_added = True

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Time (s)")
    plt.legend(title="Legend", title_fontproperties={'weight': 'bold'})
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"tests_performance/scalability/plots/nearest_neighbors_{title.replace(' ', '_')}.png")


def main():   
    NUM_RUNS = 10
    NEIGHBORS_TO_TEST = [2, 5, 10]

    models_to_test = [
        (neighbors.KNeighborsRegressor, KNeighborsRegressor, NEIGHBORS_TO_TEST, 4),
        (neighbors.KNeighborsClassifier, KNeighborsClassifier, NEIGHBORS_TO_TEST, 4)
    ]

    test_compare_libraries(models_to_test, NEIGHBORS_TO_TEST, NUM_RUNS)


if __name__ == '__main__':
    main()
