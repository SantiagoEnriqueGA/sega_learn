import unittest
import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.nearest_neighbors import *
import sega_learn.nearest_neighbors

from sega_learn.utils import make_classification, make_regression


def time_function(func, num_repeats, *args, **kwargs):
    times = []
    for _ in range(num_repeats):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = np.mean(times)
    stddev_time = np.std(times)
    return avg_time, stddev_time, result

def time_knn_classifiers(num_repeats=10, n_neighbors=list, n_records=list, n_jobs=int, num_features=int, num_classes=int):
    from sklearn.neighbors import KNeighborsClassifier as sklearn_knn
    
    knn_classifiers = [KNeighborsClassifier, sklearn_knn]
        
    # Create a DataFrame to store the results with correct data types
    results = pd.DataFrame(columns=["Module", "N_Neighbors", "N_Records", "Time (s)", "StdDev (s)"],
                           dtype=object)
    
    total_iterations = len(n_neighbors) * len(n_records) * len(knn_classifiers)
    with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:    
        for n in n_neighbors:
            for n_record in n_records:
                # Generate synthetic data
                X, y = make_classification(n_samples=n_record, n_features=num_features, n_classes=num_classes)
                
                for knn_classifier in knn_classifiers:
                    # Create an instance of the classifier
                    if knn_classifier == KNeighborsClassifier:
                        knn_classifier = KNeighborsClassifier(n_neighbors=n)
                    else:
                        knn_classifier = sklearn_knn(n_neighbors=n, n_jobs=n_jobs)
                    
                    avg_time, stddev_time, _ = time_function(knn_classifier.fit, num_repeats, X, y)
                    
                    # Append the results to the DataFrame
                    results.loc[len(results)] = ["sega_learn" if isinstance(knn_classifier, KNeighborsClassifier) else "sklearn",
                                                n, n_record, avg_time, stddev_time]
                    pbar.update(1)  # Update the progress bar for each iteration
    
    return results
    

def time_knn_regressors(num_repeats=10, n_neighbors=list, n_records=list, n_jobs=int, num_features=int):
    from sklearn.neighbors import KNeighborsRegressor as sklearn_knn
    
    knn_regressors = [KNeighborsRegressor, sklearn_knn]
        
    # Create a DataFrame to store the results with correct data types
    results = pd.DataFrame(columns=["Module", "N_Neighbors", "N_Records", "Time (s)", "StdDev (s)"],
                           dtype=object)
    
    total_iterations = len(n_neighbors) * len(n_records) * len(knn_regressors)
    with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:    
        for n in n_neighbors:
            for n_record in n_records:
                # Generate synthetic data
                X, y = make_regression(n_samples=n_record, n_features=num_features)
                
                for knn_regressor in knn_regressors:
                    # Create an instance of the regressor
                    if knn_regressor == KNeighborsRegressor:
                        knn_regressor = KNeighborsRegressor(n_neighbors=n)
                    else:
                        knn_regressor = sklearn_knn(n_neighbors=n, n_jobs=n_jobs)
                    
                    avg_time, stddev_time, _ = time_function(knn_regressor.fit, num_repeats, X, y)
                    
                    # Append the results to the DataFrame
                    results.loc[len(results)] = ["sega_learn" if isinstance(knn_regressor, KNeighborsRegressor) else "sklearn",
                                                n, n_record, avg_time, stddev_time]
                    pbar.update(1)
    
    return results
    
def plot_results(results, title):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Group the results by Module and N_Neighbors
    grouped_results = results.groupby(["Module", "N_Neighbors", "N_Records"]).agg({"Time (s)": "mean", "StdDev (s)": "mean"}).reset_index()

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")
    # Create a line plot for the results
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="N_Records", y="Time (s)", hue="Module", style="N_Neighbors", markers=True, dashes=False, data=grouped_results)
    
    # Fill standard deviation areas
    for module in grouped_results["Module"].unique():
        for n_neighbors in grouped_results["N_Neighbors"].unique():
            subset = grouped_results[(grouped_results["Module"] == module) & (grouped_results["N_Neighbors"] == n_neighbors)]
            plt.fill_between(subset["N_Records"], subset["Time (s)"] - subset["StdDev (s)"], subset["Time (s)"] + subset["StdDev (s)"], alpha=0.2)

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Number of Records")
    plt.ylabel("Time (s)")
    plt.legend(title="Legend", title_fontproperties={'weight': 'bold'})
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"tests_performance/nearest_neighbors/{title}.png")
    # plt.show()
    
def run():
    # Define parameters for testing
    n_neighbors = [10, 20, 50, 500]
    n_records = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    num_features = 20
    num_classes = 2
    num_repeats = 5
    n_jobs = 1
    
    # Run the tests
    print("Testing KNN Classifiers...")
    class_times = time_knn_classifiers(num_repeats=num_repeats, n_neighbors=n_neighbors, n_records=n_records, n_jobs=n_jobs, num_features=num_features, num_classes=num_classes)
    plot_results(class_times, "KNN Classifier Performance")
    
    print("\nTesting KNN Regressors...")
    reg_times = time_knn_regressors(num_repeats=num_repeats, n_neighbors=n_neighbors, n_records=n_records, n_jobs=n_jobs, num_features=num_features)
    plot_results(reg_times, "KNN Regressor Performance")
    
    
if __name__ == "__main__":
    run()