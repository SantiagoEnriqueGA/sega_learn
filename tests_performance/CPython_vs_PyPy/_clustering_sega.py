import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Adjust the path to import the sega_learn package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.clustering import *
from sega_learn.utils import make_blobs

# -------------------------------------------------------------
# Functions for testing the performance of the KMeans algorithm
# -------------------------------------------------------------
def test_kmeans_sega(X, num_k, num_runs, max_iter):

    def sega_kmeans():
        start_time = time.time()
        kmeans = KMeans(X, n_clusters=num_k, max_iter=max_iter)
        kmeans.fit()
        kmeans.predict(X)
        end_time = time.time()
        return end_time - start_time

    sega_times = [sega_kmeans() for _ in range(num_runs)]
    avg_sega_time = sum(sega_times) / num_runs
    std_sega_time = np.std(sega_times)
    return avg_sega_time, std_sega_time

# -------------------------------------------------------------
# Functions for testing the performance of the DBSCAN algorithm
# -------------------------------------------------------------
def test_dbscan_sega(X, num_runs):
    
    def sega_dbscan():
        start_time = time.time()
        dbscan = DBSCAN(X, eps=0.5, min_samples=5)
        dbscan.fit()
        end_time = time.time()
        return end_time - start_time

    sega_times = [sega_dbscan() for _ in range(num_runs)]
    avg_sega_time = sum(sega_times) / num_runs
    std_sega_time = np.std(sega_times)
    return avg_sega_time, std_sega_time


# -------------------------------------------------------------
# Run the tests
# -------------------------------------------------------------
def kmeans_sega():
    # Configuration
    num_k = 5
    num_runs = 10
    max_iter = 100
    num_zeros = 6
    sample_sizes = [10**i for i in range(1, num_zeros)]

    results = pd.DataFrame(columns=["sample_size", "avg_time", "std_time"])
    for sample_size in sample_sizes:
        X, _, _ = make_blobs(n_samples=sample_size, n_features=5, centers=num_k, cluster_std=0.60, random_state=1)
        
        # Run KMeans tests
        avg_sega_time, std_sega_time = test_kmeans_sega(X, num_k, num_runs, max_iter)
        print(f"sega_kmeans sample_size: {sample_size:<-7} avg_time: {avg_sega_time:.4f} std_time: {std_sega_time:.4f}")
        results.loc[len(results)] = [sample_size, avg_sega_time, std_sega_time]
    
    return results

def dbscan_sega():
    # Configuration
    num_k = 5
    num_runs = 10
    num_zeros = 5
    sample_sizes = [10**i for i in range(1, num_zeros)]
    
    results = pd.DataFrame(columns=["sample_size", "avg_time", "std_time"])
    for sample_size in sample_sizes:
        X, _, _ = make_blobs(n_samples=sample_size, n_features=5, centers=num_k, cluster_std=0.60, random_state=1)
        
        # Run DBSCAN tests
        avg_sega_time, std_sega_time = test_dbscan_sega(X, num_runs)
        print(f"sega_dbscan sample_size: {sample_size:<-7} avg_time: {avg_sega_time:.4f} std_time: {std_sega_time:.4f}")
        results.loc[len(results)] = [sample_size, avg_sega_time, std_sega_time]

    return results

if __name__ == "__main__":
    os.makedirs(f"tests_performance/CPython_vs_PyPy/logs_clustering", exist_ok=True)

    # Get the current python executable
    python_executable = sys.executable
    python_executable = python_executable.split("envs\\")[1]
    python_executable = python_executable.split("\\python.exe")[0]

    results = kmeans_sega()
    results["version"] = python_executable
    results.to_csv(f"tests_performance/CPython_vs_PyPy/logs_clustering/kmeans_{python_executable}.csv", index=False)
    
    results = dbscan_sega()
    results["version"] = python_executable
    results.to_csv(f"tests_performance/CPython_vs_PyPy/logs_clustering/dbscan_{python_executable}.csv", index=False)