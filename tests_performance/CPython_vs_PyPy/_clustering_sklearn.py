import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Adjust the path to import the sklearn_learn package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.clustering import *
from sega_learn.utils import make_blobs


# -------------------------------------------------------------
# Functions for testing the performance of the KMeans algorithm
# -------------------------------------------------------------
def test_kmeans_sklearn(X, num_k, num_runs, max_iter):
    from sklearn import cluster

    def sklearn_kmeans():
        start_time = time.time()
        kmeans = cluster.KMeans(n_clusters=num_k, max_iter=max_iter)
        kmeans.fit(X)
        kmeans.predict(X)
        end_time = time.time()
        return end_time - start_time

    sklearn_times = [sklearn_kmeans() for _ in range(num_runs)]
    avg_sklearn_time = sum(sklearn_times) / num_runs
    std_sklearn_time = np.std(sklearn_times)
    return avg_sklearn_time, std_sklearn_time


# -------------------------------------------------------------
# Functions for testing the performance of the DBSCAN algorithm
# -------------------------------------------------------------
def test_dbscan_sklearn(X, num_runs):
    from sklearn import cluster

    def sklearn_dbscan():
        start_time = time.time()
        dbscan = cluster.DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(X)
        end_time = time.time()
        return end_time - start_time

    sklearn_times = [sklearn_dbscan() for _ in range(num_runs)]
    avg_sklearn_time = sum(sklearn_times) / num_runs
    std_sklearn_time = np.std(sklearn_times)
    return avg_sklearn_time, std_sklearn_time


# -------------------------------------------------------------
# Run the tests
# -------------------------------------------------------------
def kmeans_sklearn():
    # Configuration
    num_k = 5
    num_runs = 10
    max_iter = 100
    num_zeros = 6
    sample_sizes = [10**i for i in range(1, num_zeros)]

    results = pd.DataFrame(columns=["sample_size", "avg_time", "std_time"])
    for sample_size in sample_sizes:
        X, _, _ = make_blobs(
            n_samples=sample_size,
            n_features=5,
            centers=num_k,
            cluster_std=0.60,
            random_state=1,
        )

        # Run KMeans tests
        avg_sklearn_time, std_sklearn_time = test_kmeans_sklearn(
            X, num_k, num_runs, max_iter
        )
        print(
            f"sklearn_kmeans sample_size: {sample_size:<-7} avg_time: {avg_sklearn_time:.4f} std_time: {std_sklearn_time:.4f}"
        )
        results.loc[len(results)] = [sample_size, avg_sklearn_time, std_sklearn_time]

    return results


def dbscan_sklearn():
    # Configuration
    num_k = 5
    num_runs = 10
    num_zeros = 5
    sample_sizes = [10**i for i in range(1, num_zeros)]

    results = pd.DataFrame(columns=["sample_size", "avg_time", "std_time"])
    for sample_size in sample_sizes:
        X, _, _ = make_blobs(
            n_samples=sample_size,
            n_features=5,
            centers=num_k,
            cluster_std=0.60,
            random_state=1,
        )

        # Run DBSCAN tests
        avg_sklearn_time, std_sklearn_time = test_dbscan_sklearn(X, num_runs)
        print(
            f"sklearn_dbscan sample_size: {sample_size:<-7} avg_time: {avg_sklearn_time:.4f} std_time: {std_sklearn_time:.4f}"
        )
        results.loc[len(results)] = [sample_size, avg_sklearn_time, std_sklearn_time]

    return results


if __name__ == "__main__":
    os.makedirs("tests_performance/CPython_vs_PyPy/logs_clustering", exist_ok=True)

    results = kmeans_sklearn()
    results["version"] = "sklearn"
    results.to_csv(
        "tests_performance/CPython_vs_PyPy/logs_clustering/kmeans_sklearn.csv",
        index=False,
    )

    results = dbscan_sklearn()
    results["version"] = "sklearn"
    results.to_csv(
        "tests_performance/CPython_vs_PyPy/logs_clustering/dbscan_sklearn.csv",
        index=False,
    )
