import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from sega_learn.clustering import *
from sega_learn.utils import make_blobs

np.random.seed(0)

import time

import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN as SklearnDBSCAN


def run_dbscan(X, y):
    clust = DBSCAN(X, eps=0.5, min_samples=10)

    start_time = time.time()
    clust.fit()
    end_time = time.time()

    return end_time - start_time


def run_dbscan_jit(X, y):
    clust = DBSCAN(X, eps=0.5, min_samples=10, compile_numba=True)

    start_time = time.time()
    clust.fit(numba=True)
    end_time = time.time()

    return end_time - start_time


def run_dbscan_sk(X, y):
    clust = SklearnDBSCAN(eps=0.5, min_samples=10)

    start_time = time.time()
    clust.fit(X)
    end_time = time.time()

    return end_time - start_time


num_zeros = 5
sample_sizes = [10**i for i in range(1, num_zeros)]

# Create list to store results
results = []
for sample_size in sample_sizes:
    true_k = 8
    X, y, _ = make_blobs(
        n_samples=sample_size,
        n_features=2,
        centers=true_k,
        cluster_std=0.60,
        random_state=1,
    )

    # Average the time over multiple runs
    n_runs = 3
    base_time_total = 0
    jit_time_total = 0
    skl_time_total = 0
    for i in range(n_runs):
        base_time_total += run_dbscan(X, y)
        jit_time_total += run_dbscan_jit(X, y)
        skl_time_total += run_dbscan_sk(X, y)

    base_time = base_time_total / n_runs
    jit_time = jit_time_total / n_runs
    skl_time = skl_time_total / n_runs

    # Append the results to the DataFrame
    results.append((sample_size, base_time, jit_time, skl_time))


print("Numba vs Non-Numba DBSCAN Times")
print("-" * 80)
# Convert the results to a DataFrame
df = pd.DataFrame(
    results, columns=["Sample Size", "Base Time", "JIT Time", "Sklearn Time"]
)
print(df)


def plot_results(results, title):
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a line plot for the results
    plt.figure(figsize=(12, 6))

    # Plot mean times
    sns.lineplot(
        x="Sample Size",
        y="Sklearn Time",
        markers=True,
        dashes=False,
        data=results,
        label="sklearn",
        color="blue",
    )
    sns.lineplot(
        x="Sample Size",
        y="JIT Time",
        markers=True,
        dashes=False,
        data=results,
        label="sega_learn",
        color="orange",
    )
    sns.lineplot(
        x="Sample Size",
        y="Base Time",
        markers=True,
        dashes=False,
        data=results,
        label="sega_learn (no jit)",
        color="green",
    )

    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Time (s)")
    plt.legend(title="Legend", title_fontproperties={"weight": "bold"})
    plt.tight_layout()
    plt.grid(True)
    # plt.savefig(f"tests_performance/scalability/plots/linear_models_{title.replace(' ', '_')}.png")
    plt.show()


plot_results(df, "DBSCAN Performance")
