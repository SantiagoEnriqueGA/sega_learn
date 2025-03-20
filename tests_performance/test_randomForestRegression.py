import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.trees import RandomForestRegressor as rfr
from sega_learn.utils import make_regression
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor as SklearnRFR

num_zeros = 8
sample_sizes = [10**i for i in range(1, num_zeros)]

def run_forest(X, y, num_trees):
    reg = rfr(forest_size=num_trees, max_depth=5, random_seed=0)
    
    start_time = time.time()
    reg.fit(X, y)
    end_time = time.time()
    
    return end_time - start_time


def run_forest_sk(X, y, num_trees):
    reg = SklearnRFR(n_estimators=num_trees, max_depth=5, random_state=0)
    
    start_time = time.time()
    reg.fit(X, y)
    end_time = time.time()
    
    return end_time - start_time

# Create list to store results
results = []
X, y = make_regression(n_samples=1_000, n_features=5, noise=.5, random_state=42)
trees = [2, 5, 10]
for num_trees in trees:
    # Average the time over multiple runs
    n_runs = 1
    base_time_total = jit_time_total = skl_time_total = 0
    for i in range(n_runs):
        base_time_total += run_forest(X, y, num_trees)
        # jit_time_total += run_ridge_jit(X, y)
        skl_time_total += run_forest_sk(X, y, num_trees)
        
    base_time = base_time_total / n_runs
    jit_time = jit_time_total / n_runs
    skl_time = skl_time_total / n_runs
    
    # Append the results to the DataFrame
    # results.append((num_trees, base_time, jit_time, skl_time))
    results.append((num_trees, base_time, skl_time))
    
    
print("Numba vs Non-Numba vs Sklearn Random Forest Regression Times")
print("-"*80)
# Convert the results to a DataFrame
# df = pd.DataFrame(results, columns=['Sample Size', 'Base Time', 'JIT Time', 'Sklearn Time'])
df = pd.DataFrame(results, columns=['Number of Trees', 'Base Time', 'Sklearn Time'])
print(df)



def plot_results(results, title):
    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create a line plot for the results
    plt.figure(figsize=(12, 6))

    # Plot mean times
    sns.lineplot(x="Number of Trees", y="Sklearn Time", markers=True, dashes=False, data=results, label="sklearn", color="blue")
    # sns.lineplot(x="Sample Size", y="JIT Time", markers=True, dashes=False, data=results, label="sega_learn", color="orange")
    sns.lineplot(x="Number of Trees", y="Base Time", markers=True, dashes=False, data=results, label="sega_learn (no jit)", color="green")    

    # plt.xscale("log")
    plt.title(title)
    plt.xlabel("Number of Trees")
    plt.ylabel("Time (s)")
    plt.legend(title="Legend", title_fontproperties={'weight': 'bold'})
    plt.tight_layout()
    plt.grid(True)
    # plt.savefig(f"tests_performance/scalability/plots/linear_models_{title.replace(' ', '_')}.png")
    plt.show()

plot_results(df, "Random Forest Regression Performance Comparison")