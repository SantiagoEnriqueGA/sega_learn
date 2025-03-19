import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.linear_models import Ridge
from sega_learn.utils import make_regression
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

import time
import pandas as pd
from sklearn.linear_model import Ridge as SklearnRidge

num_zeros = 8
sample_sizes = [10**i for i in range(1, num_zeros)]

def run_ridge_jit(X, y):
    reg = Ridge(alpha=1.0, fit_intercept=True)
    
    start_time = time.time()
    reg.fit(X, y, numba=True)
    end_time = time.time()
    
    return end_time - start_time

def run_ridge_sk(X, y):
    reg = SklearnRidge(alpha=1.0, fit_intercept=True)
    reg.set_params(max_iter=10000)
    reg.set_params(tol=0.0001)
    
    start_time = time.time()
    reg.fit(X, y)
    end_time = time.time()
    
    return end_time - start_time


# Create list to store results
results = []
for sample_size in sample_sizes:
    X, y = make_regression(n_samples=sample_size, n_features=5, noise=.5, random_state=42)
    
    # Average the time over multiple runs
    n_runs = 5
    jit_time_total = non_jit_time_total = 0
    for i in range(n_runs):
        jit_time_total += run_ridge_jit(X, y)
        non_jit_time_total += run_ridge_sk(X, y)
    
    jit_time = jit_time_total / n_runs
    non_jit_time = non_jit_time_total / n_runs
    
    # Append the results to the DataFrame
    results.append((sample_size, jit_time, non_jit_time))
    
    
print("Numba vs Non-Numba Ridge Regression Times")
print("-"*80)
# Convert the results to a DataFrame
df = pd.DataFrame(results, columns=['Sample Size', 'Numba Time', 'Sklearn Time'])
print(df)
