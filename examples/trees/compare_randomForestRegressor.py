import numpy as np
import time 

# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sega_learn.trees.randomForestRegressor as rfg

from sega_learn.utils import make_regression
X, y = make_regression(n_samples=1000, n_features=3, noise=.5, random_state=42)


start_time = time.time()
# Initialize random forest object
rfObj = rfg.RandomForestRegressor(X, y, forest_size=10, random_seed=0, max_depth=5)

# Train random forest model
rfObj.fit()
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# stats dict
stats = rfObj.get_stats()
for key, value in stats.items():
    print(f"\t{key}: {value:.2f}")
    
# --------------------------------------------------------------------------------------------

import sega_learn.trees_dev.randomForestRegressor as rfg

start_time = time.time()
# Initialize random forest object
rfObj = rfg.RandomForestRegressor(X=X, y=y, forest_size=10, random_seed=0, max_depth=5)

# Train random forest model
rfObj.fit()
end_time = time.time()
print(f"\nTraining time: {end_time - start_time:.2f} seconds")
stats = rfObj.get_stats()
for key, value in stats.items():
    print(f"\t{key}: {value:.2f}")
