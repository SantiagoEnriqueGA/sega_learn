import time

import numpy as np
from sega_learn.trees.treeRegressor import RegressorTree
from sklearn.model_selection import train_test_split

# Parameters
num_samples = 100_000
num_features = 10
test_size = 0.2
max_depth_range = [1, 2, 3, 4, 5]
num_trials = 10

# Generate synthetic data
np.random.seed(42)  # For reproducibility
X = np.random.rand(num_samples, num_features)
true_coefficients = np.random.rand(num_features)
y = np.dot(X, true_coefficients) + np.random.normal(0, 1, num_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
num_test_samples = len(X_test)

# Test for each max_depth
for max_depth in max_depth_range:
    training_times = []
    prediction_times = []
    for _trial in range(num_trials):
        tree = RegressorTree(max_depth=max_depth)

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time
        training_times.append(training_time)

        # Measure prediction time
        start_time = time.perf_counter()
        predictions = tree.predict(X_test)
        prediction_time = time.perf_counter() - start_time
        prediction_times.append(prediction_time)

    # Compute averages
    avg_training_time = np.mean(training_times)
    avg_prediction_time = np.mean(prediction_times)
    avg_prediction_time_per_sample = avg_prediction_time / num_test_samples

    # Print results
    print(
        f"max_depth={max_depth}, "
        f"avg_training_time={avg_training_time:.4f} s, "
        f"avg_prediction_time_per_sample={avg_prediction_time_per_sample:.6f} s"
    )
