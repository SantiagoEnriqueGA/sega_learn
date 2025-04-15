import os

# Import Custom Classes
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sega_learn.trees.gradientBoostedClassifier as gbc
import sega_learn.utils.dataPrep as dp
from sega_learn.utils import Metrics


def basic_example(num_trees=10, max_depth=5):
    """Basic example of using the Gradient Boosted Classifier on a synthetic dataset."""
    print("\n\nGradient Boosted Classifier on Synthetic Dataset\n")

    from sega_learn.utils import make_classification

    X, y = make_classification(
        n_samples=1000, n_features=5, n_classes=2, random_state=42
    )

    # Initialize random forest object
    gbObj = gbc.GradientBoostedClassifier(
        X=X, y=y, max_depth=max_depth, n_estimators=num_trees, random_seed=0
    )

    # Train random forest model
    gbObj.fit(verbose=True)

    print(f"Accuracy: {Metrics.accuracy(y, gbObj.predict(X))}")


def cancer_example(num_trees=10, max_depth=5):
    """Basic example of using the Gradient Boosted Classifier on the Wisconsin Breast Prognostic dataset."""
    print("\n\nGradient Boosted Classifier on Wisconsin Breast Prognostic dataset\n")

    # Source file location
    file_orig = "example_datasets/Wisconsin_breast_prognostic.csv"
    df = pd.read_csv(file_orig)  # Load the CSV file
    X, y = dp.DataPrep.df_to_ndarray(df, y_col=-1)

    # Initialize random forest object
    gbObj = gbc.GradientBoostedClassifier(
        X=X, y=y, max_depth=max_depth, n_estimators=num_trees, random_seed=0
    )

    # Train random forest model
    gbObj.fit()

    print(f"Accuracy: {Metrics.accuracy(y, gbObj.predict(X))}")


def grid_search():
    """Grid search example to find the best hyperparameters for the Gradient Boosted Classifier model."""
    print(
        "\n\nGradient Boosted Classifier Grid Search on Wisconsin Breast Prognostic dataset\n"
    )

    # Source file location
    file_orig = "example_datasets/Wisconsin_breast_prognostic.csv"
    df = pd.read_csv(file_orig)  # Load the CSV file
    X, y = dp.DataPrep.df_to_ndarray(df, y_col=-1)

    # Define the range of forest sizes and maximum depths to test
    forest_sizes = [10, 20, 100]
    max_depths = [2, 10, 15]

    # Store results
    results = np.zeros((len(forest_sizes), len(max_depths)))

    # Loop over different forest sizes and maximum depths
    for i, forest_size in enumerate(forest_sizes):
        for j, max_depth in enumerate(max_depths):
            # Initialize random forest object
            gbObj = gbc.GradientBoostedClassifier(
                X=X, y=y, max_depth=max_depth, n_estimators=forest_size, random_seed=0
            )

            # Train random forest model and get accuracy
            gbObj.fit()
            stats = gbObj.get_stats(y, y_pred=gbObj.predict(X))
            results[i, j] = stats["Accuracy"]

            print(
                f"Forest Size: {forest_size}, Max Depth: {max_depth}, Accuracy: {stats['Accuracy']:.4f}\n"
            )

    # Plot the results
    plt.figure(figsize=(12, 6))
    for i, forest_size in enumerate(forest_sizes):
        plt.plot(max_depths, results[i, :], label=f"Forest Size: {forest_size}")

    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title(
        "Gradient Boosted Classifier Accuracy for Different Forest Sizes and Max Depths (Breast Cancer Dataset)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    basic_example()
    # cancer_example()
    # grid_search()
