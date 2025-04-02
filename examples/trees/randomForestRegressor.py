import os

# Import Custom Classes
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sega_learn.trees.randomForestRegressor as rfg
import sega_learn.utils.dataPrep as dp

# Load and prepare dataset
df = pd.read_csv("example_datasets/output_May-06-2024_cleaned.csv")
df = df[["Miles", "Stock", "Year", "Sub_Model", "Price"]]
df.to_csv("example_datasets/carsDotCom.csv", index=False)

# Source file location
file_orig = "example_datasets/carsDotCom.csv"

# Prepare and format data
df, file_loc = dp.DataPrep.prepare_data(
    file_orig, label_col_index=4, cols_to_encode=[1, 2, 3]
)
X, y = dp.DataPrep.df_to_ndarray(df, y_col=4)


def basic_example(num_trees=10, max_depth=5):
    """
    Basic example of using the Random Forest Regressor on a synthetic dataset.
    """
    print("\n\nRandom Forest Regressor on Synthetic Dataset\n")

    from sega_learn.utils import make_regression

    X, y = make_regression(n_samples=1000, n_features=3, noise=5, random_state=42)

    # Initialize random forest object
    rfObj = rfg.RandomForestRegressor(
        X=X, y=y, max_depth=max_depth, forest_size=num_trees, random_seed=0
    )

    # Train random forest model
    rfObj.fit()


def cars_example(num_trees=50, max_depth=5):
    """
    Basic example of using the Random Forest Regressor on the Cars.com dataset.
    """
    print("\n\nRandom Forest Regressor on Cars.com dataset\n")

    # Initialize random forest object
    rfObj = rfg.RandomForestRegressor(
        X=X, y=y, max_depth=max_depth, forest_size=num_trees, random_seed=0
    )

    # Train random forest model
    rfObj.fit()


def grid_search():
    """
    Grid search example to find the best hyperparameters for the Random Forest Regressor model.
    """
    print("\n\nRandom Forest Regressor Grid Search on Cars.com dataset\n")

    # Define the range of forest sizes and maximum depths to test
    forest_sizes = [10, 20, 50]
    max_depths = [5, 10, 20]

    # Store results
    results = {
        metric: np.zeros((len(forest_sizes), len(max_depths)))
        for metric in ["MSE", "R^2", "MAPE", "MAE", "RMSE"]
    }

    # Loop over different forest sizes and maximum depths
    for i, forest_size in enumerate(forest_sizes):
        for j, max_depth in enumerate(max_depths):
            # Initialize random forest object
            rfObj = rfg.RandomForestRegressor(
                X=X, y=y, max_depth=max_depth, forest_size=forest_size, random_seed=0
            )

            # Train random forest model and get stats
            rfObj.fit()
            stats = rfObj.get_stats()

            for metric in results:
                results[metric][i, j] = stats[metric]

            print(
                f"Forest Size: {forest_size}, Max Depth: {max_depth}, Stats: {stats}\n"
            )

    # Plot the results for each metric
    for metric in results:
        plt.figure(figsize=(12, 6))
        for i, forest_size in enumerate(forest_sizes):
            plt.plot(
                max_depths, results[metric][i, :], label=f"Forest Size: {forest_size}"
            )

        plt.xlabel("Max Depth")
        plt.ylabel(metric)
        plt.title(
            f"Random Forest Regressor {metric} for Different Forest Sizes and Max Depths (Cars.com Dataset)"
        )
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    basic_example()
    # cars_example()
    # grid_search()
