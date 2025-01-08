import matplotlib.pyplot as plt
import numpy as np

# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sega_learn.trees.gradientBoostedRegressor as gbr
import sega_learn.utils.dataPrep as dp

# Source file location
file_orig = "example_datasets/carsDotCom.csv"

# Prepare and format data
df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1, 2, 3])


def basic_example(num_trees=50, max_depth=5):
    """
    Basic example of using the Gradient Boosted Regressor on the Cars.com dataset.
    """
    # Initialize GBDT object
    gbdtDiab = gbr.GradientBoostedRegressor(file_loc, num_trees=num_trees, random_seed=0, max_depth=max_depth)
    
    # Train GBDT model
    gbdtDiab.fit(stats=True)

    # Predict target values
    predictions = gbdtDiab.predict()

    # Get stats
    stats = gbdtDiab.get_stats(predictions)
    print(stats)


def grid_search():
    """
    Grid search example to find the best hyperparameters for the GBDT model.
    """
    # Define the range of numbers of trees and maximum depths to test
    num_trees_list = [10, 20, 100]
    max_depths = [5, 10, 20]

    # Store results
    results = {metric: np.zeros((len(num_trees_list), len(max_depths))) for metric in ['MSE', 'R^2', 'MAPE', 'MAE', 'RMSE']}
    mean_absolute_residuals = {}

    # Loop over different numbers of trees and maximum depths
    for i, num_trees in enumerate(num_trees_list):
        for j, max_depth in enumerate(max_depths):
            # Initialize GBDT object
            gbdtDiab = gbr.GradientBoostedRegressor(file_loc, num_trees=num_trees, random_seed=0, max_depth=max_depth)
            
            # Train GBDT model
            gbdtDiab.fit(stats=True)

            # Predict target values
            predictions = gbdtDiab.predict()

            # Get stats
            stats = gbdtDiab.get_stats(predictions)
            for metric in results:
                results[metric][i, j] = stats[metric]
            mean_absolute_residuals[(num_trees, max_depth)] = stats['Mean_Absolute_Residuals']

            print(f"Num Trees: {num_trees}, Max Depth: {max_depth}, Stats: {stats}\n")

    # Plot the results for each metric
    for metric in results:
        plt.figure(figsize=(12, 6))
        for i, num_trees in enumerate(num_trees_list):
            plt.plot(max_depths, results[metric][i, :], label=f'Num Trees: {num_trees}')
        
        plt.xlabel('Max Depth')
        plt.ylabel(metric)
        plt.title(f'Gradient Boosted Regressor {metric} for Different Numbers of Trees and Max Depths (Cars.com Dataset)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot Mean Absolute Residuals
    plt.figure(figsize=(12, 6))
    for (num_trees, max_depth), residuals in mean_absolute_residuals.items():
        plt.plot(range(len(residuals)), residuals, label=f'Num Trees: {num_trees}, Max Depth: {max_depth}')

    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Residual')
    plt.title('Gradient Boosted Regressor Mean Absolute Residuals for Different Numbers of Trees and Max Depths (Cars.com Dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    print("\n\nGradient Boosted Regressor on Cars.com dataset\n")
    basic_example()
    # grid_search()