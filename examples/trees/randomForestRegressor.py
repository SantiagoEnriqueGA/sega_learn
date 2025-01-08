import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sega_learn.trees.randomForestRegressor as rfg
import sega_learn.utils.dataPrep as dp

df = pd.read_csv("example_datasets/output_May-06-2024_cleaned.csv")
df = df[['Miles', 'Stock', 'Year', 'Sub_Model', 'Price']]
df.to_csv("example_datasets/carsDotCom.csv", index=False)

# Source file location
file_orig = "example_datasets/carsDotCom.csv"

# Prepare and format data
df, file_loc = dp.DataPrep.prepare_data(file_orig, label_col_index=4, cols_to_encode=[1, 2, 3])


def basic_example(num_trees=50, max_depth=5):
    """
    Basic example of using the Random Forest Regressor on the Cars.com dataset.
    """
    # Initialize random forest object
    rfObj = rfg.RunRandomForestRegressor(file_loc, forest_size=num_trees, random_seed=0, max_depth=max_depth)
    
    # Train random forest model
    randomForest, stats = rfObj.run()

    print(f"Stats: {stats}\n")
    

def grid_search():
    """
    Grid search example to find the best hyperparameters for the Random Forest Regressor model.
    """    
    # Define the range of forest sizes and maximum depths to test
    forest_sizes = [10, 20, 50]
    max_depths = [5, 10, 20]

    # Store results
    results = {metric: np.zeros((len(forest_sizes), len(max_depths))) for metric in ['MSE', 'R^2', 'MAPE', 'MAE', 'RMSE']}

    # Loop over different forest sizes and maximum depths
    for i, forest_size in enumerate(forest_sizes):
        for j, max_depth in enumerate(max_depths):
            # Initialize random forest object
            rfObj = rfg.RunRandomForestRegressor(file_loc, forest_size=forest_size, random_seed=0, max_depth=max_depth)
            
            # Train random forest model and get stats
            randomForest, stats = rfObj.run()
            for metric in results:
                results[metric][i, j] = stats[metric]

            print(f"Forest Size: {forest_size}, Max Depth: {max_depth}, Stats: {stats}\n")

    # Plot the results for each metric
    for metric in results:
        plt.figure(figsize=(12, 6))
        for i, forest_size in enumerate(forest_sizes):
            plt.plot(max_depths, results[metric][i, :], label=f'Forest Size: {forest_size}')
        
        plt.xlabel('Max Depth')
        plt.ylabel(metric)
        plt.title(f'Random Forest Regressor {metric} for Different Forest Sizes and Max Depths (Cars.com Dataset)')
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    print("\n\nRandom Forest Regressor on Cars.com dataset\n")
    basic_example()
    # grid_search()