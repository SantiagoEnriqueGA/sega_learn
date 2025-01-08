import matplotlib.pyplot as plt
import numpy as np

# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sega_learn.trees.randomForestClassifier as rfc
import sega_learn.utils.dataPrep as dp

# Source file location
file_orig = "example_datasets/Wisconsin_breast_prognostic.csv"

# File already formatted
file_loc = file_orig


def basic_example(num_trees=50, max_depth=5):
    """
    Basic example of using the Random Forest Classifier on the Wisconsin Breast Prognostic dataset.
    """
    # Initialize random forest object
    rfObjBreastCancer = rfc.RunRandomForestClassifier(file_loc, False, forest_size=num_trees, random_seed=0, max_depth=max_depth)
    
    # Train random forest model
    randomForest, accuracy = rfObjBreastCancer.run()

    print(f"Accuracy: {accuracy}\n")
    
def grid_search():
    """
    Grid search example to find the best hyperparameters for the Random Forest Classifier model.
    """    
    # Define the range of forest sizes and maximum depths to test
    forest_sizes = [10, 20, 100]
    max_depths = [2, 10, 15]

    # Store results
    results = np.zeros((len(forest_sizes), len(max_depths)))

    # Loop over different forest sizes and maximum depths
    for i, forest_size in enumerate(forest_sizes):
        for j, max_depth in enumerate(max_depths):
            # Initialize random forest object
            rfObjBreastCancer = rfc.RunRandomForestClassifier(file_loc, False, forest_size=forest_size, random_seed=0, max_depth=max_depth)
            
            # Train random forest model and get accuracy
            randomForest, accuracy = rfObjBreastCancer.run()
            results[i, j] = accuracy

            print(f"Forest Size: {forest_size}, Max Depth: {max_depth}, Accuracy: {accuracy}\n")

    # Plot the results
    plt.figure(figsize=(12, 6))
    for i, forest_size in enumerate(forest_sizes):
        plt.plot(max_depths, results[i, :], label=f'Forest Size: {forest_size}')

    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Classifier Accuracy for Different Forest Sizes and Max Depths (Breast Cancer Dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("\n\nRandom Forest Classifier on Wisconsin Breast Prognostic dataset\n")
    basic_example()
    # grid_search()