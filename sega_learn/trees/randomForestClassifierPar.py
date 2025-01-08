"""
This module contains the implementation of a Random Forest Classifier.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- RandomForestWithInfoGain: A class representing a Random Forest model that returns information gain for vis.
- runRandomForest: A class that runs the Random Forest algorithm.
"""

# Importing the required libraries
import csv
import numpy as np
import ast
from datetime import datetime
from math import log, floor, ceil
import random
import matplotlib.pyplot as plt
import multiprocessing

from .treeClassifier import ClassifierTreeUtility, ClassifierTree, ClassifierTreeInfoGain
from .randomForestClassifier import RandomForestClassifier

class RunRandomForestClassifierPar(object):
    """
    A class that represents a random forest algorithm.

    Attributes:
        random_seed (int): The random seed for reproducibility.
        forest_size (int): The number of trees in the random forest.
        max_depth (int): The maximum depth of each decision tree in the random forest.
        display (bool): Whether to display additional information about info gain.
        X (list): The list of data features.
        y (list): The list of data labels.
        XX (list): The list that contains both data features and data labels.
        numerical_cols (int): The number of numeric attributes (columns).

    Methods:
        __init__(self, file_loc, display=False, forest_size=5, random_seed=0, max_depth=10):
            Initializes the random forest object.

        reset(self):
            Resets the random forest object.

        run(self):
            Runs the random forest algorithm.

    Example:
        randomForest, accuracy = runRandomForest('data.csv', display=True, forest_size=10, random_seed=42)
    """
    # Initialize class variables
    random_seed = 0     # Random seed for reproducibility
    forest_size = 10    # Number of trees in the random forest
    max_depth = 10      # Maximum depth of each decision tree
    display = False     # Flag to display additional information about info gain
    
    X = list()          # Data features
    y = list()          # Data labels
    XX = list()         # Contains both data features and data labels
    numerical_cols = 0  # Number of numeric attributes (columns)

    def __init__(self, file_loc, display=False, forest_size=5, random_seed=0, max_depth=10):
        """
        Initializes the random forest object.

        Args:
            file_loc (str): The file location of the dataset.
            display (bool, optional): Whether to display additional information about info gain. Defaults to False.
            forest_size (int, optional): The number of trees in the random forest. Defaults to 5.
            random_seed (int, optional): The random seed for reproducibility. Defaults to 0.
            max_depth (int, optional): The maximum depth of each decision tree in the random forest. Defaults to 10.
        """
        self.reset()    # Reset the random forest object

        self.random_seed = random_seed  # Set the random seed for reproducibility
        np.random.seed(random_seed)     # Set the random seed for NumPy

        self.forest_size = forest_size  # Set the number of trees in the random forest
        self.max_depth = max_depth      # Set the maximum depth of each decision tree
        self.display = display          # Set the flag to display additional information about info gain
        
        self.numerical_cols = set()         # Initialize the set of indices of numeric attributes (columns)
        with open(file_loc, 'r') as f:      # Open the file in read mode
            reader = csv.reader(f)          # Create a CSV reader
            headers = next(reader)          # Get the headers of the CSV file
            for i in range(len(headers)):   # Loop over the indices of the headers
                try:
                    float(next(reader)[i])      # If successful, add the index to the set of numerical columns
                    self.numerical_cols.add(i)  # Add the index to the set of numerical columns
                except ValueError:
                    continue

        print("reading the data")
        try:
            with open(file_loc) as f:                       # Open the file
                next(f, None)                               # Skip the header
                for line in csv.reader(f, delimiter=","):   # Read the file line by line
                    xline = []                  
                    for i in range(len(line)):              # Loop over the indices of the line
                        if i in self.numerical_cols:                # If the index is in the set of numerical columns
                            xline.append(ast.literal_eval(line[i])) # Append the value to the input data features
                        
                        else:                                       # If the index is not in the set of numerical columns
                            xline.append(line[i])                   # Append the value to the input data features    

                    self.X.append(xline[:-1])   # Append the input data features to the list of input data features
                    self.y.append(xline[-1])    # Append the target value to the list of target values
                    self.XX.append(xline[:])    # Append the input data features and target value to the list of input data features and target values
        except FileNotFoundError:
            print(f"File {file_loc} not found.")
            return None, None
        
    def reset(self):
        """
        Resets the random forest object.
        """
        # Reset the random forest object
        self.random_seed = 0
        self.forest_size = 10
        self.max_depth = 10
        self.display = False
        self.X = list()
        self.y = list()
        self.XX = list()
        self.numerical_cols = 0

    def run(self):
        """
        Runs the random forest algorithm.

        Returns:
            tuple: A tuple containing the random forest object and the accuracy of the random forest algorithm.

        Raises:
            FileNotFoundError: If the file specified by file_loc does not exist.

        Notes:
            - The file should have the following format:
                - Each row represents a data point (record).
                - The last column represents the class label.
                - The remaining columns represent the features (attributes).
                - Features are numerical and class labels are binary (0 or 1).
            - The random seed is used to initialize the random number generator for reproducibility.
            - The random forest object contains the trained random forest model.
            - The accuracy is calculated as the ratio of correctly predicted labels to the total number of labels.
        """
        start = datetime.now()  # Start time

        randomForest = RandomForestClassifier(self.forest_size,self.max_depth)    # Initialize the random forest object

        print("creating the bootstrap datasets")
        randomForest.bootstrapping(self.XX)         # Create the bootstrapped datasets

        print("fitting the forest")
        randomForest.fitting()                      # Fit the decision trees to the bootstrapped datasets
        y_predicted = randomForest.voting(self.X)   # Perform voting to classify the input records

        results = [prediction == truth for prediction, truth in zip(y_predicted, self.y)]   # Compare the predicted labels with the true labels

        accuracy = float(results.count(True)) / float(len(results)) # Calculate the accuracy
        
        # Display the results
        print("accuracy: %.4f" % accuracy)
        print("OOB estimate: %.4f" % (1 - accuracy))
        print("Execution time: " + str(datetime.now() - start))

        return randomForest,accuracy

