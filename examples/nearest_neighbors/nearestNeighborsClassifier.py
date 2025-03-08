import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.nearest_neighbors import *
from sega_learn.utils import make_classification
from sklearn.metrics import classification_report, accuracy_score

X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
   
# Add a categorical feature to the dataset
X_categorical = np.random.choice(['A', 'B', 'C'], size=(X.shape[0], 1))
X = np.hstack((X, X_categorical))
    
# Instantiate the KNNClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, one_hot_encode=True)

# Fit the model
knn_classifier.fit(X, y)

# Make predictions
predictions = knn_classifier.predict(X)

# Print the classification report and accuracy
print(classification_report(y, predictions))
print(f"Accuracy: {accuracy_score(y, predictions)}")

