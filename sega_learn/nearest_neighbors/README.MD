# Nearest Neighbors Module

The `nearest_neighbors` module implements the Nearest Neighbors algorithm for classification and regression tasks. 
It implements the K-Nearest Neighbors (KNN) algorithm, which is a simple yet effective method for making predictions based on the proximity of data points in feature space.
It has classes for both classification and regression tasks, allowing users to choose the appropriate method based on their needs.

## KNeighborsBase

The `KNeighborsBase` class is an abstract base class that provides common functionality for both `KNeighborsClassifier` and `KNeighborsRegressor`.

### Features

- **Initialization**: Allows setting the number of neighbors, distance metric, one-hot encoding, and floating point precision.
- **Fit Method**: Stores the training data and target values.
- **Data Precision**: Ensures the input data is in the specified floating point precision.
- **Data Validation**: Checks if the input data is valid.
- **One-Hot Encoding**: Applies one-hot encoding to categorical columns if specified.
- **Distance Computation**: Computes distances using Euclidean, Manhattan, or Minkowski metrics.


The `KNeighborsBase` class is not used directly. Instead, it is subclassed by `KNeighborsClassifier` and `KNeighborsRegressor` to implement specific prediction methods.


## KNeighborsClassifier

The `KNeighborsClassifier` class implements the K-Nearest Neighbors algorithm for classification tasks.

### Usage

```python
from sega_learn.nearest_neighbors import KNeighborsClassifier

# Initialize the classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3, distance_metric='euclidean', one_hot_encode=False)

# Fit the model
knn_classifier.fit(X_train, y_train)

# Predict class labels for new data
predictions = knn_classifier.predict(X_test)
```

## KNeighborsRegressor

The `KNeighborsRegressor` class implements the K-Nearest Neighbors algorithm for regression tasks.

### Usage

```python
from sega_learn.nearest_neighbors import KNeighborsRegressor

# Initialize the regressor
knn_regressor = KNeighborsRegressor(n_neighbors=3, distance_metric='euclidean', one_hot_encode=False)

# Fit the model
knn_regressor.fit(X_train, y_train)

# Predict values for new data
predictions = knn_regressor.predict(X_test)
```

