# Trees Module

The trees module in SEGA_LEARN provides implementations of decision tree algorithms for both classification and regression tasks. 
These algorithms are designed to create tree-based models that can be used for predicting target values based on input features.
Trees are a popular choice for machine learning tasks due to their interpretability and ability to handle both numerical and categorical data.

## Classifier vs Regressor Trees
Both classifier and regressor trees are based on the same underlying decision tree structure, but they differ in their objectives:
- **Classifier Trees**: Aim to predict categorical class labels. They use metrics like information gain or Gini impurity to determine the best splits.
- **Regressor Trees**: Aim to predict continuous target values. They use metrics like variance reduction or mean squared error to determine the best splits.

A key difference is in the way they handle the target variable:
- Classifier trees use categorical labels and focus on maximizing the purity of the resulting subsets.
- Regressor trees use continuous values and focus on minimizing the variance of the resulting subsets.

### Difference in How Trees are Split
- **Classifier Trees**: Use entropy to measure the quality of a split. The best split is the one that maximizes the information.
    - **Entropy**: Measures the impurity of a dataset. Lower entropy indicates higher purity.
        - $\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$ 
        - where $S$ is the dataset, $c$ is the number of classes, and $p_i$ is the proportion of instances in class $i$.
  - **Information Gain**: The reduction in entropy after a dataset is split on an attribute.


- **Regressor Trees**: Use variance reduction to measure the quality of a split. The best split is the one that minimizes the variance.
  - **Variance**: Measures the spread of the target values. Lower variance indicates that the target values are closer to the mean.
    - $\text{Variance}(S) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$
    - where $S$ is the dataset, $n$ is the number of instances, $x_i$ is the target value of instance $i$, and $\bar{x}$ is the mean of the target values.
  - **Variance Reduction**: The reduction in variance after a dataset is split on an attribute.
    - $\text{Variance Reduction} = \text{Variance}(S) - \left( \frac{N_L}{N} \cdot \text{Variance}(S_L) + \frac{N_R}{N} \cdot \text{Variance}(S_R) \right)$
    - where $S$ is the dataset, $N$ is the total number of instances, $N_L$ and $N_R$ are the number of instances in the left and right subsets, and $S_L$ and $S_R$ are the left and right subsets after the split.


## Gradient Boosted Trees
Gradient Boosted Trees are an ensemble learning method that builds multiple decision trees sequentially. 
By combining the predictions of multiple weak learners (single trees), it creates a strong predictive model.
Each tree tries to correct the errors of the previous tree, resulting in a strong predictive model. This method is particularly effective for regression tasks.

- **Residuals**: The difference between the actual target values and the predicted values from the current model.
  - $r_i^{(m)} = y_i - F_{m-1}(x_i)$
  - where $r_i^{(m)}$ is the residual for the $i$-th instance at the $m$-th iteration, $y_i$ is the actual target value, and $F_{m-1}(x_i)$ is the predicted value from the model at the $(m-1)$-th iteration.

- **Model Update**: The model is updated by adding the predictions of the new tree to the current model.
  - $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$
  - where $F_m(x)$ is the updated model at the $m$-th iteration, $\eta$ is the learning rate, and $h_m(x)$ is the prediction from the new tree at the $m$-th iteration.


## Algorithms

### Decision Tree Classifier
A decision tree classifier is a tree-based model used for classification tasks. It splits the data into subsets based on the value of input features, aiming to maximize the information gain at each split.

#### Algorithm
1. Calculate the entropy of the target variable.
2. For each feature, calculate the information gain for all possible splits.
3. Choose the feature and split that maximize the information gain.
4. Split the data into subsets based on the chosen feature and split value.
5. Repeat steps 1-4 recursively for each subset until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf).

#### Usage
```python
from sega_learn.trees import ClassifierTree

# Initialize the ClassifierTree object
classifier_tree = ClassifierTree(max_depth=5)

# Fit the model
classifier_tree.learn(X, y)

# Predict class labels for new data
labels = classifier_tree.classify(new_X)
```

### Decision Tree Regressor
A decision tree regressor is a tree-based model used for regression tasks. It splits the data into subsets based on the value of input features, aiming to minimize the variance of the target variable at each split.

#### Algorithm
1. Calculate the variance of the target variable.
2. For each feature, calculate the reduction in variance for all possible splits.
3. Choose the feature and split that maximize the reduction in variance.
4. Split the data into subsets based on the chosen feature and split value.
5. Repeat steps 1-4 recursively for each subset until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf).

#### Usage
```python
from sega_learn.trees import RegressorTree

# Initialize the RegressorTree object
regressor_tree = RegressorTree(max_depth=5)

# Fit the model
regressor_tree.learn(X, y)

# Predict target values for new data
predictions = regressor_tree.predict(new_X)
```

### Random Forest Classifier
A random forest classifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks.

#### Algorithm
1. Create multiple bootstrapped datasets from the original dataset.
2. Train a decision tree on each bootstrapped dataset.
3. For each tree, use a random subset of features to determine the best split.
4. Aggregate the predictions from all trees to determine the final class label.

#### Usage
```python
from sega_learn.trees import RandomForestClassifier

# Initialize the RandomForestClassifier object
rf_classifier = RandomForestClassifier(forest_size=10, max_depth=5)

# Fit the model
rf_classifier.fit(X, y)

# Predict class labels for new data
labels = rf_classifier.predict(new_X)
```

### Random Forest Regressor
A random forest regressor is an ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction for regression tasks.

#### Algorithm
1. Create multiple bootstrapped datasets from the original dataset.
2. Train a decision tree on each bootstrapped dataset.
3. For each tree, use a random subset of features to determine the best split.
4. Aggregate the predictions from all trees to determine the final target value.

#### Usage
```python
from sega_learn.trees import RandomForestRegressor

# Initialize the RandomForestRegressor object
rf_regressor = RandomForestRegressor(forest_size=10, max_depth=5)

# Fit the model
rf_regressor.fit(X, y)

# Predict target values for new data
predictions = rf_regressor.predict(new_X)
```

### Gradient Boosted Regressor
A gradient boosted regressor is an ensemble learning method that builds multiple decision trees sequentially, where each tree tries to correct the errors of the previous tree.

#### Algorithm
1. Initialize the model with a constant value (e.g., the mean of the target values).
2. For each iteration:
   - Compute the residuals (errors) of the current model.
   - Train a decision tree on the residuals.
   - Update the model by adding the predictions of the new tree to the current model.
3. Repeat step 2 for a specified number of iterations or until convergence.

#### Usage
```python
from sega_learn.trees import GradientBoostedRegressor

# Initialize the GradientBoostedRegressor object
gb_regressor = GradientBoostedRegressor(num_trees=10, max_depth=5)

# Fit the model
gb_regressor.fit(X, y)

# Predict target values for new data
predictions = gb_regressor.predict(new_X)
```

## Examples

### Decision Tree Classifier Example
```python
from sega_learn.trees import ClassifierTree
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, size=100)

# Initialize and fit ClassifierTree
classifier_tree = ClassifierTree(max_depth=5)
classifier_tree.learn(X, y)

# Predict class labels
labels = classifier_tree.classify(X)

# Print class labels
print(labels)
```

### Decision Tree Regressor Example
```python
from sega_learn.trees import RegressorTree
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.rand(100)

# Initialize and fit RegressorTree
regressor_tree = RegressorTree(max_depth=5)
regressor_tree.learn(X, y)

# Predict target values
predictions = regressor_tree.predict(X)

# Print predictions
print(predictions)
```

### Random Forest Classifier Example
```python
from sega_learn.trees import RandomForestClassifier
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, size=100)

# Initialize and fit RandomForestClassifier
rf_classifier = RandomForestClassifier(forest_size=10, max_depth=5)
rf_classifier.fit(X, y)

# Predict class labels
labels = rf_classifier.predict(X)

# Print class labels
print(labels)
```

### Random Forest Regressor Example
```python
from sega_learn.trees import RandomForestRegressor
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.rand(100)

# Initialize and fit RandomForestRegressor
rf_regressor = RandomForestRegressor(forest_size=10, max_depth=5)
rf_regressor.fit(X, y)

# Predict target values
predictions = rf_regressor.predict(X)

# Print predictions
print(predictions)
```

### Gradient Boosted Regressor Example
```python
from sega_learn.trees import GradientBoostedRegressor
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.rand(100)

# Initialize and fit GradientBoostedRegressor
gb_regressor = GradientBoostedRegressor(num_trees=10, max_depth=5)
gb_regressor.fit(X, y)

# Predict target values
predictions = gb_regressor.predict(X)

# Print predictions
print(predictions)
```
