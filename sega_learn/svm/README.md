# SVM Module

The SVM module in SEGA_LEARN provides implementations of Support Vector Machines (SVM) for classification, regression, and anomaly detection tasks. These models are designed to handle both linear and non-linear problems using various kernel functions.

## How SVMs Work

Support Vector Machines are supervised learning models that aim to find the optimal hyperplane that separates data points into different classes (for classification) or predicts continuous values (for regression). The key concepts behind SVMs include:

1. **Hyperplane**: A decision boundary that separates data points into different classes. In higher dimensions, this is referred to as a hyperplane.
2. **Support Vectors**: The data points closest to the hyperplane. These points are critical in defining the position and orientation of the hyperplane.
3. **Margin**: The distance between the hyperplane and the nearest data points (support vectors). SVMs aim to maximize this margin to improve generalization.

### Classification
For classification tasks, SVMs aim to find the hyperplane that maximizes the margin between classes. If the data is not linearly separable, SVMs use kernel functions to map the data into a higher-dimensional space where a linear hyperplane can separate the classes. This is known as the "kernel trick."

### Regression
For regression tasks, SVMs use an epsilon-insensitive loss function to fit a model that predicts continuous values. The goal is to find a function that deviates from the actual target values by at most epsilon, while minimizing the model complexity.

### Anomaly Detection
In anomaly detection, One-Class SVMs are used to identify inliers and outliers in the data. The model learns a decision boundary that encompasses the majority of the data points, treating points outside this boundary as anomalies.

## Kernel Functions

Kernel functions are a key component of SVMs, enabling them to handle non-linear problems by mapping the input data into a higher-dimensional space. The choice of kernel function depends on the nature of the data and the problem being solved.

### Supported Kernel Functions

1. **Linear Kernel**
   - Computes the dot product between input vectors.
   - Suitable for linearly separable data.
   - Formula: $K(x, y) = x \cdot y$
   - Parameters: None
   - Example: ```K(x, y) = x[0] * y[0] + x[1] * y[1] + ... + x[n] * y[n]```

2. **Polynomial Kernel**
   - Computes a polynomial transformation of the input data.
   - Useful for capturing interactions between features.
   - Formula: $K(x, y) = (\gamma \cdot x \cdot y + c_0)^{d}$
   - Parameters:
     - $\gamma$: Scaling factor for the dot product.
     - $c_0$: Independent term.
     - $d$: Degree of the polynomial.
   - Example: ```K(x, y) = (0.5 * x[0] * y[0] + 1)^2```

3. **RBF (Radial Basis Function) Kernel**
   - Also known as the Gaussian kernel.
   - Maps data into an infinite-dimensional space.
   - Effective for non-linear problems.
   - Formula: $K(x, y) = \exp(-\gamma \cdot ||x - y||^2)$
   - Parameters:
     - $\gamma$: Controls the width of the Gaussian function.
   - Example: ```K(x, y) = exp(-0.5 * ||x - y||^2)```

4. **Sigmoid Kernel**
   - Computes a sigmoid transformation of the input data.
   - Similar to a neural network activation function.
   - Formula: $K(x, y) = \tanh(\gamma \cdot x \cdot y + c_0)$
   - Parameters:
     - $\gamma$: Scaling factor for the dot product.
     - $c_0$: Independent term.
   - Example: ```K(x, y) = tanh(0.5 * x[0] * y[0] + 1)```

### Choosing a Kernel
- **Linear Kernel**: Use when the data is linearly separable or when the number of features is very large compared to the number of samples.
- **Polynomial Kernel**: Use when feature interactions are important and the data is not linearly separable.
- **RBF Kernel**: Use for most non-linear problems, as it is a versatile and widely used kernel.
- **Sigmoid Kernel**: Use when the data resembles the behavior of a neural network. (Less common in practice.)

### Kernel Trick
The kernel trick allows SVMs to compute the dot product in the higher-dimensional space without explicitly transforming the data. This reduces computational complexity and enables SVMs to handle high-dimensional data efficiently.

## Algorithms

### Linear SVM Classifier
A linear SVM classifier is used for binary and multi-class classification tasks. It uses a linear decision boundary to separate classes.

#### Usage
```python
from sega_learn.svm import LinearSVC

# Initialize the LinearSVC object
linear_svc = LinearSVC(C=1.0, max_iter=1000)

# Fit the model
linear_svc.fit(X, y)

# Predict class labels for new data
labels = linear_svc.predict(new_X)
```

### Linear SVM Regressor
A linear SVM regressor is used for regression tasks. It uses a linear decision boundary to predict continuous target values.

#### Usage
```python
from sega_learn.svm import LinearSVR

# Initialize the LinearSVR object
linear_svr = LinearSVR(C=1.0, max_iter=1000)

# Fit the model
linear_svr.fit(X, y)

# Predict target values for new data
predictions = linear_svr.predict(new_X)
```

### Generalized SVM Classifier
A generalized SVM classifier supports multiple kernels for binary and multi-class classification tasks.

#### Usage
```python
from sega_learn.svm import GeneralizedSVC

# Initialize the GeneralizedSVC object
generalized_svc = GeneralizedSVC(kernel="rbf", C=1.0, gamma="scale")

# Fit the model
generalized_svc.fit(X, y)

# Predict class labels for new data
labels = generalized_svc.predict(new_X)
```

### Generalized SVM Regressor
A generalized SVM regressor supports multiple kernels for regression tasks.

#### Usage
```python
from sega_learn.svm import GeneralizedSVR

# Initialize the GeneralizedSVR object
generalized_svr = GeneralizedSVR(kernel="poly", C=1.0, degree=3)

# Fit the model
generalized_svr.fit(X, y)

# Predict target values for new data
predictions = generalized_svr.predict(new_X)
```

### One-Class SVM
A one-class SVM is used for anomaly detection by identifying inliers and outliers in the data.

#### Usage
```python
from sega_learn.svm import OneClassSVM

# Initialize the OneClassSVM object
one_class_svm = OneClassSVM(kernel="rbf", C=1.0, gamma="scale")

# Fit the model
one_class_svm.fit(X)

# Predict whether samples are inliers (1) or outliers (-1)
predictions = one_class_svm.predict(new_X)
```

## Examples

### Linear SVM Classifier Example
```python
from sega_learn.svm import LinearSVC
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, size=100)

# Initialize and fit LinearSVC
linear_svc = LinearSVC(C=1.0, max_iter=1000)
linear_svc.fit(X, y)

# Predict class labels
labels = linear_svc.predict(X)

# Print class labels
print(labels)
```

### Generalized SVM Regressor Example
```python
from sega_learn.svm import GeneralizedSVR
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.rand(100)

# Initialize and fit GeneralizedSVR
generalized_svr = GeneralizedSVR(kernel="rbf", C=1.0, gamma="scale")
generalized_svr.fit(X, y)

# Predict target values
predictions = generalized_svr.predict(X)

# Print predictions
print(predictions)
```

### One-Class SVM Example
```python
from sega_learn.svm import OneClassSVM
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)

# Initialize and fit OneClassSVM
one_class_svm = OneClassSVM(kernel="rbf", C=1.0, gamma="scale")
one_class_svm.fit(X)

# Predict inliers and outliers
predictions = one_class_svm.predict(X)

# Print predictions
print(predictions)
```
