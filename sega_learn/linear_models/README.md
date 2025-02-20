# Linear Models Module

The linear models module in SEGA_LEARN provides implementations of popular linear regression algorithms such as Ordinary Least Squares (OLS), Ridge Regression, Lasso Regression, Bayesian Regression, and RANSAC. These algorithms are designed to model the relationship between a dependent variable and one or more independent variables.

## Algorithms

### Ordinary Least Squares (OLS)
OLS is a linear regression algorithm that minimizes the sum of squared residuals between the observed and predicted values.

#### Algorithm
1. Add a column of ones to the input data if `fit_intercept` is True.
2. Compute the coefficients using the normal equation: $ w = (X^T X)^{-1} X^T y $.
3. If `fit_intercept` is True, separate the intercept from the coefficients.

#### Formula
The formula for OLS is:
$` y = Xw + \epsilon `$
where:
- $` y `$ is the dependent variable,
- $` X `$ is the matrix of independent variables,
- $` w `$ is the vector of coefficients,
- $` \epsilon `$ is the error term.

#### Usage
```python
from sega_learn.linear_models import OrdinaryLeastSquares

# Initialize the OLS object
ols = OrdinaryLeastSquares(fit_intercept=True)

# Fit the model
ols.fit(X, y)

# Predict values for new data
predictions = ols.predict(new_X)

# Get the formula of the model
formula = ols.get_formula()
```

### Ridge Regression
Ridge Regression is a linear regression algorithm that includes an L2 regularization term to prevent overfitting.

#### Algorithm
1. Add a column of ones to the input data if `fit_intercept` is True.
2. Initialize the coefficients to zeros.
3. Update the coefficients using coordinate descent until convergence or the maximum number of iterations is reached.

#### Formula
The formula for Ridge Regression is:
$` w = (X^T X + \alpha I)^{-1} X^T y `$
where:
- $` \alpha `$ is the regularization parameter,
- $` I `$ is the identity matrix.

#### Usage
```python
from sega_learn.linear_models import Ridge

# Initialize the Ridge object
ridge = Ridge(alpha=1.0, fit_intercept=True, max_iter=10000, tol=1e-4)

# Fit the model
ridge.fit(X, y)

# Predict values for new data
predictions = ridge.predict(new_X)

# Get the formula of the model
formula = ridge.get_formula()
```

### Lasso Regression
Lasso Regression is a linear regression algorithm that includes an L1 regularization term to prevent overfitting.

#### Algorithm
1. Add a column of ones to the input data if `fit_intercept` is True.
2. Initialize the coefficients to zeros.
3. Update the coefficients using coordinate descent until convergence or the maximum number of iterations is reached.

#### Formula
The formula for Lasso Regression is:
$` \min_w \left( \frac{1}{2n} \sum_{i=1}^n (y_i - X_i w)^2 + \alpha \|w\|_1 \right) `$
where:
- $` \|w\|_1 `$ is the L1 norm of the coefficients.

#### Usage
```python
from sega_learn.linear_models import Lasso

# Initialize the Lasso object
lasso = Lasso(alpha=1.0, fit_intercept=True, max_iter=10000, tol=1e-4)

# Fit the model
lasso.fit(X, y)

# Predict values for new data
predictions = lasso.predict(new_X)

# Get the formula of the model
formula = lasso.get_formula()
```

### Bayesian Regression
Bayesian Regression is a linear regression algorithm that includes both L1 and L2 regularization terms and uses Bayesian inference to estimate the coefficients.

#### Algorithm
1. Initialize the parameters.
2. Update the coefficients using coordinate descent until convergence or the maximum number of iterations is reached.
3. Optionally, tune the hyperparameters using the ADAM optimizer.

#### Formula
The formula for Bayesian Regression is:
$` p(w|X,y) = \frac{p(y|X,w) p(w)}{p(y|X)} `$
where:
- $` p(w|X,y) $` is the posterior distribution of the coefficients,
- $` p(y|X,w) $` is the likelihood,
- $` p(w) $` is the prior distribution,
- $` p(y|X) $` is the marginal likelihood.

#### Usage
```python
from sega_learn.linear_models import Bayesian

# Initialize the Bayesian object
bayesian = Bayesian(max_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True)

# Fit the model
bayesian.fit(X, y)

# Predict values for new data
predictions = bayesian.predict(new_X)

# Get the formula of the model
formula = bayesian.get_formula()

# Tune the hyperparameters
best_alpha_1, best_alpha_2, best_lambda_1, best_lambda_2 = bayesian.tune(X, y)
```

### RANSAC
RANSAC (RANdom SAmple Consensus) is a robust linear regression algorithm that fits a model to the data while ignoring outliers.

#### Algorithm
1. Randomly select a subset of the data points.
2. Fit the model to the subset.
3. Compute the residuals for all data points.
4. Identify the inliers and update the best model if the current model has a lower error.
5. Optionally, scale the threshold or the number of data points until a model is fit.

#### Formula
The formula for RANSAC is iterative and does not have a closed-form solution. It involves repeatedly fitting a model to random subsets of the data and selecting the model with the best fit.

#### Usage
```python
from sega_learn.linear_models import RANSAC

# Initialize the RANSAC object
ransac = RANSAC(n=10, k=100, t=0.05, d=10, model=None, auto_scale_t=False, scale_t_factor=2, auto_scale_n=False, scale_n_factor=2)

# Fit the model
ransac.fit(X, y)

# Predict values for new data
predictions = ransac.predict(new_X)

# Get the formula of the model
formula = ransac.get_formula()
```

### Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis (LDA) is a classification algorithm that finds a linear combination of features that best separates two or more classes.

#### Algorithm
1. Compute the mean and covariance of each class.
2. Compute the within-class and between-class scatter matrices.
3. Solve the generalized eigenvalue problem to find the linear discriminants.
4. Project the data onto the linear discriminants.

#### Formula
The formula for LDA is:
$` w = \Sigma^{-1} (\mu_1 - \mu_2) `$
where:
- $` \Sigma `$ is the pooled covariance matrix,
- $` \mu_1 $` and $` \mu_2 `$ are the means of the two classes.

#### Usage
```python
from sega_learn.linear_models import LinearDiscriminantAnalysis

# Initialize the LDA object
lda = LinearDiscriminantAnalysis(solver='svd')

# Fit the model
lda.fit(X, y)

# Predict class labels for new data
predictions = lda.predict(new_X)
```

### Quadratic Discriminant Analysis (QDA)
Quadratic Discriminant Analysis (QDA) is a classification algorithm that finds a quadratic combination of features that best separates two or more classes.

#### Algorithm
1. Compute the mean and covariance of each class.
2. Compute the log-likelihood of each class for the input samples.
3. Assign each sample to the class with the highest log-likelihood.

#### Formula
The formula for QDA is:
$` \delta_k(x) = -\frac{1}{2} \log |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k `$
where:
- $` \Sigma_k `$ is the covariance matrix of class $` k `$,
- $` \mu_k `$ is the mean of class $` k `$,
- $` \pi_k `$ is the prior probability of class $` k `$.

#### Usage
```python
from sega_learn.linear_models import QuadraticDiscriminantAnalysis

# Initialize the QDA object
qda = QuadraticDiscriminantAnalysis()

# Fit the model
qda.fit(X, y)

# Predict class labels for new data
predictions = qda.predict(new_X)
```

## Examples

### OLS Example
```python
from sega_learn.linear_models import OrdinaryLeastSquares
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# Initialize and fit OLS
ols = OrdinaryLeastSquares(fit_intercept=True)
ols.fit(X, y)

# Predict values
predictions = ols.predict(X)

# Print the formula
print(ols.get_formula())
```

### Ridge Regression Example
```python
from sega_learn.linear_models import Ridge
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# Initialize and fit Ridge Regression
ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(X, y)

# Predict values
predictions = ridge.predict(X)

# Print the formula
print(ridge.get_formula())
```

### Lasso Regression Example
```python
from sega_learn.linear_models import Lasso
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# Initialize and fit Lasso Regression
lasso = Lasso(alpha=1.0, fit_intercept=True)
lasso.fit(X, y)

# Predict values
predictions = lasso.predict(X)

# Print the formula
print(lasso.get_formula())
```

### Bayesian Regression Example
```python
from sega_learn.linear_models import Bayesian
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# Initialize and fit Bayesian Regression
bayesian = Bayesian(max_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True)
bayesian.fit(X, y)

# Predict values
predictions = bayesian.predict(X)

# Print the formula
print(bayesian.get_formula())

# Tune the hyperparameters
best_alpha_1, best_alpha_2, best_lambda_1, best_lambda_2 = bayesian.tune(X, y)
```

### RANSAC Example
```python
from sega_learn.linear_models import RANSAC
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# Initialize and fit RANSAC
ransac = RANSAC(n=10, k=100, t=0.05, d=10, model=None, auto_scale_t=False, scale_t_factor=2, auto_scale_n=False, scale_n_factor=2)
ransac.fit(X, y)

# Predict values
predictions = ransac.predict(X)

# Print the formula
print(ransac.get_formula())
```

### LDA Example
```python
from sega_learn.linear_models import LinearDiscriminantAnalysis
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

# Initialize and fit LDA
lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(X, y)

# Predict class labels
predictions = lda.predict(X)

# Print predictions
print(predictions)
```

### QDA Example
```python
from sega_learn.linear_models import QuadraticDiscriminantAnalysis
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

# Initialize and fit QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)

# Predict class labels
predictions = qda.predict(X)

# Print predictions
print(predictions)
```
