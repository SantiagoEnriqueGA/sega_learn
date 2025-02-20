# Utilities Module

The utils module in SEGA_LEARN provides various utility functions and classes for data preparation, polynomial transformation, model selection, and evaluation metrics. These utilities are designed to facilitate the machine learning workflow.

## Modules

### Data Preparation
The `dataPrep` module provides functions for preparing data for machine learning models, including one-hot encoding, writing data to CSV files, and splitting data into k folds for cross-validation.

#### Usage
```python
from sega_learn.utils.dataPrep import DataPrep

# One-hot encode specified columns
df = DataPrep.one_hot_encode(df, cols)

# Write DataFrame to CSV file
DataPrep.write_data(df, csv_file, print_path=True)

# Prepare data by loading a CSV file and one-hot encoding specified columns
df, prepared_csv_file = DataPrep.prepare_data(csv_file, label_col_index, cols_to_encode, write_to_csv=True)

# Convert DataFrame to NumPy arrays
X, y = DataPrep.df_to_ndarray(df, y_col=0)

# Split data into k folds for cross-validation
X_folds, y_folds = DataPrep.k_split(X, y, k=5)
```

### Polynomial Transformation
The `polynomialTransform` module provides a class for polynomial feature transformation, which creates new features by raising existing features to a power or creating interaction terms.

#### Formula
The polynomial transformation creates new features by raising existing features to a power or creating interaction terms. For example, given features $ x_1 $ and $ x_2 $, the polynomial transformation of degree 2 would create the following features:

```math
[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]
```

Where:
- $` x_1 `$ and $` x_2 `$ are the original features
- $` x_1^2 `$ is the square of feature $` x_1 `$
- $` x_1 x_2 `$ is the interaction term between features $` x_1 `$ and $` x_2 `$

#### Usage
```python
from sega_learn.utils.polynomialTransform import PolynomialTransform

# Initialize the PolynomialTransform object
poly_transform = PolynomialTransform(degree=3)

# Fit the model to the data
poly_transform.fit(X)

# Transform the data into polynomial features
X_poly = poly_transform.transform(X)

# Fit and transform the data
X_poly = poly_transform.fit_transform(X)
```

### Model Selection
The `model_selection` module provides classes for hyperparameter tuning using grid search, random search, and a custom search method.

#### Usage
```python
from sega_learn.utils.model_selection import GridSearchCV, RandomSearchCV, segaSearchCV

# Initialize the GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, metric='mse', direction='minimize')

# Fit the model to the data
best_model = grid_search.fit(X, y, verbose=True)

# Initialize the RandomSearchCV object
random_search = RandomSearchCV(model, param_grid, iter=10, cv=5, metric='mse', direction='minimize')

# Fit the model to the data
best_model = random_search.fit(X, y, verbose=True)

# Initialize the segaSearchCV object
sega_search = segaSearchCV(model, param_space, iter=10, cv=5, metric='mse', direction='minimize')

# Fit the model to the data
best_model = sega_search.fit(X, y, verbose=True)
```

### Data Augmentation
The `dataAugmentation` module provides a class for imbalanced classification tasks using SMOTE (Synthetic Minority Over-sampling Technique), Under-sampling, Over-sampling, and/or a combination of each. 

#### SMOTE
SMOTE is a technique used to create synthetic samples for the minority class in imbalanced datasets. It works by selecting a minority class sample and creating new samples along the line segments connecting it to its nearest neighbors. In this way, it generates new samples that are similar to the existing minority class samples, helping to balance the dataset.

#### Usage
```python
from sega_learn.utils.dataAugmentation import SMOTE

# Create SMOTE object
smote = SMOTE(k_neighbors=5, random_state=42)

# Fit and resample the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### Over-Sampling
Over-sampling is a technique used to increase the number of samples in the minority class by duplicating existing samples or generating new samples. This can help to balance the dataset and improve the performance of machine learning models.
#### Usage
```python
from sega_learn.utils.dataAugmentation import OverSampling

# Create OverSampling object
over_sampling = OverSampling(random_state=42)

# Fit and resample the training data
X_resampled, y_resampled = over_sampling.fit_resample(X_train, y_train)
```

#### Under-Sampling
Under-sampling is a technique used to reduce the number of samples in the majority class by randomly removing samples. This can help to balance the dataset and improve the performance of machine learning models.
#### Usage
```python
from sega_learn.utils.dataAugmentation import UnderSampling

# Create UnderSampling object
under_sampling = UnderSampling(random_state=42)

# Fit and resample the training data
X_resampled, y_resampled = under_sampling.fit_resample(X_train, y_train)
```

#### Augmentor for Combination of Techniques
The `Augmentor` class allows you to combine different data augmentation techniques, such as SMOTE, Over-sampling, and Under-sampling, to create a balanced dataset. Simply initialize the `Augmentor` class with the desired techniques and their parameters, and then call the `fit_resample` method to apply the augmentation.
```python
from sega_learn.utils.dataAugmentation import Augmentor, SMOTE, OverSampling, UnderSampling

# Create Augmentor object with desired techniques
augmentor = Augmentor(
    techniques=[
        SMOTE(k_neighbors=5, random_state=42),
        OverSampling(random_state=42),
        UnderSampling(random_state=42)
    ]
)

# Fit and resample the training data
X_resampled, y_resampled = augmentor.augment(X_train, y_train)
```

### Evaluation Metrics
The `metrics` module provides various evaluation metrics for regression and classification tasks.

#### Formulas
- **Mean Squared Error (MSE)**:  Measures the average squared difference between the actual and predicted values. It is suitable for regression tasks where larger errors are more significant, as it penalizes larger errors more heavily.
```math
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

- **R-squared (R²)**: indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. It is useful for evaluating the goodness of fit in regression models, showing how well the model explains the variability of the response data.
```math
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
```

- **Mean Absolute Error (MAE)**: measures the average absolute difference between the actual and predicted values. It is suitable for regression tasks where all errors are equally weighted, providing a straightforward interpretation of the average error magnitude.
```math
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
```

- **Root Mean Squared Error (RMSE)**: measures the square root of the average squared difference between the actual and predicted values. It is suitable for regression tasks where larger errors are more significant and need to be penalized, providing a more sensitive measure than MAE.
```math
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
```

- **Mean Absolute Percentage Error (MAPE)**: measures the average absolute percentage difference between the actual and predicted values. It is useful for regression tasks where the relative error is more important than the absolute error, providing a percentage-based error measure.
```math
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
```

- **Mean Percentage Error (MPE)**: measures the average percentage difference between the actual and predicted values. It is useful for regression tasks where the direction of the error (overestimation or underestimation) is important, providing insight into the bias of the predictions.
```math
\text{MPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{y_i - \hat{y}_i}{y_i}
```

- **Accuracy**: measures the proportion of correct predictions out of the total predictions. It is suitable for classification tasks with balanced classes, providing a straightforward measure of overall prediction correctness.
```math
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
```

- **Precision**: measures the proportion of true positive predictions out of the total positive predictions. It is useful for classification tasks where the cost of false positives is high, indicating the accuracy of positive predictions.
```math
\text{Precision} = \frac{TP}{TP + FP}
```

- **Recall**: measures the proportion of true positive predictions out of the total actual positives. It is useful for classification tasks where the cost of false negatives is high, indicating the model's ability to capture all positive instances.
```math
\text{Recall} = \frac{TP}{TP + FN}
```

- **F1 Score**: the harmonic mean of precision and recall, providing a balance between the two. It is suitable for classification tasks with imbalanced classes, offering a single measure that considers both precision and recall.
```math
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

- **Log Loss**: measures the performance of a classification model where the output is a probability value between 0 and 1. It is useful for classification tasks where the predicted probabilities are important, penalizing incorrect predictions more heavily.
```math
\text{Log Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)
```

#### Usage
```python
from sega_learn.utils.metrics import Metrics

# Regression Metrics
mse = Metrics.mean_squared_error(y_true, y_pred)
r2 = Metrics.r_squared(y_true, y_pred)
mae = Metrics.mean_absolute_error(y_true, y_pred)
rmse = Metrics.root_mean_squared_error(y_true, y_pred)
mape = Metrics.mean_absolute_percentage_error(y_true, y_pred)
mpe = Metrics.mean_percentage_error(y_true, y_pred)

# Classification Metrics
accuracy = Metrics.accuracy(y_true, y_pred)
precision = Metrics.precision(y_true, y_pred)
recall = Metrics.recall(y_true, y_pred)
f1 = Metrics.f1_score(y_true, y_pred)
log_loss = Metrics.log_loss(y_true, y_pred)

# Additional Metrics
confusion_matrix = Metrics.confusion_matrix(y_true, y_pred)
Metrics.show_confusion_matrix(y_true, y_pred)
classification_report = Metrics.classification_report(y_true, y_pred)
Metrics.show_classification_report(y_true, y_pred)
```
