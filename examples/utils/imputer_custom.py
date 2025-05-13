import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import LogisticRegression, OrdinaryLeastSquares
from sega_learn.trees import RandomForestClassifier, RandomForestRegressor
from sega_learn.utils.imputation import CustomImputer

# Example Data
X_num = np.random.rand(10, 2) * 10
X_cat_val = np.random.randint(0, 3, size=(10, 1))  # 3 categories
X_cat = np.array(["A", "B", "C"])[X_cat_val].reshape(-1, 1)
X = np.hstack((X_num, X_cat)).astype(object)
# Introduce NaNs
X[0, 0] = np.nan
X[2, 1] = np.nan
X[4, 2] = np.nan
X[7, 0] = np.nan
X[8, 2] = np.nan
print("Original Data:\n", X)

# --- Example 1: Linear Regressor + Logistic Classifier ---
lr = OrdinaryLeastSquares()
logreg = LogisticRegression()

custom_imputer_1 = CustomImputer(regressor=lr, classifier=logreg)
X_imputed_1 = custom_imputer_1.fit_transform(X)
print("\nImputed with LinearRegression/LogisticRegression:\n", X_imputed_1)

# --- Example 2: Random Forests ---
rfr = RandomForestRegressor(n_estimators=5)
rfc = RandomForestClassifier(n_estimators=5)

custom_imputer_2 = CustomImputer(regressor=rfr, classifier=rfc)
X_imputed_2 = custom_imputer_2.fit_transform(X)
print("\nImputed with RandomForests:\n", X_imputed_2)

# --- Example 3: Only Regressor (Categorical imputation skipped) ---
lr2 = OrdinaryLeastSquares()
custom_imputer_3 = CustomImputer(regressor=lr2)
X_imputed_3 = custom_imputer_3.fit_transform(X)
print("\nImputed with only LinearRegression (cat skipped):\n", X_imputed_3)
