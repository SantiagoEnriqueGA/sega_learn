
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils.dataAugmentation import *
from sega_learn.utils import make_classification
from sega_learn.utils import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.7, 0.3], random_state=42, class_sep=.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show class distribution imbalance
print("Class distribution before SMOTE:")
print(f"Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")

# Create SMOTE object
smote = SMOTE(k_neighbors=5, random_state=42)

# Fit and resample the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:")
print(f"Class 0: {np.sum(y_resampled == 0)}, Class 1: {np.sum(y_resampled == 1)}")

print(f"\nOriginal dataset shape: {X_train.shape}, {y_train.shape}")
print(f"Resampled dataset shape: {X_resampled.shape}, {y_resampled.shape}")


# Fit both models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sega_learn.utils import Metrics
accuracy_score = Metrics.accuracy

# Train a Random Forest classifier on the original data
rf_original = RandomForestClassifier(random_state=42)
rf_original.fit(X_train, y_train)
y_pred_original = rf_original.predict(X_test)
print("\nClassification report for original data:")
print(classification_report(y_test, y_pred_original))
print(f"Accuracy for original data: {accuracy_score(y_test, y_pred_original)}")

# Train a Random Forest classifier on the resampled data
rf_resampled = RandomForestClassifier(random_state=42)
rf_resampled.fit(X_resampled, y_resampled)
y_pred_resampled = rf_resampled.predict(X_test)
print("\nClassification report for resampled data:")
print(classification_report(y_test, y_pred_resampled))
print(f"Accuracy for resampled data: {accuracy_score(y_test, y_pred_resampled)}")

