import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import LogisticRegression
from sega_learn.svm import LinearSVC
from sega_learn.trees import RandomForestClassifier
from sega_learn.utils import Metrics, make_classification, train_test_split
from sega_learn.utils.voting import VotingClassifier

# Define metrics
accuracy_score = Metrics.accuracy

# --- 1. Generate Data ---
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- 2. Create and Fit Base Classifiers ---
print("Fitting base classifiers...")

# Logistic Regression
clf1 = LogisticRegression(max_iter=500)
clf1.fit(X_train, y_train)
print(
    f"  Logistic Regression fitted. Accuracy: {accuracy_score(y_test, clf1.predict(X_test)):.4f}"
)

# Random Forest Classifier
clf2 = RandomForestClassifier(forest_size=20, max_depth=5, random_seed=42)
clf2.fit(X_train, y_train)
print(
    f"  Random Forest fitted. Accuracy: {accuracy_score(y_test, clf2.predict(X_test)):.4f}"
)

# Linear SVC
y_train_svm = np.where(y_train == 0, -1, 1)  # Adapt labels if needed by SVC
clf3 = LinearSVC(max_iter=500)
clf3.fit(X_train, y_train_svm)
# Wrap predict to return 0/1 for consistency in hard voting evaluation
original_predict_svc = clf3.predict
clf3.predict = lambda x: np.where(original_predict_svc(x) == -1, 0, 1)
print(
    f"  Linear SVC fitted. Accuracy: {accuracy_score(y_test, clf3.predict(X_test)):.4f}"
)


# --- 3. Create and Use Voting Classifiers ---

# Hard Voting (Equal Weights)
print("\n--- Hard Voting Classifier (Equal Weights) ---")
estimators_hard = [clf1, clf2, clf3]
voting_clf_hard = VotingClassifier(estimators=estimators_hard)
# NOTE: VotingClassifier doesn't need fit if estimators are pre-fitted

y_pred_hard = voting_clf_hard.predict(X_test)
hard_accuracy = accuracy_score(y_test, y_pred_hard)
print(f"Hard Voting Accuracy: {hard_accuracy:.4f}")
voting_clf_hard.show_models()

# Hard Voting with Weights
print("\n--- Hard Voting Classifier (Weighted) ---")
weights_hard = [0.3, 0.4, 0.3]  # Example weights
voting_clf_hard_weighted = VotingClassifier(
    estimators=estimators_hard, weights=weights_hard
)
y_pred_hard_weighted = voting_clf_hard_weighted.predict(X_test)
hard_weighted_accuracy = accuracy_score(y_test, y_pred_hard_weighted)
print(f"Hard Voting (Weighted) Accuracy: {hard_weighted_accuracy:.4f}")
voting_clf_hard_weighted.show_models()
