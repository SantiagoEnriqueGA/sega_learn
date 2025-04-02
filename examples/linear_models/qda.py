import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import sega_learn.utils.metrics as mt
from sega_learn.linear_models import QuadraticDiscriminantAnalysis, make_sample_data
from sega_learn.utils import train_test_split

X, y = make_sample_data(
    n_samples=1000,
    n_features=2,
    cov_class_1=np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0,
    cov_class_2=np.array([[0.0, -1.0], [2.5, 0.7]]).T * 2.0,
    shift=[4, 1],
    seed=1,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)

print("\nConfusion Matrix")
cm = mt.Metrics.show_confusion_matrix(y, qda.predict(X))

print("\nClassification Report")
cls = mt.Metrics.show_classification_report(y, qda.predict(X))
