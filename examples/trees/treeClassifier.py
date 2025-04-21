import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.trees import treeClassifier
from sega_learn.utils import Metrics, make_classification

X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

# Train random forest model
tree = treeClassifier.ClassifierTree(max_depth=5)
tree.fit(X, y)

preds = tree.predict(X)
print(f"Accuracy:  {Metrics.accuracy(y, preds):.4f}")
print(f"Precision: {Metrics.precision(y, preds):.4f}")
print(f"Recall:    {Metrics.recall(y, preds):.4f}")
print(f"F1 Score:  {Metrics.f1_score(y, preds):.4f}")
