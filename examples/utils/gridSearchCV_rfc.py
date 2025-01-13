# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sega_learn.trees.randomForestClassifier as rfc
import sega_learn.utils.dataPrep as dp
import sega_learn.utils.model_selection as ms

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

grid = [
    {'forest_size': [5, 20]},
    {'max_depth': [5, 20]}
]
# grid_search = ms.GridSearchCV(rfc.RandomForestClassifier, grid, cv=3, metric='precision', direction='maximize')
# grid_search = ms.GridSearchCV(rfc.RandomForestClassifier, grid, cv=3, metric='recall', direction='maximize')
grid_search = ms.GridSearchCV(rfc.RandomForestClassifier, grid, cv=3, metric='accuracy', direction='minimize')
model = grid_search.fit(X, y)

print(f"Best Score: {grid_search.best_score_}")
print(f"Best Params: {grid_search.best_params_}")
