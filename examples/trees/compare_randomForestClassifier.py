import numpy as np
import time

# Import Custom Classes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sega_learn.trees.randomForestClassifier as rfc
from sega_learn.utils import Metrics


from sega_learn.utils import make_classification
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    
    
start_time = time.time()
# Initialize random forest object
rfObj = rfc.RandomForestClassifier(X, y, max_depth=5, forest_size=10, display=False, random_seed=0)

# Train random forest model
rfObj.fit()
predictions = rfObj.predict(X)

end_time = time.time()
print(f"\nTraining and prediction time: {end_time - start_time:.2f} seconds")
print(f"Accuracy: {Metrics.accuracy(y, predictions)}")
    

# --------------------------------------------------------------------------------------------

import sega_learn.trees_dev.randomForestClassifier as rfc

start_time = time.time()
# Initialize random forest object
rfObj = rfc.RandomForestClassifier(X=X, y=y, max_depth=5, forest_size=10, random_seed=0)

# Train random forest model
rfObj.fit()
predictions = rfObj.predict(X)

end_time = time.time()
print(f"\nTraining and prediction time: {end_time - start_time:.2f} seconds")
stats = rfObj.get_stats()
for key, value in stats.items():
    print(f"\t{key}: {value:.2f}")
