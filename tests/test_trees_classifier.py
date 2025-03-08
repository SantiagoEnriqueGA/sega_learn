import unittest
import warnings
import sys
import os
from matplotlib.pylab import f
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.trees import *
from sega_learn.utils import make_classification
from tests.utils import synthetic_data_regression, suppress_print

class TestClassifierTreeUtility(unittest.TestCase):
    """
    Unit test for the ClassifierTreeUtility class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_entropy: Tests the entropy method of the ClassifierTreeUtility class.
    - test_entropy_with_single_class: Tests the entropy method with a single class.
    - test_partition_classes: Tests the partition_classes method of the ClassifierTreeUtility class.
    - test_information_gain: Tests the information_gain method of the ClassifierTreeUtility class.
    - test_best_split: Tests the best_split method of the ClassifierTreeUtility class.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Classifier Tree Utility")
    
    def setUp(self):
        self.utility = ClassifierTreeUtility()
    
    def test_entropy(self):
        class_y = [0, 0, 1, 1, 1, 1]
        expected_entropy = 0.9182958340544896 #Expected entropy value (calculated manually)
        self.assertAlmostEqual(self.utility.entropy(class_y), expected_entropy, places=5)
           
    def test_entropy_with_single_class(self):
        class_y = [0, 0, 0, 0, 0, 0]
        expected_entropy = 0.0
        self.assertAlmostEqual(self.utility.entropy(class_y), expected_entropy, places=5)
        
    def test_partition_classes(self):
        X = [[2, 3], [1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0, 1]
        split_attribute = 0
        split_val = 2.5
        X_left, X_right, y_left, y_right = self.utility.partition_classes(X, y, split_attribute, split_val)
        self.assertEqual(X_left.tolist(), [[2, 3], [1, 2]])
        self.assertEqual(X_right.tolist(), [[3, 4], [5, 6]])
        self.assertEqual(y_left.tolist(), [0, 1])
        self.assertEqual(y_right.tolist(), [0, 1])

    def test_information_gain(self):
        previous_y = [0, 0, 1, 1, 1, 1]
        current_y = [[0, 0], [1, 1, 1, 1]]
        expected_info_gain = 0.9182958340544896 #Expected information gain value (calculated manually)
        self.assertAlmostEqual(self.utility.information_gain(previous_y, current_y), expected_info_gain, places=5)
 
    def test_best_split(self):
        X = [[2, 3], [1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0, 1]
        best_split = self.utility.best_split(X, y)

        self.assertIn('split_attribute', best_split)
        self.assertIn('split_val', best_split)
        self.assertIn('X_left', best_split)
        self.assertIn('X_right', best_split)
        self.assertIn('y_left', best_split)
        self.assertIn('y_right', best_split)
        self.assertIn('info_gain', best_split)
        
        self.assertIsInstance(best_split['split_attribute'], (int, np.integer))
        self.assertIsInstance(best_split['split_val'], (int, float, np.integer, np.float64))
        self.assertIsInstance(best_split['X_left'], np.ndarray)
        self.assertIsInstance(best_split['X_right'], np.ndarray)
        self.assertIsInstance(best_split['y_left'], np.ndarray)
        self.assertIsInstance(best_split['y_right'], np.ndarray)
        self.assertIsInstance(best_split['info_gain'], float)

class TestClassifierTree(unittest.TestCase):
    """
    Unit test for the ClassifierTree class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_init: Tests the initialization of the ClassifierTree class.
    - test_learn: Tests the learn method of the ClassifierTree class.
    - test_learn_single_value: Tests the learn method with a single value.
    - test_learn_bad_type: Tests the learn method with a bad type.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Classifier Tree")
        
    def setUp(self):
        self.tree = ClassifierTree(max_depth=5)
        
    def test_init(self):
        self.assertEqual(self.tree.max_depth, 5)
        self.assertDictEqual(self.tree.tree, {})
        
    def test_learn(self):
        from sega_learn.utils import make_classification
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
        self.tree.learn(X, y)
        self.assertIsInstance(self.tree.tree, dict)
    
    def test_learn_single_value(self):
        X = [[1, 2], [1, 2], [1, 2]]
        y = [0, 0, 0]
        self.tree.learn(X, y)
        self.assertEqual(self.tree.tree, {})       
        
    def test_learn_bad_type(self):
        X = "not a list"
        y = "not a list"
        with self.assertRaises(TypeError):
            self.tree.learn(X, y)    

class TestRandomForestClassifier(unittest.TestCase):
    """
    Unit test for the RandomForestClassifier class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_init: Tests the initialization of the RandomForestClassifier class.
    - test_reset: Tests the reset method of the RandomForestClassifier class.
    - test_boostraping: Tests the bootstrapping method of the RandomForestClassifier class.
    - test_bootstrapping_empty: Tests the bootstrapping method with an empty dataset.
    - test_bootstrapping_single_value: Tests the bootstrapping method with a single value.
    - test_bootstrapping_bad_type: Tests the bootstrapping method with a bad type.
    - test_fitting: Tests the fitting method of the RandomForestClassifier class.
    - test_fitting_single_value: Tests the fitting method with a single value.
    - test_fitting_bad_type: Tests the fitting method with a bad type.
    - test_voting: Tests the voting method of the RandomForestClassifier class.
    - test_voting_single_value: Tests the voting method with a single value.
    - test_fit: Tests the fit method of the RandomForestClassifier class.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Random Forest Classifier")
        
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
        self.rf = RandomForestClassifier(X, y, max_depth=10, forest_size=10, display=False, random_seed=0)
        
    def test_init(self):
        self.assertEqual(self.rf.max_depth, 10)
        self.assertEqual(self.rf.forest_size, 10)
        self.assertIsInstance(self.rf.decision_trees, list)
        self.assertIsInstance(self.rf.decision_trees[0], ClassifierTree)
        
    def test_reset(self):
        self.rf.reset()
        self.assertEqual(self.rf.forest_size, 10)
        self.assertEqual(self.rf.max_depth, 10)
        self.assertEqual(self.rf.random_seed, 0)
        self.assertIsInstance(self.rf.decision_trees, list)
        self.assertIsInstance(self.rf.decision_trees[0], ClassifierTree)
        
    def test_boostraping(self):
        self.rf.bootstrapping(self.rf.XX)
        self.assertEqual(len(self.rf.bootstraps_datasets), self.rf.forest_size)
        self.assertEqual(len(self.rf.bootstraps_labels), self.rf.forest_size)
        self.assertEqual(len(self.rf.bootstraps_datasets[0]), len(self.rf.XX))
        self.assertEqual(len(self.rf.bootstraps_labels[0]), len(self.rf.XX))

    def test_bootstrapping_empty(self):
        X = []
        y = []
        self.rf.bootstrapping(X)
        self.assertEqual(len(self.rf.bootstraps_datasets), 10)
        self.assertEqual(len(self.rf.bootstraps_labels), 10)
        for i in range(10):
            self.assertEqual(len(self.rf.bootstraps_datasets[i]), 0)
            self.assertEqual(len(self.rf.bootstraps_labels[i]), 0)

    def test_bootstrapping_single_value(self):
        X = [[1, 2, 3]]
        y = [1]
        self.rf.bootstrapping(X)
        self.assertEqual(len(self.rf.bootstraps_datasets), 10)
        self.assertEqual(len(self.rf.bootstraps_labels), 10)
        for i in range(10):
            self.assertEqual(len(self.rf.bootstraps_datasets[i]), 1)
            self.assertEqual(len(self.rf.bootstraps_labels[i]), 1)
    
    def test_bootstrapping_bad_type(self):
        X = "not a list"
        with self.assertRaises(TypeError):
            self.rf.bootstrapping(X)
    
    def test_fitting(self):
        self.rf.bootstrapping(self.rf.XX)
        self.rf.fitting()
        for tree in self.rf.decision_trees:
            self.assertIsInstance(tree, dict)
            
    def test_fitting_single_value(self):
        X = [[1, 2, 3]]
        self.rf.bootstrapping(X)
        self.rf.fitting()
        for tree in self.rf.decision_trees:
            self.assertIn('label', tree)           

    def test_voting(self):
        self.rf.bootstrapping(self.rf.XX)
        self.rf.fitting()
        predictions = self.rf.voting(self.rf.X)
        self.assertEqual(len(predictions), len(self.rf.X))
        self.assertIsInstance(predictions, list)

    def test_voting_single_value(self):
        X = [[1, 2, 3]]
        self.rf.bootstrapping(X)
        self.rf.fitting()
        predictions = self.rf.voting(X)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, list)

    def test_fit(self):
        self.rf.fit(verbose=False)
        self.assertGreaterEqual(self.rf.accuracy, 0.0)
        self.assertLessEqual(self.rf.accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()