import unittest
import warnings
import sys
import os
from matplotlib.pylab import f
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.trees import *

class TestIsolationUtils(unittest.TestCase):
    """
    Unit test for the IsolationTreeUtility class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_compute_avg_path_length: Tests the compute_avg_path_length method of the IsolationTreeUtility class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Isolation Tree Utility", end="", flush=True)
    
    def test_compute_avg_path_length(self):
        self.assertAlmostEqual(IsolationUtils.compute_avg_path_length(1), 0)
        self.assertAlmostEqual(IsolationUtils.compute_avg_path_length(2), 0.15443132979999996)
        self.assertAlmostEqual(IsolationUtils.compute_avg_path_length(256), 10.244770920116851)

class TestIsolationTree(unittest.TestCase):
    """
    Unit test for the IsolationTree class.
    Methods:
    - setUpClass: Initializes a new instance of the class before each test method is run.
    - test_init: Tests the initialization of the IsolationTree class.
    - test_fit: Tests the fit method of the IsolationTree class.
    - test_fit_with_empty_data: Tests the fit method with empty data.
    - test_path_length: Tests the path_length method of the IsolationTree class.
    - test_path_length_with_empty_data: Tests the path_length method with empty data.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Isolation Tree", end="", flush=True)
        
    def setUp(self):
        self.X = np.random.randn(100, 2)
        self.tree = IsolationTree(max_depth=10)
        self.tree.fit(self.X)
        
    def test_fit(self):
        self.assertIsNotNone(self.tree.tree)
        
    def test_fit_with_empty_data(self):
        with self.assertRaises(ValueError):
            empty_data = np.empty((0, 2))
            tree = IsolationTree(max_depth=10)
            tree.fit(empty_data)
        
    def test_path_length(self):
        sample = np.random.randn(2)
        path_length = self.tree.path_length(sample)
        self.assertGreaterEqual(path_length, 0)
        
    def test_path_length_with_empty_data(self):
        with self.assertRaises(ValueError):
            sample = np.empty((0, 2))
            self.tree.path_length(sample)                     
            
class TestIsolationForest(unittest.TestCase):
    """
    Unit test for the IsolationForest class.
    Methods:
    - setUpClass: Initializes a new instance of the class before each test method is run.
    - test_init: Tests the initialization of the IsolationForest class.
    - test_fit: Tests the fit method of the IsolationForest class.
    - test_fit_with_empty_data: Tests the fit method with empty data.
    - test_anomaly_score: Tests the anomaly_score method of the IsolationForest class.
    - test_anomaly_score_with_empty_data: Tests the anomaly_score method with empty data.
    - test_predict: Tests the predict method of the IsolationForest class.
    - test_predict_with_empty_data: Tests the predict method with empty data.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Isolation Forest", end="", flush=True)
        
    def setUp(self):
        self.X = np.random.randn(100, 2)
        self.forest = IsolationForest(n_trees=10, max_samples=50, max_depth=10)
        self.forest.fit(self.X)
        
    def test_fit(self):
        self.assertEqual(len(self.forest.trees), 10)
    
    def test_fit_with_empty_data(self):
        with self.assertRaises(ValueError):
            empty_data = np.empty((0, 2))
            forest = IsolationForest(n_trees=10, max_samples=50, max_depth=10)
            forest.fit(empty_data)
        
    def test_anomaly_score(self):
        sample = np.random.randn(2)
        score = self.forest.anomaly_score(sample)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_anomaly_score_with_empty_data(self):
        with self.assertRaises(ValueError):
            sample = np.empty((0, 2))
            self.forest.anomaly_score(sample)
        
    def test_predict(self):
        sample = np.random.randn(2)
        prediction = self.forest.predict(sample)
        self.assertIn(prediction, [0, 1])

    def test_predict_with_empty_data(self):
        with self.assertRaises(ValueError):
            sample = np.empty((0, 2))
            self.forest.predict(sample)

if __name__ == '__main__':
    unittest.main()