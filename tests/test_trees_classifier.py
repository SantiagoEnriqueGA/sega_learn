import unittest
import warnings
import sys
import os
from matplotlib.pylab import f
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.trees import *
from sega_learn.trees.randomForestClassifier import _fit_tree, _classify_oob
from sega_learn.utils import make_classification
from tests.utils import synthetic_data_regression, suppress_print

class TestClassifierTreeUtility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Classifier Tree Utility", end="", flush=True)
    
    def setUp(self):
        """Set up the ClassifierTreeUtility instance for testing."""
        self.utility = ClassifierTreeUtility()
    
    def test_entropy(self):
        """Tests the entropy method of the ClassifierTreeUtility class."""
        class_y = [0, 0, 1, 1, 1, 1]
        expected_entropy = 0.9182958340544896 #Expected entropy value (calculated manually)
        self.assertAlmostEqual(self.utility.entropy(class_y), expected_entropy, places=5)
           
    def test_entropy_with_single_class(self):
        """Tests the entropy method with a single class."""
        class_y = [0, 0, 0, 0, 0, 0]
        expected_entropy = 0.0
        self.assertAlmostEqual(self.utility.entropy(class_y), expected_entropy, places=5)
        
    def test_partition_classes(self):
        """Tests the partition_classes method of the ClassifierTreeUtility class."""
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
        """Tests the information_gain method of the ClassifierTreeUtility class."""
        previous_y = [0, 0, 1, 1, 1, 1]
        current_y = [[0, 0], [1, 1, 1, 1]]
        expected_info_gain = 0.9182958340544896 #Expected information gain value (calculated manually)
        self.assertAlmostEqual(self.utility.information_gain(previous_y, current_y), expected_info_gain, places=5)
 
    def test_best_split(self):
        """Tests the best_split method of the ClassifierTreeUtility class."""
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

    def test_entropy_empty(self):
        """Test entropy with an empty class list."""
        class_y = []
        expected_entropy = 0.0
        self.assertAlmostEqual(self.utility.entropy(class_y), expected_entropy, places=5)

    def test_entropy_single_element(self):
        """Test entropy with a single element."""
        class_y = [1]
        expected_entropy = 0.0
        self.assertAlmostEqual(self.utility.entropy(class_y), expected_entropy, places=5)

    def test_partition_classes_empty(self):
        """Test partition_classes with empty inputs."""
        X = []
        y = []
        split_attribute = 0
        split_val = 0.5
        X_left, X_right, y_left, y_right = self.utility.partition_classes(X, y, split_attribute, split_val)
        self.assertEqual(len(X_left), 0)
        self.assertEqual(len(X_right), 0)
        self.assertEqual(len(y_left), 0)
        self.assertEqual(len(y_right), 0)

    def test_partition_classes_single_element(self):
        """Test partition_classes with a single element."""
        X = [[1, 2]]
        y = [1]
        split_attribute = 0
        split_val = 1.5
        X_left, X_right, y_left, y_right = self.utility.partition_classes(X, y, split_attribute, split_val)
        self.assertEqual(X_left.tolist(), [[1, 2]])
        self.assertEqual(X_right.tolist(), [])
        self.assertEqual(y_left.tolist(), [1])
        self.assertEqual(y_right.tolist(), [])

    def test_information_gain_empty(self):
        """Test information_gain with empty inputs."""
        previous_y = []
        current_y = [[], []]
        expected_info_gain = 0.0
        self.assertAlmostEqual(self.utility.information_gain(previous_y, current_y), expected_info_gain, places=5)

    def test_information_gain_no_split(self):
        """Test information_gain when no split occurs."""
        previous_y = [1, 1, 1, 1]
        current_y = [[1, 1, 1, 1], []]
        expected_info_gain = 0.0
        self.assertAlmostEqual(self.utility.information_gain(previous_y, current_y), expected_info_gain, places=5)

    def test_best_split_empty(self):
        """Test best_split with empty inputs."""
        X = []
        y = []
        best_split = self.utility.best_split(X, y)
        self.assertIsNone(best_split['split_attribute'])
        self.assertIsNone(best_split['split_val'])
        self.assertEqual(len(best_split['X_left']), 0)
        self.assertEqual(len(best_split['X_right']), 0)
        self.assertEqual(len(best_split['y_left']), 0)
        self.assertEqual(len(best_split['y_right']), 0)
        self.assertEqual(best_split['info_gain'], 0)

    def test_best_split_single_element(self):
        """Test best_split with a single element."""
        X = [[1, 2]]
        y = [1]
        best_split = self.utility.best_split(X, y)
        self.assertIsNone(best_split['split_attribute'])
        self.assertIsNone(best_split['split_val'])
        self.assertEqual(len(best_split['X_left']), 0)
        self.assertEqual(len(best_split['X_right']), 0)
        self.assertEqual(len(best_split['y_left']), 0)
        self.assertEqual(len(best_split['y_right']), 0)
        self.assertEqual(best_split['info_gain'], 0)

class TestClassifierTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Classifier Tree", end="", flush=True)
        
    def setUp(self):
        """Set up the ClassifierTree instance for testing."""
        self.tree = ClassifierTree(max_depth=5)
        
    def test_init(self):
        """Tests the initialization of the ClassifierTree class."""
        self.assertEqual(self.tree.max_depth, 5)
        self.assertDictEqual(self.tree.tree, {})
        
    def test_learn(self):
        """Tests the learn method of the ClassifierTree class."""
        from sega_learn.utils import make_classification
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
        self.tree.learn(X, y)
        self.assertIsInstance(self.tree.tree, dict)
    
    def test_learn_single_value(self):
        """Tests the learn method with a single value."""
        X = [[1, 2], [1, 2], [1, 2]]
        y = [0, 0, 0]
        self.tree.learn(X, y)
        self.assertEqual(self.tree.tree, {})       
        
    def test_learn_bad_type(self):
        """Tests the learn method with a bad type."""
        X = "not a list"
        y = "not a list"
        with self.assertRaises(TypeError):
            self.tree.learn(X, y)    

    def test_learn_empty_dataset(self):
        """Test learning with an empty dataset."""
        X = []
        y = []
        tree = self.tree.learn(X, y)
        self.assertEqual(tree, {})

    def test_learn_single_data_point(self):
        """Test learning with a single data point."""
        X = [[1, 2, 3]]
        y = [1]
        tree = self.tree.learn(X, y)
        self.assertEqual(tree, {'label': 1})

    def test_learn_pure_labels(self):
        """Test learning when all labels are the same."""
        X = [[1, 2], [3, 4], [5, 6]]
        y = [1, 1, 1]
        tree = self.tree.learn(X, y)
        self.assertEqual(tree, {'label': 1})

    def test_learn_max_depth(self):
        """Test learning when the maximum depth is reached."""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 1, 0, 1]
        self.tree.max_depth = 1
        tree = self.tree.learn(X, y)
        self.assertIn('split_attribute', tree)
        self.assertIn('split_val', tree)

    def test_classify_empty_tree(self):
        """Test classification with an empty tree."""
        record = [1, 2, 3]
        result = self.tree.classify({}, record)
        self.assertIsNone(result)

    def test_classify_single_node_tree(self):
        """Test classification with a single-node tree."""
        tree = {'label': 1}
        record = [1, 2, 3]
        result = self.tree.classify(tree, record)
        self.assertEqual(result, 1)

    def test_classify_with_split(self):
        """Test classification with a tree containing a split."""
        tree = {
            'split_attribute': 0,
            'split_val': 2.5,
            'left': {'label': 0},
            'right': {'label': 1}
        }
        record_left = [2, 3]
        record_right = [3, 4]
        result_left = self.tree.classify(tree, record_left)
        result_right = self.tree.classify(tree, record_right)
        self.assertEqual(result_left, 0)
        self.assertEqual(result_right, 1)

class TestRandomForestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Random Forest Classifier", end="", flush=True)
        
    def setUp(self):
        """Set up the RandomForestClassifier instance for testing."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
        self.rf = RandomForestClassifier(X=X, y=y, max_depth=10, forest_size=10, random_seed=0)
        
    def test_init(self):
        """Tests the initialization of the RandomForestClassifier class."""
        self.assertEqual(self.rf.max_depth, 10)
        self.assertEqual(self.rf.n_estimators, 10)
        self.assertIsInstance(self.rf.trees, list)
    
    def test_fitting(self):
        """Tests the fitting method of the RandomForestClassifier class."""
        self.rf.fit()
        for tree in self.rf.trees:
            self.assertIsInstance(tree, dict)
            
    def test_fitting_single_value(self):
        """Tests the fitting method with a single value."""
        X = [[1, 2, 3]]
        self.rf.fit()
        for tree in self.rf.trees:
            self.assertIn('split_attribute', tree)           

    def test_voting(self):
        """Tests the voting method of the RandomForestClassifier class."""
        self.rf.fit()
        predictions = self.rf.predict(self.rf.X)
        self.assertEqual(len(predictions), len(self.rf.X))
        self.assertIsInstance(predictions, list)

    def test_voting_single_value(self):
        """Tests the voting method with a single value."""
        X = [[1, 2, 3, 4, 5]]
        self.rf.fit()
        predictions = self.rf.predict(X)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, list)

    def test_fit(self):
        """Tests the fit method of the RandomForestClassifier class."""
        self.rf.fit(verbose=False)
        self.assertGreaterEqual(self.rf.accuracy, 0.0)
        self.assertLessEqual(self.rf.accuracy, 1.0)

    def test_fit_single_data_point(self):
        """Test fitting the RandomForestClassifier with a single data point."""
        X_single = np.random.rand(1, 5)  # Single sample, 5 features
        y_single = np.array([1])  # Single label
        self.rf.fit(X_single, y_single)
        self.assertEqual(len(self.rf.trees), 10)  # Ensure 10 trees are trained

    def test_fit_empty_dataset(self):
        """Test fitting the RandomForestClassifier with an empty dataset."""
        X_empty = np.empty((0, 5))  # No samples, 5 features
        y_empty = np.empty((0,))
        with self.assertRaises(ValueError):
            self.rf.fit(X_empty, y_empty)

    def test_fit_no_features(self):
        """Test fitting the RandomForestClassifier with no features."""
        X = np.empty((10, 0))  # 10 samples, 0 features
        y = np.random.randint(0, 2, size=10)
        with self.assertRaises(ValueError):
            self.rf.fit(X, y)

    def test_fit_no_samples(self):
        """Test fitting the RandomForestClassifier with no samples."""
        X = np.empty((0, 5))  # 0 samples, 5 features
        y = np.empty((0,))
        with self.assertRaises(ValueError):
            self.rf.fit(X, y)

    def test_fit_single_class(self):
        """Test fitting the RandomForestClassifier with a single class."""
        X = np.random.rand(10, 5)  # 10 samples, 5 features
        y = np.zeros(10)  # Single class
        self.rf.fit(X, y)
        for tree in self.rf.trees:
            self.assertEqual(tree['label'], 0)

    def test_predict_single_sample(self):
        """Test predicting with a single sample."""
        X = [[1, 2, 3, 4, 5]]
        self.rf.fit()
        predictions = self.rf.predict(X)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, list)

    def test_oob_predictions(self):
        """Test out-of-bag predictions."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, size=100)
        self.rf.fit(X, y)
        oob_predictions = _classify_oob(X, self.rf.trees, self.rf.bootstraps)
        self.assertEqual(len(oob_predictions), len(X))
        self.assertIsInstance(oob_predictions, list)

if __name__ == '__main__':
    unittest.main()