import unittest
import warnings
import sys
import os
from matplotlib.pylab import f
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.trees import *
from tests.utils import synthetic_data_regression, suppress_print

class TestRegressorTreeUtility(unittest.TestCase):
    """
    Unit test for the RegressorTreeUtility class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_calculate_variance: Tests the calculate_variance method with various inputs.
    - test_calculate_variance_empty: Tests the calculate_variance method with an empty list.
    - test_calculate_variance_single_value: Tests the calculate_variance method with a list containing a single value.
    - test_calculate_variance_negative_values: Tests the calculate_variance method with a list containing negative values.
    - test_calculate_variance_bad_type: Tests the calculate_variance method with a bad type input.
    - test_partition_classes: Tests the partition_classes method with various inputs.
    - test_partition_classes_empty: Tests the partition_classes method with an empty list.
    - test_partition_classes_single_value: Tests the partition_classes method with a list containing a single value.
    - test_partition_classes_bad_type: Tests the partition_classes method with a bad type input.
    - test_information_gain: Tests the information_gain method with various inputs.
    - test_information_gain_empty: Tests the information_gain method with an empty list.
    - test_information_gain_single_value: Tests the information_gain method with a list containing a single value.
    - test_information_gain_bad_type: Tests the information_gain method with a bad type input.
    - test_best_split: Tests the best_split method with various inputs.
    - test_best_split_empty: Tests the best_split method with an empty list.
    - test_best_split_single_value: Tests the best_split method with a list containing a single value.
    - test_best_split_bad_type: Tests the best_split method with a bad type input.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Regressor Tree Utility")
    
    def setUp(self):
        self.utility = RegressorTreeUtility()
        
    def test_calculate_variance(self):
        y = [1, 2, 3, 4, 5]
        expected_variance = np.var(y)
        calculated_variance = self.utility.calculate_variance(y)
        self.assertAlmostEqual(calculated_variance, expected_variance, places=5)
        
    def test_calculate_variance_empty(self):
        y = []
        expected_variance = 0
        calculated_variance = self.utility.calculate_variance(y)
        self.assertEqual(calculated_variance, expected_variance)
        
    def test_calculate_variance_single_value(self):
        y = [5, 5, 5, 5, 5]
        expected_variance = 0
        calculated_variance = self.utility.calculate_variance(y)
        self.assertEqual(calculated_variance, expected_variance)
    
    def test_calculate_variance_negative_values(self):
        y = [-1, -2, -3, -4, -5]
        expected_variance = np.var(y)
        calculated_variance = self.utility.calculate_variance(y)
        self.assertAlmostEqual(calculated_variance, expected_variance, places=5)
    
    def test_calculate_variance_bad_type(self):
        y = "not a list"
        with self.assertRaises(TypeError):
            self.utility.calculate_variance(y)
            
    def test_partition_classes(self):
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [1, 2, 3, 4]
        split_attribute = 0
        split_val = 4
        X_left, X_right, y_left, y_right = self.utility.partition_classes(X, y, split_attribute, split_val)
        self.assertEqual(X_left.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(X_right.tolist(), [[5, 6], [7, 8]])
        self.assertEqual(y_left.tolist(), [1, 2])
        self.assertEqual(y_right.tolist(), [3, 4])
        
    def test_partition_classes_empty(self):
        X = []
        y = []
        split_attribute = 0
        split_val = 4
        X_left, X_right, y_left, y_right = self.utility.partition_classes(X, y, split_attribute, split_val)
        self.assertEqual(X_left.shape, (0, 1))
        self.assertEqual(X_right.shape, (0, 1))
        self.assertEqual(y_left.shape, (0,))
        self.assertEqual(y_right.shape, (0,))
        
    def test_partition_classes_single_value(self):
        X = [[1, 2]]
        y = [1]
        split_attribute = 0
        split_val = 4
        X_left, X_right, y_left, y_right = self.utility.partition_classes(X, y, split_attribute, split_val)
        self.assertEqual(X_left.tolist(), [[1, 2]])
        self.assertEqual(X_right.shape, (0, 2))
        self.assertEqual(y_left.tolist(), [1])
        self.assertEqual(y_right.shape, (0,))
        
    def test_partition_classes_bad_type(self):
        X = "not a list"
        y = "not a list"
        split_attribute = 0
        split_val = 4
        with self.assertRaises(TypeError):
            self.utility.partition_classes(X, y, split_attribute, split_val)
            
    def test_information_gain(self):
        previous_y = [1, 2, 3, 4, 5]
        current_y = [[1, 2], [3, 4, 5]]
        expected_gain = self.utility.calculate_variance(previous_y) - (
            self.utility.calculate_variance(current_y[0]) * len(current_y[0]) +
            self.utility.calculate_variance(current_y[1]) * len(current_y[1])
        ) / len(previous_y)
        calculated_gain = self.utility.information_gain(previous_y, current_y)
        self.assertAlmostEqual(calculated_gain, expected_gain, places=5)
        
    def test_information_gain_empty(self):
        previous_y = []
        current_y = [[], []]
        expected_gain = 0
        calculated_gain = self.utility.information_gain(previous_y, current_y)
        self.assertEqual(calculated_gain, expected_gain)
        
    def test_information_gain_single_value(self):
        previous_y = [5, 5, 5, 5, 5]
        current_y = [[5, 5], [5, 5]]
        expected_gain = self.utility.calculate_variance(previous_y) - (
            self.utility.calculate_variance(current_y[0]) * len(current_y[0]) +
            self.utility.calculate_variance(current_y[1]) * len(current_y[1])
        ) / len(previous_y)
        calculated_gain = self.utility.information_gain(previous_y, current_y)
        self.assertAlmostEqual(calculated_gain, expected_gain, places=5)
        
    def test_information_gain_bad_type(self):
        previous_y = "not a list"
        current_y = "not a list"
        with self.assertRaises(TypeError):
            self.utility.information_gain(previous_y, current_y)

    def test_best_split(self):
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [1, 2, 3, 4]
        best_split = self.utility.best_split(X, y)
        self.assertIn('split_attribute', best_split)
        self.assertIn('split_val', best_split)
        self.assertIn('X_left', best_split)
        self.assertIn('X_right', best_split)
        self.assertIn('y_left', best_split)
        self.assertIn('y_right', best_split)
        self.assertIn('info_gain', best_split)
        self.assertGreaterEqual(best_split['info_gain'], 0)
        
    def test_best_split_empty(self):
        X = []
        y = []
        best_split = self.utility.best_split(X, y)
        self.assertEqual(best_split['split_attribute'], None)
        self.assertEqual(best_split['split_val'], None)
        self.assertEqual(best_split['X_left'].shape, (0, 1))
        self.assertEqual(best_split['X_right'].shape, (0, 1))
        self.assertEqual(best_split['y_left'].shape, (0,))
        self.assertEqual(best_split['y_right'].shape, (0,))
        
    def test_best_split_single_value(self):
        X = [[1, 2]]
        y = [1]
        best_split = self.utility.best_split(X, y)
        self.assertEqual(best_split['split_attribute'], None)
        self.assertEqual(best_split['split_val'], None)
        self.assertEqual(best_split['X_left'].shape, (0, 2))
        self.assertEqual(best_split['X_right'].shape, (0, 2))
        self.assertEqual(best_split['y_left'].shape, (0,))
        self.assertEqual(best_split['y_right'].shape, (0,))
        
    def test_best_split_bad_type(self):
        X = "not a list"
        y = "not a list"
        with self.assertRaises(TypeError):
            self.utility.best_split(X, y)    

class TestRegressorTree(unittest.TestCase):
    """
    Unit test for the RegressorTree class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_init: Tests the initialization of the RegressorTree class.
    - test_learn: Tests the learn method of the RegressorTree class.
    - test_learn_empty: Tests the learn method with an empty dataset.
    - test_learn_single_value: Tests the learn method with a single value dataset.
    - test_learn_bad_type: Tests the learn method with a bad type input.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Regressor Tree")
        
    def setUp(self):
        self.tree = RegressorTree()
        
    def test_init(self):
        self.assertEqual(self.tree.max_depth, 5)
        self.assertDictEqual(self.tree.tree, {})
        
    def test_learn(self):
        X, y = synthetic_data_regression()
        self.tree.learn(X, y)
        self.assertIsInstance(self.tree.tree, dict)
    
    def test_learn_empty(self):
        X = []
        y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.tree.learn(X, y)
        self.assertEqual(self.tree.tree, {})

    def test_learn_single_value(self):
        X = [[1, 2]]
        y = [1]
        self.tree.learn(X, y)
        self.assertEqual(self.tree.tree, {})
    
    def test_learn_bad_type(self):
        X = "not a list"
        y = "not a list"
        with self.assertRaises(TypeError):
            self.tree.learn(X, y)    

class TestRandomForestRegressor(unittest.TestCase):
    """
    Unit test for the RandomForestRegressor class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_init: Tests the initialization of the RandomForestRegressor class.
    - test_reset: Tests the reset method of the RandomForestRegressor class.
    - test_bootstrapping: Tests the bootstrapping method of the RandomForestRegressor class.
    - test_bootstrapping_empty: Tests the bootstrapping method with an empty dataset.
    - test_bootstrapping_single_value: Tests the bootstrapping method with a single value dataset.
    - test_bootstrapping_bad_type: Tests the bootstrapping method with a bad type input.
    - test_fitting: Tests the fitting method of the RandomForestRegressor class.
    - test_fitting_empty: Tests the fitting method with an empty dataset.
    - test_fitting_single_value: Tests the fitting method with a single value dataset.
    - test_fitting_bad_type: Tests the fitting method with a bad type input.
    - test_voting: Tests the voting method of the RandomForestRegressor class.
    - test_voting_empty: Tests the voting method with an empty dataset.
    - test_voting_single_value: Tests the voting method with a single value dataset.
    - test_fit: Tests the fit method of the RandomForestRegressor class.
    - test_get_stats: Tests the get_stats method of the RandomForestRegressor class.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Random Forest Regressor")
        
    def setUp(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=3)
        self.rf = RandomForestRegressor(X, y, forest_size=10, random_seed=0, max_depth=10)
        
    def test_init(self):
        self.assertEqual(self.rf.forest_size, 10)
        self.assertEqual(self.rf.max_depth, 10)
        self.assertEqual(self.rf.random_seed, 0)
        self.assertIsInstance(self.rf.decision_trees, list)
        self.assertIsInstance(self.rf.decision_trees[0], RegressorTree)
        
    def test_reset(self):
        self.rf.reset()
        self.assertEqual(self.rf.forest_size, 10)
        self.assertEqual(self.rf.max_depth, 10)
        self.assertEqual(self.rf.random_seed, 0)
        self.assertIsInstance(self.rf.decision_trees, list)
        self.assertIsInstance(self.rf.decision_trees[0], RegressorTree)
        
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
            
    def test_fitting_empty(self):
        X = []
        y = []
        self.rf.bootstrapping(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.rf.fitting()
        for tree in self.rf.decision_trees:
            self.assertIn('value', tree)
            self.assertTrue(np.isnan(tree['value']))

    def test_fitting_single_value(self):
        X = [[1, 2, 3]]
        self.rf.bootstrapping(X)
        self.rf.fitting()
        for tree in self.rf.decision_trees:
            self.assertIn('value', tree)           

    def test_voting(self):
        self.rf.bootstrapping(self.rf.XX)
        self.rf.fitting()
        predictions = self.rf.voting(self.rf.X)
        self.assertEqual(len(predictions), len(self.rf.X))
        self.assertIsInstance(predictions, list)
        
    def test_voting_empty(self):
        X = []
        y = []
        self.rf.bootstrapping(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.rf.fitting()
        predictions = self.rf.voting(X)
        self.assertEqual(len(predictions), 0)
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
        self.assertGreaterEqual(self.rf.r2, 0)
        self.assertLessEqual(self.rf.r2, 1)      

    def test_get_stats(self):
        self.rf.fit(verbose=False)
        stats = self.rf.get_stats()
        self.assertIn("MSE", stats)
        self.assertIn("R^2", stats)
        self.assertIn("MAPE", stats)
        self.assertIn("MAE", stats)
        self.assertIn("RMSE", stats)

class TestGradientBoostedRegressor(unittest.TestCase):
    """
    Unit test for the GradientBoostedRegressor class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_init: Tests the initialization of the GradientBoostedRegressor class.
    - test_reset: Tests the reset method of the GradientBoostedRegressor class.
    - test_fit: Tests the fit method of the GradientBoostedRegressor class.
    - test_fit_stats: Tests the fit method with stats=False.
    - test_fit_empty: Tests the fit method with an empty dataset.
    - test_fit_single_value: Tests the fit method with a single value dataset.
    - test_fit_bad_type: Tests the fit method with a bad type input.
    - test_predict: Tests the predict method of the GradientBoostedRegressor class.
    - test_predict_empty: Tests the predict method with an empty dataset.
    - test_predict_single_value: Tests the predict method with a single value dataset.
    - test_predict_bad_type: Tests the predict method with a bad type input.
    - test_get_stats: Tests the get_stats method of the GradientBoostedRegressor class.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Gradient Boosted Regressor")
        
    def setUp(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=3)
        self.gbr = GradientBoostedRegressor(X, y, num_trees=10, max_depth=10, random_seed=0)
        
    def test_init(self):
        self.assertEqual(self.gbr.num_trees, 10)
        self.assertEqual(self.gbr.max_depth, 10)
        self.assertEqual(self.gbr.random_seed, 0)
        self.assertIsInstance(self.gbr.trees, list)
        self.assertIsInstance(self.gbr.trees[0], RegressorTree)
    
    def test_reset(self):
        self.gbr.reset()
        self.assertEqual(self.gbr.num_trees, 10)
        self.assertEqual(self.gbr.max_depth, 10)
        self.assertEqual(self.gbr.random_seed, 0)
        self.assertIsInstance(self.gbr.trees, list)
        self.assertIsInstance(self.gbr.trees[0], RegressorTree)
    
    def test_fit(self):
        self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.trees[0], dict)
        self.assertEqual(len(self.gbr.mean_absolute_residuals), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.mean_absolute_residuals[0], float)
        
    def test_fit_stats(self):
        with suppress_print():
            self.gbr.fit(stats=False)
        self.assertEqual(len(self.gbr.trees), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.trees[0], dict)
        self.assertEqual(len(self.gbr.mean_absolute_residuals), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.mean_absolute_residuals[0], float)
    
    def test_fit_empty(self):
        self.gbr.X = []
        self.gbr.y = []
        with self.assertRaises(ValueError):
            self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), 10)
        
    def test_fit_single_value(self):
        self.gbr.X = [[1, 2, 3]]
        self.gbr.y = [1]
        self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), 10)
        self.assertIsInstance(self.gbr.trees[0], dict)
        self.assertEqual(len(self.gbr.mean_absolute_residuals), 10)
        self.assertIsInstance(self.gbr.mean_absolute_residuals[0], float)
    
    def test_fit_bad_type(self):
        self.gbr.X = "not a list"
        self.gbr.y = "not a list"
        with self.assertRaises(TypeError):
            self.gbr.fit()
            
    def test_predict(self):
        self.gbr.fit()
        predictions = self.gbr.predict()
        self.assertEqual(len(predictions), len(self.gbr.X))
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_predict_empty(self):
        self.gbr.X = []
        self.gbr.y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            predictions = self.gbr.predict()
        self.assertEqual(len(predictions), 0)
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_predict_single_value(self):
        self.gbr.X = [[1, 2, 3]]
        self.gbr.y = [1]
        self.gbr.fit()
        predictions = self.gbr.predict()
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, np.ndarray)
        
    def test_predict_bad_type(self):
        self.gbr.X = "not a list"
        self.gbr.y = "not a list"
        with self.assertRaises(TypeError):
            self.gbr.predict()
            
    def test_get_stats(self):
        self.gbr.fit()
        predictions = self.gbr.predict()
        stats = self.gbr.get_stats(predictions)
        self.assertIn("MSE", stats)
        self.assertIn("R^2", stats)
        self.assertIn("MAPE", stats)
        self.assertIn("MAE", stats)
        self.assertIn("RMSE", stats)
    
if __name__ == '__main__':
    unittest.main()