import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.trees import *
from sega_learn.trees.randomForestRegressor import _predict_oob
from tests.utils import suppress_print, synthetic_data_regression


class TestRegressorTreeUtility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Regressor Tree Utility", end="", flush=True)

    def setUp(self):
        """Sets up the RegressorTreeUtility instance for testing."""
        self.utility = RegressorTreeUtility()

    def test_calculate_variance(self):
        """Tests the calculate_variance method with various inputs."""
        y = [1, 2, 3, 4, 5]
        expected_variance = np.var(y)
        calculated_variance = self.utility.calculate_variance(y)
        self.assertAlmostEqual(calculated_variance, expected_variance, places=5)

    def test_calculate_variance_empty(self):
        """Tests the calculate_variance method with an empty list."""
        y = []
        expected_variance = 0
        calculated_variance = self.utility.calculate_variance(y)
        self.assertEqual(calculated_variance, expected_variance)

    def test_calculate_variance_single_value(self):
        """Tests the calculate_variance method with a list containing a single value."""
        y = [5, 5, 5, 5, 5]
        expected_variance = 0
        calculated_variance = self.utility.calculate_variance(y)
        self.assertEqual(calculated_variance, expected_variance)

    def test_calculate_variance_negative_values(self):
        """Tests the calculate_variance method with a list containing negative values."""
        y = [-1, -2, -3, -4, -5]
        expected_variance = np.var(y)
        calculated_variance = self.utility.calculate_variance(y)
        self.assertAlmostEqual(calculated_variance, expected_variance, places=5)

    def test_calculate_variance_bad_type(self):
        """Tests the calculate_variance method with a bad type input."""
        y = "not a list"
        with self.assertRaises(TypeError):
            self.utility.calculate_variance(y)

    def test_partition_classes(self):
        """Tests the partition_classes method with various inputs."""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [1, 2, 3, 4]
        split_attribute = 0
        split_val = 4
        X_left, X_right, y_left, y_right = self.utility.partition_classes(
            X, y, split_attribute, split_val
        )
        self.assertEqual(X_left.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(X_right.tolist(), [[5, 6], [7, 8]])
        self.assertEqual(y_left.tolist(), [1, 2])
        self.assertEqual(y_right.tolist(), [3, 4])

    def test_partition_classes_empty(self):
        """Tests the partition_classes method with an empty list."""
        X = []
        y = []
        split_attribute = 0
        split_val = 4
        X_left, X_right, y_left, y_right = self.utility.partition_classes(
            X, y, split_attribute, split_val
        )
        self.assertEqual(X_left.shape, (0, 1))
        self.assertEqual(X_right.shape, (0, 1))
        self.assertEqual(y_left.shape, (0,))
        self.assertEqual(y_right.shape, (0,))

    def test_partition_classes_single_value(self):
        """Tests the partition_classes method with a list containing a single value."""
        X = [[1, 2]]
        y = [1]
        split_attribute = 0
        split_val = 4
        X_left, X_right, y_left, y_right = self.utility.partition_classes(
            X, y, split_attribute, split_val
        )
        self.assertEqual(X_left.tolist(), [[1, 2]])
        self.assertEqual(X_right.shape, (0, 2))
        self.assertEqual(y_left.tolist(), [1])
        self.assertEqual(y_right.shape, (0,))

    def test_partition_classes_bad_type(self):
        """Tests the partition_classes method with a bad type input."""
        X = "not a list"
        y = "not a list"
        split_attribute = 0
        split_val = 4
        with self.assertRaises(TypeError):
            self.utility.partition_classes(X, y, split_attribute, split_val)

    def test_information_gain(self):
        """Tests the information_gain method with various inputs."""
        previous_y = [1, 2, 3, 4, 5]
        current_y = [[1, 2], [3, 4, 5]]
        expected_gain = self.utility.calculate_variance(previous_y) - (
            self.utility.calculate_variance(current_y[0]) * len(current_y[0])
            + self.utility.calculate_variance(current_y[1]) * len(current_y[1])
        ) / len(previous_y)
        calculated_gain = self.utility.information_gain(previous_y, current_y)
        self.assertAlmostEqual(calculated_gain, expected_gain, places=5)

    def test_information_gain_empty(self):
        """Tests the information_gain method with an empty list."""
        previous_y = []
        current_y = [[], []]
        expected_gain = 0
        calculated_gain = self.utility.information_gain(previous_y, current_y)
        self.assertEqual(calculated_gain, expected_gain)

    def test_information_gain_single_value(self):
        """Tests the information_gain method with a list containing a single value."""
        previous_y = [5, 5, 5, 5, 5]
        current_y = [[5, 5], [5, 5]]
        expected_gain = self.utility.calculate_variance(previous_y) - (
            self.utility.calculate_variance(current_y[0]) * len(current_y[0])
            + self.utility.calculate_variance(current_y[1]) * len(current_y[1])
        ) / len(previous_y)
        calculated_gain = self.utility.information_gain(previous_y, current_y)
        self.assertAlmostEqual(calculated_gain, expected_gain, places=5)

    def test_information_gain_bad_type(self):
        """Tests the information_gain method with a bad type input."""
        previous_y = "not a list"
        current_y = "not a list"
        with self.assertRaises(TypeError):
            self.utility.information_gain(previous_y, current_y)

    def test_best_split(self):
        """Tests the best_split method with various inputs."""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [1, 2, 3, 4]
        best_split = self.utility.best_split(X, y)
        self.assertIn("split_attribute", best_split)
        self.assertIn("split_val", best_split)
        self.assertIn("X_left", best_split)
        self.assertIn("X_right", best_split)
        self.assertIn("y_left", best_split)
        self.assertIn("y_right", best_split)
        self.assertIn("info_gain", best_split)
        self.assertGreaterEqual(best_split["info_gain"], 0)

    def test_best_split_empty(self):
        """Tests the best_split method with an empty list."""
        X = []
        y = []
        best_split = self.utility.best_split(X, y)
        self.assertEqual(best_split["split_attribute"], None)
        self.assertEqual(best_split["split_val"], None)
        self.assertEqual(best_split["X_left"].shape, (0, 1))
        self.assertEqual(best_split["X_right"].shape, (0, 1))
        self.assertEqual(best_split["y_left"].shape, (0,))
        self.assertEqual(best_split["y_right"].shape, (0,))

    def test_best_split_single_value(self):
        """Tests the best_split method with a list containing a single value."""
        X = [[1, 2]]
        y = [1]
        best_split = self.utility.best_split(X, y)
        self.assertEqual(best_split["split_attribute"], None)
        self.assertEqual(best_split["split_val"], None)
        self.assertEqual(best_split["X_left"].shape, (0, 2))
        self.assertEqual(best_split["X_right"].shape, (0, 2))
        self.assertEqual(best_split["y_left"].shape, (0,))
        self.assertEqual(best_split["y_right"].shape, (0,))

    def test_best_split_bad_type(self):
        """Tests the best_split method with a bad type input."""
        X = "not a list"
        y = "not a list"
        with self.assertRaises(TypeError):
            self.utility.best_split(X, y)


class TestRegressorTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Regressor Tree", end="", flush=True)

    def setUp(self):
        """Sets up the RegressorTree instance for testing."""
        self.tree = RegressorTree()

    def test_init(self):
        """Tests the initialization of the RegressorTree class."""
        self.assertEqual(self.tree.max_depth, 5)
        self.assertDictEqual(self.tree.tree, {})

    def test_learn(self):
        """Tests the learn method of the RegressorTree class."""
        X, y = synthetic_data_regression()
        self.tree.learn(X, y)
        self.assertIsInstance(self.tree.tree, dict)

    def test_learn_empty(self):
        """Tests the learn method with an empty dataset."""
        X = []
        y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.tree.learn(X, y)
        self.assertEqual(self.tree.tree, {})

    def test_learn_single_value(self):
        """Tests the learn method with a single value dataset."""
        X = [[1, 2]]
        y = [1]
        self.tree.learn(X, y)
        self.assertEqual(self.tree.tree, {})

    def test_learn_bad_type(self):
        """Tests the learn method with a bad type input."""
        X = "not a list"
        y = "not a list"
        with self.assertRaises(TypeError):
            self.tree.learn(X, y)

    def test_learn_empty_dataset(self):
        """Tests learning with an empty dataset."""
        X = []
        y = []
        tree = self.tree.learn(X, y)
        self.assertEqual(tree, {})

    def test_learn_single_data_point(self):
        """Tests learning with a single data point."""
        X = [[1, 2, 3]]
        y = [1]
        tree = self.tree.learn(X, y)
        self.assertEqual(tree, {"value": 1})

    def test_learn_pure_values(self):
        """Tests learning when all values are the same."""
        X = [[1, 2], [3, 4], [5, 6]]
        y = [1, 1, 1]
        tree = self.tree.learn(X, y)
        self.assertEqual(tree, {"value": 1})

    def test_learn_max_depth(self):
        """Tests learning when the maximum depth is reached."""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 1, 0, 1]
        self.tree.max_depth = 1
        tree = self.tree.learn(X, y)
        self.assertIn("split_attribute", tree)
        self.assertIn("split_val", tree)

    def test_predict_empty_tree(self):
        """Tests prediction with an empty tree."""
        record = [1, 2, 3]
        result = self.tree.evaluate_tree({}, record)
        self.assertIsNone(result)

    def test_predict_single_node_tree(self):
        """Tests prediction with a single-node tree."""
        tree = {"value": 1}
        record = [1, 2, 3]
        result = self.tree.evaluate_tree(tree, record)
        self.assertEqual(result, 1)

    def test_predict_with_split(self):
        """Tests prediction with a tree containing a split."""
        tree = {
            "split_attribute": 0,
            "split_val": 2,
            "left": {"value": 1},
            "right": {"value": 0},
        }
        record = [1, 2, 3]
        result = self.tree.evaluate_tree(tree, record)
        self.assertEqual(result, 1)


class TestRandomForestRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Random Forest Regressor", end="", flush=True)

    def setUp(self):
        """Sets up the RandomForestRegressor instance for testing."""
        X, y = synthetic_data_regression(n_samples=100, n_features=3)
        self.rf = RandomForestRegressor(
            X=X, y=y, forest_size=10, random_seed=0, max_depth=10
        )

    def test_init(self):
        """Tests the initialization of the RandomForestRegressor class."""
        self.assertEqual(self.rf.n_estimators, 10)
        self.assertEqual(self.rf.max_depth, 10)
        self.assertEqual(self.rf.random_state, 0)
        self.assertIsInstance(self.rf.trees, list)

    def test_fit_empty(self):
        """Tests the fit method with an empty dataset."""
        _X = []
        _y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.rf.fit()
        for tree in self.rf.trees:
            self.assertIn("split_val", tree)

    def test_fit_single_value(self):
        """Tests the fit method with a single value dataset."""
        _X = [[1, 2, 3]]
        self.rf.fit()
        for tree in self.rf.trees:
            self.assertIn("split_val", tree)

    def test_predict(self):
        """Tests the predict method of the RandomForestRegressor class."""
        self.rf.fit()
        predictions = self.rf.predict(self.rf.X)
        self.assertEqual(len(predictions), len(self.rf.X))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_empty(self):
        """Tests the predict method with an empty dataset."""
        X = []
        _y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.rf.fit()
        predictions = self.rf.predict(X)
        self.assertEqual(len(predictions), 0)

    def test_predict_single_value(self):
        """Tests the predict method with a single value dataset."""
        X = [[1, 2, 3]]
        self.rf.fit()
        predictions = self.rf.predict(X)
        self.assertEqual(len(predictions), 1)

    def test_fit(self):
        """Tests the fit method of the RandomForestRegressor class."""
        self.rf.fit(verbose=False)
        self.assertGreaterEqual(self.rf.r2, 0)
        self.assertLessEqual(self.rf.r2, 1)

    def test_get_stats(self):
        """Tests the get_stats method of the RandomForestRegressor class."""
        self.rf.fit(verbose=False)
        stats = self.rf.get_stats()
        self.assertIn("MSE", stats)
        self.assertIn("R^2", stats)
        self.assertIn("MAPE", stats)
        self.assertIn("MAE", stats)
        self.assertIn("RMSE", stats)

    def test_fit_single_data_point(self):
        """Tests fitting the RandomForestRegressor with a single data point."""
        X_single = np.random.rand(1, 5)  # Single sample, 5 features
        y_single = np.array([1])  # Single value
        self.rf.fit(X_single, y_single)
        self.assertEqual(len(self.rf.trees), 10)  # Ensure 10 trees are trained

    def test_fit_empty_dataset(self):
        """Tests fitting the RandomForestRegressor with an empty dataset."""
        X_empty = np.empty((0, 5))  # No samples, 5 features
        y_empty = np.empty((0,))
        with self.assertRaises(ValueError):
            self.rf.fit(X_empty, y_empty)

    def test_fit_no_features(self):
        """Tests fitting the RandomForestRegressor with no features."""
        X = np.empty((10, 0))  # 10 samples, 0 features
        y = np.random.randint(0, 2, size=10)
        with self.assertRaises(ValueError):
            self.rf.fit(X, y)

    def test_fit_no_samples(self):
        """Tests fitting the RandomForestRegressor with no samples."""
        X = np.empty((0, 5))  # 0 samples, 5 features
        y = np.empty((0,))
        with self.assertRaises(ValueError):
            self.rf.fit(X, y)

    def test_fit_single_class(self):
        """Tests fitting the RandomForestRegressor with a single class."""
        X = np.random.rand(10, 5)  # 10 samples, 5 features
        y = np.zeros(10)  # Single class
        self.rf.fit(X, y)
        for tree in self.rf.trees:
            self.assertEqual(tree["value"], 0)

    def test_predict_single_sample(self):
        """Tests predicting with a single sample."""
        X = [[1, 2, 3]]
        self.rf.fit()
        predictions = self.rf.predict(X)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions[0], float)

    def test_oob_predictions(self):
        """Tests out-of-bag predictions."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, size=100)
        self.rf.fit(X, y)
        oob_predictions = _predict_oob(X, self.rf.trees, self.rf.bootstraps)
        self.assertEqual(len(oob_predictions), len(X))
        self.assertIsInstance(oob_predictions, list)


class TestGradientBoostedRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Gradient Boosted Regressor", end="", flush=True)

    def setUp(self):
        """Sets up the GradientBoostedRegressor instance for testing."""
        X, y = synthetic_data_regression(n_samples=100, n_features=3)
        self.gbr = GradientBoostedRegressor(
            X, y, num_trees=10, max_depth=10, random_seed=0
        )

    def test_init(self):
        """Tests the initialization of the GradientBoostedRegressor class."""
        self.assertEqual(self.gbr.num_trees, 10)
        self.assertEqual(self.gbr.max_depth, 10)
        self.assertEqual(self.gbr.random_seed, 0)
        self.assertIsInstance(self.gbr.trees, list)
        self.assertIsInstance(self.gbr.trees[0], RegressorTree)

    def test_reset(self):
        """Tests the reset method of the GradientBoostedRegressor class."""
        self.gbr.reset()
        self.assertEqual(self.gbr.num_trees, 10)
        self.assertEqual(self.gbr.max_depth, 10)
        self.assertEqual(self.gbr.random_seed, 0)
        self.assertIsInstance(self.gbr.trees, list)
        self.assertIsInstance(self.gbr.trees[0], RegressorTree)

    def test_fit(self):
        """Tests the fit method of the GradientBoostedRegressor class."""
        self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.trees[0], dict)
        self.assertEqual(len(self.gbr.mean_absolute_residuals), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.mean_absolute_residuals[0], float)

    def test_fit_stats(self):
        """Tests the fit method with stats=False."""
        with suppress_print():
            self.gbr.fit(stats=False)
        self.assertEqual(len(self.gbr.trees), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.trees[0], dict)
        self.assertEqual(len(self.gbr.mean_absolute_residuals), self.gbr.num_trees)
        self.assertIsInstance(self.gbr.mean_absolute_residuals[0], float)

    def test_fit_empty(self):
        """Tests the fit method with an empty dataset."""
        self.gbr.X = []
        self.gbr.y = []
        with self.assertRaises(ValueError):
            self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), 10)

    def test_fit_single_value(self):
        """Tests the fit method with a single value dataset."""
        self.gbr.X = [[1, 2, 3]]
        self.gbr.y = [1]
        self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), 10)
        self.assertIsInstance(self.gbr.trees[0], dict)
        self.assertEqual(len(self.gbr.mean_absolute_residuals), 10)
        self.assertIsInstance(self.gbr.mean_absolute_residuals[0], float)

    def test_fit_bad_type(self):
        """Tests the fit method with a bad type input."""
        self.gbr.X = "not a list"
        self.gbr.y = "not a list"
        with self.assertRaises(TypeError):
            self.gbr.fit()

    def test_predict(self):
        """Tests the predict method of the GradientBoostedRegressor class."""
        self.gbr.fit()
        predictions = self.gbr.predict()
        self.assertEqual(len(predictions), len(self.gbr.X))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_empty(self):
        """Tests the predict method with an empty dataset."""
        self.gbr.X = []
        self.gbr.y = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            predictions = self.gbr.predict()
        self.assertEqual(len(predictions), 0)
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_single_value(self):
        """Tests the predict method with a single value dataset."""
        self.gbr.X = [[1, 2, 3]]
        self.gbr.y = [1]
        self.gbr.fit()
        predictions = self.gbr.predict()
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_bad_type(self):
        """Tests the predict method with a bad type input."""
        self.gbr.X = "not a list"
        self.gbr.y = "not a list"
        with self.assertRaises(TypeError):
            self.gbr.predict()

    def test_get_stats(self):
        """Tests the get_stats method of the GradientBoostedRegressor class."""
        self.gbr.fit()
        predictions = self.gbr.predict()
        stats = self.gbr.get_stats(predictions)
        self.assertIn("MSE", stats)
        self.assertIn("R^2", stats)
        self.assertIn("MAPE", stats)
        self.assertIn("MAE", stats)
        self.assertIn("RMSE", stats)


if __name__ == "__main__":
    unittest.main()
