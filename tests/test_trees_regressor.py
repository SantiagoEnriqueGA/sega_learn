import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.trees import *
from tests.utils import suppress_print, synthetic_data_regression


class TestRegressorTreeUtility(unittest.TestCase):
    """Set up the RegressorTreeUtility instance for testing."""

    @classmethod
    def setUpClass(cls):
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Regressor Tree Utility", end="", flush=True)

    def setUp(self):
        """Sets up the RegressorTreeUtility instance for testing."""
        X, y = synthetic_data_regression()
        self.utility = RegressorTreeUtility(X, y, min_samples_split=2, n_features=5)

    # TODO: Reformat all tests for RegressorTreeUtility
    def test_calculate_variance(self):
        """Tests the calculate_variance method with various inputs."""
        # y = [1, 2, 3, 4, 5]
        # expected_variance = np.var(y)
        # calculated_variance = self.utility.calculate_variance(y)
        # self.assertAlmostEqual(calculated_variance, expected_variance, places=5)

    def test_calculate_variance_empty(self):
        """Tests the calculate_variance method with an empty list."""
        # y = []
        # expected_variance = 0
        # calculated_variance = self.utility.calculate_variance(y)
        # self.assertEqual(calculated_variance, expected_variance)

    def test_calculate_variance_single_value(self):
        """Tests the calculate_variance method with a list containing a single value."""
        # y = [5, 5, 5, 5, 5]
        # expected_variance = 0
        # calculated_variance = self.utility.calculate_variance(y)
        # self.assertEqual(calculated_variance, expected_variance)

    def test_calculate_variance_negative_values(self):
        """Tests the calculate_variance method with a list containing negative values."""
        # y = [-1, -2, -3, -4, -5]
        # expected_variance = np.var(y)
        # calculated_variance = self.utility.calculate_variance(y)
        # self.assertAlmostEqual(calculated_variance, expected_variance, places=5)

    def test_calculate_variance_bad_type(self):
        """Tests the calculate_variance method with a bad type input."""
        # y = "not a list"
        # with self.assertRaises(TypeError):
        #     self.utility.calculate_variance(y)

    def test_partition_classes(self):
        """Tests the partition_classes method with various inputs."""
        # X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        # y = [1, 2, 3, 4]
        # split_attribute = 0
        # split_val = 4
        # X_left, X_right, y_left, y_right = self.utility.partition_classes(
        #     X, y, split_attribute, split_val
        # )
        # self.assertEqual(X_left.tolist(), [[1, 2], [3, 4]])
        # self.assertEqual(X_right.tolist(), [[5, 6], [7, 8]])
        # self.assertEqual(y_left.tolist(), [1, 2])
        # self.assertEqual(y_right.tolist(), [3, 4])

    def test_partition_classes_empty(self):
        """Tests the partition_classes method with an empty list."""
        # X = []
        # y = []
        # split_attribute = 0
        # split_val = 4
        # X_left, X_right, y_left, y_right = self.utility.partition_classes(
        #     X, y, split_attribute, split_val
        # )
        # self.assertEqual(X_left.shape, (0, 1))
        # self.assertEqual(X_right.shape, (0, 1))
        # self.assertEqual(y_left.shape, (0,))
        # self.assertEqual(y_right.shape, (0,))

    def test_partition_classes_single_value(self):
        """Tests the partition_classes method with a list containing a single value."""
        # X = [[1, 2]]
        # y = [1]
        # split_attribute = 0
        # split_val = 4
        # X_left, X_right, y_left, y_right = self.utility.partition_classes(
        #     X, y, split_attribute, split_val
        # )
        # self.assertEqual(X_left.tolist(), [[1, 2]])
        # self.assertEqual(X_right.shape, (0, 2))
        # self.assertEqual(y_left.tolist(), [1])
        # self.assertEqual(y_right.shape, (0,))

    def test_partition_classes_bad_type(self):
        """Tests the partition_classes method with a bad type input."""

    #     X = "not a list"
    #     y = "not a list"
    #     split_attribute = 0
    #     split_val = 4
    #     with self.assertRaises(TypeError):
    #         self.utility.partition_classes(X, y, split_attribute, split_val)

    def test_information_gain(self):
        """Tests the information_gain method with various inputs."""

    #     previous_y = [1, 2, 3, 4, 5]
    #     current_y = [[1, 2], [3, 4, 5]]
    #     parent_variance = self.utility.calculate_variance(previous_y)
    #     expected_gain = parent_variance - (
    #         self.utility.calculate_variance(current_y[0]) * len(current_y[0])
    #         + self.utility.calculate_variance(current_y[1]) * len(current_y[1])
    #     ) / len(previous_y)
    #     calculated_gain = self.utility.information_gain(parent_variance, current_y)
    #     self.assertAlmostEqual(calculated_gain, expected_gain, places=5)

    def test_information_gain_empty(self):
        """Tests the information_gain method with an empty list."""

    #     previous_y = []
    #     current_y = [[], []]
    #     parent_variance = self.utility.calculate_variance(previous_y)
    #     expected_gain = 0
    #     calculated_gain = self.utility.information_gain(parent_variance, current_y)
    #     self.assertEqual(calculated_gain, expected_gain)

    def test_information_gain_single_value(self):
        """Tests the information_gain method with a list containing a single value."""

    #     previous_y = [5, 5, 5, 5, 5]
    #     current_y = [[5, 5], [5, 5]]
    #     parent_variance = self.utility.calculate_variance(previous_y)
    #     expected_gain = parent_variance - (
    #         self.utility.calculate_variance(current_y[0]) * len(current_y[0])
    #         + self.utility.calculate_variance(current_y[1]) * len(current_y[1])
    #     ) / len(previous_y)
    #     calculated_gain = self.utility.information_gain(parent_variance, current_y)
    #     self.assertAlmostEqual(calculated_gain, expected_gain, places=5)

    def test_information_gain_bad_type(self):
        """Tests the information_gain method with a bad type input."""

    #     parent_variance = 1  # Dummy value since the input is invalid
    #     current_y = "not a list"
    #     with self.assertRaises(TypeError):
    #         self.utility.information_gain(parent_variance, current_y)

    def test_best_split(self):
        """Tests the best_split method with various inputs."""

    #     X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    #     y = [1, 2, 3, 4]
    #     best_split = self.utility.best_split(X, y)
    #     self.assertIn("split_attribute", best_split)
    #     self.assertIn("split_val", best_split)
    #     self.assertIn("X_left", best_split)
    #     self.assertIn("X_right", best_split)
    #     self.assertIn("y_left", best_split)
    #     self.assertIn("y_right", best_split)
    #     self.assertIn("info_gain", best_split)
    #     self.assertGreaterEqual(best_split["info_gain"], 0)

    def test_best_split_empty(self):
        """Tests the best_split method with an empty list."""

    #     X = []
    #     y = []
    #     best_split = self.utility.best_split(X, y)
    #     self.assertEqual(best_split["split_attribute"], None)
    #     self.assertEqual(best_split["split_val"], None)
    #     self.assertEqual(best_split["X_left"].shape, (0, 1))
    #     self.assertEqual(best_split["X_right"].shape, (0, 1))
    #     self.assertEqual(best_split["y_left"].shape, (0,))
    #     self.assertEqual(best_split["y_right"].shape, (0,))

    def test_best_split_single_value(self):
        """Tests the best_split method with a list containing a single value."""

    #     X = [[1, 2]]
    #     y = [1]
    #     best_split = self.utility.best_split(X, y)
    #     self.assertEqual(best_split["split_attribute"], None)
    #     self.assertEqual(best_split["split_val"], None)
    #     self.assertEqual(best_split["X_left"].shape, (0, 2))
    #     self.assertEqual(best_split["X_right"].shape, (0, 2))
    #     self.assertEqual(best_split["y_left"].shape, (0,))
    #     self.assertEqual(best_split["y_right"].shape, (0,))

    def test_best_split_bad_type(self):
        """Tests the best_split method with a bad type input."""

    #     X = "not a list"
    #     y = "not a list"
    #     with self.assertRaises(TypeError):
    #         self.utility.best_split(X, y)


class TestRegressorTree(unittest.TestCase):
    """Set up the RegressorTree instance for testing."""

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

    def test_fit(self):
        """Tests the fit method of the RegressorTree class."""
        X, y = synthetic_data_regression()
        self.tree.fit(X, y)
        self.assertIsInstance(self.tree.tree, dict)

    def test_fit_empty(self):
        """Tests the fit method with an empty dataset."""
        X = []
        y = []
        with self.assertRaises(IndexError):
            self.tree.fit(X, y)

    def test_fit_single_value(self):
        """Tests the fit method with a single value dataset."""
        X = [[1, 2]]
        y = [1]
        self.tree.fit(X, y)
        self.assertEqual(self.tree.tree, {"value": 1})

    def test_fit_bad_type(self):
        """Tests the fit method with a bad type input."""
        X = "not a list"
        y = "not a list"
        with self.assertRaises(IndexError):
            self.tree.fit(X, y)

    def test_fit_empty_dataset(self):
        """Tests fitting with an empty dataset."""
        X = []
        y = []
        with self.assertRaises(IndexError):
            self.tree.fit(X, y)

    def test_fit_single_data_point(self):
        """Tests fitting with a single data point."""
        X = [[1, 2, 3]]
        y = [1]
        self.tree.fit(X, y)
        self.assertEqual(self.tree.tree, {"value": 1})

    def test_fit_pure_values(self):
        """Tests fitting when all values are the same."""
        X = [[1, 2], [3, 4], [5, 6]]
        y = [1, 1, 1]
        self.tree.fit(X, y)
        self.assertEqual(self.tree.tree, {"value": 1})

    def test_fit_max_depth(self):
        """Tests fitting when the maximum depth is reached."""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 1, 0, 1]
        self.tree.max_depth = 1
        self.tree.fit(X, y)
        self.assertIn("feature_idx", self.tree.tree)
        self.assertIn("threshold", self.tree.tree)

    def test_predict_empty_tree(self):
        """Tests prediction with an empty tree."""
        record = [1, 2, 3]
        result = self.tree.predict(record)
        for val in result:
            self.assertEqual(np.isnan(val), True)

    def test_predict_single_node_tree(self):
        """Tests prediction with a single-node tree."""
        tree = {"value": 1}
        self.tree.tree = tree
        record = [1, 2, 3]
        result = self.tree.predict(record)
        for val in result:
            self.assertEqual(val, 1)

    def test_predict_with_split(self):
        """Tests prediction with a tree containing a split."""
        tree = {
            "feature_idx": 0,
            "threshold": 2,
            "left": {"value": 1},
            "right": {"value": 1},
        }
        self.tree.tree = tree
        record = [1, 2, 3]
        result = self.tree.predict(record)
        for val in result:
            self.assertEqual(val, 1)


class TestRandomForestRegressor(unittest.TestCase):
    """Set up the RandomForestRegressor instance for testing."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Random Forest Regressor", end="", flush=True)

    def setUp(self):  # NOQA D201
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
        X = []
        y = []
        with self.assertRaises(ValueError):
            self.rf.fit(X, y)

    def test_fit_single_value(self):
        """Tests the fit method with a single value dataset."""
        X = [[1, 2, 3]]
        y = [1]
        self.rf.fit(X, y)
        self.assertEqual(len(self.rf.trees), 10)
        self.assertIsInstance(self.rf.trees[0], RegressorTree)

    def test_predict(self):
        """Tests the predict method of the RandomForestRegressor class."""
        self.rf.fit()
        predictions = self.rf.predict(self.rf.X)
        self.assertEqual(len(predictions), len(self.rf.X))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_empty(self):
        """Tests the predict method with an empty dataset."""
        X = [[1, 2, 3]]
        y = [1]
        self.rf.fit(X, y)
        with self.assertRaises(ValueError):
            self.rf.predict([])

    def test_predict_single_value(self):
        """Tests the predict method with a single value dataset."""
        X = [[1, 2, 3]]
        y = [1]
        self.rf.fit(X, y)
        predictions = self.rf.predict(X)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions[0], float)

    def test_fit(self):
        """Tests the fit method of the RandomForestRegressor class."""
        self.rf.fit(verbose=False)
        self.assertIsInstance(self.rf.trees, list)
        self.assertEqual(len(self.rf.trees), self.rf.n_estimators)

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
        self.rf.fit(X, y)
        self.assertEqual(len(self.rf.trees), 10)
        self.assertIsInstance(self.rf.trees[0], RegressorTree)

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
        self.assertEqual(len(self.rf.trees), 10)
        self.assertIsInstance(self.rf.trees[0], RegressorTree)

    def test_predict_single_sample(self):
        """Tests predicting with a single sample."""
        X = [[1, 2, 3]]
        self.rf.fit()
        predictions = self.rf.predict(X)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions[0], float)


class TestGradientBoostedRegressor(unittest.TestCase):
    """Set up the GradientBoostedRegressor instance for testing."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        """Initializes a new instance of the Index class before each test method is run."""
        print("\nTesting Gradient Boosted Regressor", end="", flush=True)

    def setUp(self):  # NOQA D201
        """Sets up the GradientBoostedRegressor instance for testing."""
        X, y = synthetic_data_regression(n_samples=100, n_features=3)
        self.gbr = GradientBoostedRegressor(
            X, y, num_trees=10, max_depth=10, random_seed=0
        )

    def test_init(self):
        """Tests the initialization of the GradientBoostedRegressor class."""
        self.assertEqual(self.gbr.n_estimators, 10)
        self.assertEqual(self.gbr.max_depth, 10)
        self.assertEqual(self.gbr.random_state, 0)
        self.assertIsInstance(self.gbr.trees, list)

    def test_fit(self):
        """Tests the fit method of the GradientBoostedRegressor class."""
        self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), self.gbr.n_estimators)
        self.assertIsInstance(self.gbr.trees[0], RegressorTree)
        self.assertEqual(len(self.gbr.mean_absolute_residuals_), self.gbr.n_estimators)
        self.assertIsInstance(self.gbr.mean_absolute_residuals_[0], float)

    def test_fit_stats(self):
        """Tests the fit method with verbose output."""
        with suppress_print():
            self.gbr.fit(verbose=True)
        self.assertEqual(len(self.gbr.trees), self.gbr.n_estimators)
        self.assertIsInstance(self.gbr.trees[0], RegressorTree)
        self.assertEqual(len(self.gbr.mean_absolute_residuals_), self.gbr.n_estimators)
        self.assertIsInstance(self.gbr.mean_absolute_residuals_[0], float)

    def test_fit_empty(self):
        """Tests the fit method with an empty dataset."""
        self.gbr.X = []
        self.gbr.y = []
        with self.assertRaises(ValueError):
            self.gbr.fit()

    def test_fit_single_value(self):
        """Tests the fit method with a single value dataset."""
        self.gbr.X = [[1, 2, 3]]
        self.gbr.y = [1]
        self.gbr.fit()
        self.assertEqual(len(self.gbr.trees), 10)
        self.assertIsInstance(self.gbr.trees[0], RegressorTree)
        self.assertEqual(len(self.gbr.mean_absolute_residuals_), 10)
        self.assertIsInstance(self.gbr.mean_absolute_residuals_[0], float)

    def test_fit_bad_type(self):
        """Tests the fit method with a bad type input."""
        self.gbr.X = "not a list"
        self.gbr.y = "not a list"
        with self.assertRaises((TypeError, ValueError)):
            self.gbr.fit()

    def test_predict(self):
        """Tests the predict method of the GradientBoostedRegressor class."""
        self.gbr.fit()
        predictions = self.gbr.predict(self.gbr.X)
        self.assertEqual(len(predictions), len(self.gbr.X))
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_empty(self):
        """Tests the predict method with an empty dataset."""
        self.gbr.X = []
        self.gbr.y = []
        with self.assertRaises(RuntimeError):
            self.gbr.predict(self.gbr.X)

    def test_predict_single_value(self):
        """Tests the predict method with a single value dataset."""
        self.gbr.X = [[1, 2, 3]]
        self.gbr.y = [1]
        self.gbr.fit()
        predictions = self.gbr.predict(self.gbr.X)
        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_bad_type(self):
        """Tests the predict method with a bad type input."""
        self.gbr.fit()
        self.gbr.X = "not a list"
        self.gbr.y = "not a list"
        with self.assertRaises((TypeError, ValueError)):
            self.gbr.predict(self.gbr.X)

    def test_get_stats(self):
        """Tests the get_stats method of the GradientBoostedRegressor class."""
        self.gbr.fit()
        predictions = self.gbr.predict(self.gbr.X)
        stats = self.gbr.get_stats(self.gbr.y, predictions)
        self.assertIn("MSE", stats)
        self.assertIn("R^2", stats)
        self.assertIn("MAPE", stats)
        self.assertIn("MAE", stats)
        self.assertIn("RMSE", stats)


if __name__ == "__main__":
    unittest.main()
