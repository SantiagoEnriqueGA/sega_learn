import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.linear_models import *

from sega_learn.nearest_neighbors.knn_classifier import KNeighborsClassifier
from sega_learn.nearest_neighbors.knn_regressor import KNeighborsRegressor

class TestKNeighborsClassifier(unittest.TestCase):
    """
    Unit test for the KNeighborsClassifier class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit: Tests the fit method of the KNeighborsClassifier class.
    - test_predict: Tests the predict method of the KNeighborsClassifier class.
    - test_invalid_n_neighbors: Tests the behavior when an invalid number of neighbors is provided.
    - test_invalid_n_neighbors_greater_than_samples: Tests the behavior when n_neighbors is greater than the number of samples.
    - test_distance_metric: Tests the behavior when an invalid distance metric is provided.
    - test_invalid_distance_metric: Tests the behavior when an invalid distance metric is provided.
    - test_data_precision: Tests the behavior when an invalid floating point precision is provided.
    - test_invalid_fp_precision: Tests the behavior when an invalid floating point precision is provided.
    - test_one_hot_encoding: Tests the behavior when one-hot encoding is applied to the input data.
    - test_invalid_one_hot_encoding: Tests the behavior when an invalid one-hot encoding is provided.
    - test_predict_with_one_hot_encoding: Tests the behavior when one-hot encoding is applied to the input data during prediction.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting KNeighborsClassifierKNeighborsBase Class", end="", flush=True)
    
    def setUp(self):
        pass    

    def test_fit(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0, 1, 0, 1])
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        # Assert that the training data and labels are stored correctly
        self.assertIsNotNone(knn.X_train)
        self.assertIsNotNone(knn.y_train)
        np.testing.assert_array_equal(knn.X_train, X_train)
        np.testing.assert_array_equal(knn.y_train, y_train)

    def test_predict(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0, 1, 0, 1])
        X_test = np.array([[1, 2], [2, 2], [3, 3]])
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        expected_predictions = np.array([0, 0, 0])
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_invalid_n_neighbors(self):
        with self.assertRaises(ValueError):
            KNeighborsClassifier(n_neighbors=0)
            
    def test_invalid_n_neighbors_greater_than_samples(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])
        knn = KNeighborsClassifier(n_neighbors=4)
        with self.assertRaises(ValueError):
            knn.fit(X_train, y_train)
            
    def test_distance_metric(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0, 1, 0, 1])
        X_test = np.array([[1, 2], [2, 2], [3, 3]])
        knn = KNeighborsClassifier(n_neighbors=3, distance_metric='euclidean')
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        expected_predictions = np.array([0, 0, 0])
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_invalid_distance_metric(self):
        knn = KNeighborsClassifier(distance_metric='invalid_metric')
        with self.assertRaises(ValueError):
            knn._compute_distances(np.array([[1, 2], [2, 3]]))
            
    def test_data_precision(self):
        knn = KNeighborsClassifier(n_neighbors=3, fp_precision=np.float32)
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float64)
        y_train = np.array([0, 1, 0, 1], dtype=np.float64)
        knn.fit(X_train, y_train)
        self.assertEqual(knn.X_train.dtype, np.float32)
        self.assertEqual(knn.y_train.dtype, np.float32)
    
    def test_invalid_fp_precision(self):
        with self.assertRaises(ValueError):
            KNeighborsClassifier(fp_precision=int)       

    def test_one_hot_encoding(self):
        X_train = np.array([[1, 'a'], [2, 'b'], [3, 'a'], [4, 'b']])
        y_train = np.array([0, 1, 0, 1])
        knn = KNeighborsClassifier(n_neighbors=3, one_hot_encode=True)
        knn.fit(X_train, y_train)
        self.assertIsNotNone(knn.X_train)
        self.assertIsNotNone(knn.y_train)
        
    def test_invalid_one_hot_encoding(self):
        with self.assertRaises(ValueError):
            KNeighborsClassifier(one_hot_encode='invalid_value')
    
    def test_predict_with_one_hot_encoding(self):
        X_train = np.array([[1, 'a'], [2, 'b'], [3, 'a'], [4, 'b']])
        y_train = np.array([0, 1, 0, 1])
        X_test = np.array([[1, 'a'], [2, 'b'], [3, 'a']])
        knn = KNeighborsClassifier(n_neighbors=3, one_hot_encode=True)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
    
    def test_predict_with_invalid_one_hot_encoding(self):
        with self.assertRaises(ValueError):
            knn = KNeighborsClassifier(n_neighbors=3, one_hot_encode='invalid_value')
        

class TestKNeighborsRegressor(unittest.TestCase):
    """
    Unit test for the KNeighborsRegressor class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit: Tests the fit method of the KNeighborsClassifier class.
    - test_predict: Tests the predict method of the KNeighborsClassifier class.
    - test_invalid_n_neighbors: Tests the behavior when an invalid number of neighbors is provided.
    - test_invalid_n_neighbors_greater_than_samples: Tests the behavior when n_neighbors is greater than the number of samples.
    - test_distance_metric: Tests the behavior when an invalid distance metric is provided.
    - test_invalid_distance_metric: Tests the behavior when an invalid distance metric is provided.
    - test_data_precision: Tests the behavior when an invalid floating point precision is provided.
    - test_invalid_fp_precision: Tests the behavior when an invalid floating point precision is provided.
    - test_one_hot_encoding: Tests the behavior when one-hot encoding is applied to the input data.
    - test_invalid_one_hot_encoding: Tests the behavior when an invalid one-hot encoding is provided.
    - test_predict_with_one_hot_encoding: Tests the behavior when one-hot encoding is applied to the input data during prediction.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting KNeighborsRegressor Class", end="", flush=True)
    
    def setUp(self):
        pass    

    def test_fit(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0.5, 1.5, 0.5, 1.5])
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)
        # Assert that the training data and labels are stored correctly
        self.assertIsNotNone(knn.X_train)
        self.assertIsNotNone(knn.y_train)
        np.testing.assert_array_equal(knn.X_train, X_train)
        np.testing.assert_array_equal(knn.y_train, y_train)

    def test_predict(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0.5, 1.5, 0.5, 1.5])
        X_test = np.array([[1, 2], [2, 2], [3, 3]])
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        expected_predictions = np.array([0.83333333, 0.83333333, 0.83333333])
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_invalid_n_neighbors(self):
        with self.assertRaises(ValueError):
            KNeighborsRegressor(n_neighbors=0)
            
    def test_invalid_n_neighbors_greater_than_samples(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0.5, 1.5, 0.5])
        knn = KNeighborsRegressor(n_neighbors=4)
        with self.assertRaises(ValueError):
            knn.fit(X_train, y_train)
            
    def test_distance_metric(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0.5, 1.5, 0.5, 1.5])
        X_test = np.array([[1, 2], [2, 2], [3, 3]])
        knn = KNeighborsRegressor(n_neighbors=3, distance_metric='euclidean')
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        expected_predictions = np.array([0.83333333, 0.83333333, 0.83333333])
        np.testing.assert_array_almost_equal(predictions, expected_predictions)

    def test_invalid_distance_metric(self):
        knn = KNeighborsRegressor(distance_metric='invalid_metric')
        with self.assertRaises(ValueError):
            knn._compute_distances(np.array([[1, 2], [2, 3]]))
            
    def test_data_precision(self):
        knn = KNeighborsRegressor(n_neighbors=3, fp_precision=np.float32)
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float64)
        y_train = np.array([0.5, 1.5, 0.5, 1.5], dtype=np.float64)
        knn.fit(X_train, y_train)
        self.assertEqual(knn.X_train.dtype, np.float32)
        self.assertEqual(knn.y_train.dtype, np.float32)
    
    def test_invalid_fp_precision(self):
        with self.assertRaises(ValueError):
            KNeighborsRegressor(fp_precision=int)       

    def test_one_hot_encoding(self):
        X_train = np.array([[1, 'a'], [2, 'b'], [3, 'a'], [4, 'b']])
        y_train = np.array([0.5, 1.5, 0.5, 1.5])
        knn = KNeighborsRegressor(n_neighbors=3, one_hot_encode=True)
        knn.fit(X_train, y_train)
        self.assertIsNotNone(knn.X_train)
        self.assertIsNotNone(knn.y_train)
        
    def test_invalid_one_hot_encoding(self):
        with self.assertRaises(ValueError):
            KNeighborsRegressor(one_hot_encode='invalid_value')
    
    def test_predict_with_one_hot_encoding(self):
        X_train = np.array([[1, 'a'], [2, 'b'], [3, 'a'], [4, 'b']])
        y_train = np.array([0.5, 1.5, 0.5, 1.5])
        X_test = np.array([[1, 'a'], [2, 'b'], [3, 'a']])
        knn = KNeighborsRegressor(n_neighbors=3, one_hot_encode=True)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
    
    def test_predict_with_invalid_one_hot_encoding(self):
        with self.assertRaises(ValueError):
            knn = KNeighborsRegressor(n_neighbors=3, one_hot_encode='invalid_value')
        

if __name__ == '__main__':
    unittest.main()