import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.utils import *
from tests.utils import synthetic_data_regression, suppress_print

       
class TestPolynomialTransform(unittest.TestCase):
    """
    Unit test for the Polynomial Transform class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit_transform: Tests the fit_transform method of the Polynomial Transform class.
    - test_fit: Tests the fit method of the Polynomial Transform class.
    - test_invalid_fit: Tests the fit method with invalid input.
    - test_invalid_transform: Tests the transform method with invalid input.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Polynomial Transform")
    
    def setUp(self):
        self.transform = PolynomialTransform(degree=2)
        
    def test_fit_transform(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=2, noise=0.1)
        X_transformed = self.transform.fit_transform(X)
        self.assertEqual(X_transformed.shape[1], 6)
    
    def test_fit(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=2, noise=0.1)
        self.transform.fit(X)
        self.assertEqual(self.transform.degree, 2)
    
    def test_invalid_fit(self):
        with self.assertRaises(Exception):
            self.transform.fit(None)
    
    def test_invalid_transform(self):
        with self.assertRaises(Exception):
            self.transform.transform(None)      
            
class TestDataPrep(unittest.TestCase):
    """
    Unit test for the Data Prep class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each
    - test_one_hot_encode: Tests the one_hot_encode method of the Data Prep class.
    - test_one_hot_encode_multiple: Tests the one_hot_encode method with multiple columns.
    - test_write_data: Tests the write_data method of the Data
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Data Prep")
        
    def test_one_hot_encode(self):
        # DF with one categorical column (col 3)
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': ['a', 'b', 'a', 'b']})
        df_encoded = DataPrep.one_hot_encode(df, [2])
        self.assertEqual(df_encoded.shape[1], 4)
        
    def test_one_hot_encode_multiple(self):
        # DF with two categorical columns (col 2 and 3)
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'a', 'b'], 'C': ['x', 'y', 'x', 'y']})
        df_encoded = DataPrep.one_hot_encode(df, [1, 2])
        self.assertEqual(df_encoded.shape[1], 5)
        
    def test_write_data(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        DataPrep.write_data(df, 'test.csv')
        self.assertTrue(os.path.exists('test.csv'))
        os.remove('test.csv')     
        
if __name__ == '__main__':
    unittest.main()