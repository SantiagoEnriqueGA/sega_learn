import unittest
from unittest.mock import Mock
from unittest.mock import MagicMock
import logging
import sys
import os
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.linear_models import *
from test_utils import synthetic_data_regression, suppress_print

class TestOrdinaryLeastSquares(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing Ordinary Least Squares Model")
    
    def setUp(self):
        self.model = OrdinaryLeastSquares()
    
    def test_fit_predict(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.assertGreater(r2_score(y, y_pred), 0.5)
    
    def test_fit_intercept(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = OrdinaryLeastSquares(fit_intercept=True)
        self.model.fit(X, y)
        self.assertTrue(self.model.fit_intercept)
        
    def test_invalid_fit(self):
        with self.assertRaises(Exception):
            self.model.fit(None, None)
            
    def test_invalid_predict(self):
        with self.assertRaises(Exception):
            self.model.predict(None)
    
    def test_get_formula(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
    
    def test_get_formula_with_intercept(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
        
    def test_coef_(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        self.assertEqual(len(self.model.coef_), 5)
       

class TestRidge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing Ridge Regression Model")
    
    def setUp(self):
        self.model = Ridge()      
        
    def test_fit_predict(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.assertGreater(r2_score(y, y_pred), 0.5)
    
    def test_fit_intercept(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Ridge(fit_intercept=True)
        self.model.fit(X, y)
        self.assertTrue(self.model.fit_intercept)
        
    def test_fit_max_iter(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Ridge(max_iter=1000)
        self.model.fit(X, y)
        self.assertEqual(self.model.max_iter, 1000)
        
    def test_fit_tol(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Ridge(tol=0.0001)
        self.model.fit(X, y)
        self.assertEqual(self.model.tol, 0.0001)
        
    def test_invalid_fit(self):
        with self.assertRaises(Exception):
            self.model.fit(None, None)
            
    def test_invalid_predict(self):
        with self.assertRaises(Exception):
            self.model.predict(None)
    
    def test_get_formula(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
    
    def test_get_formula_with_intercept(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
        
    def test_coef_(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        self.assertEqual(len(self.model.coef_), 5)
        
class TestLasso(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing Lasso Regression Model")
    
    def setUp(self):
        self.model = Lasso()
    
    def test_fit_predict(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.assertGreater(r2_score(y, y_pred), 0.5)
    
    def test_fit_intercept(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Lasso(fit_intercept=True)
        self.model.fit(X, y)
        self.assertTrue(self.model.fit_intercept)
        
    def test_fit_max_iter(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Lasso(max_iter=1000)
        self.model.fit(X, y)
        self.assertEqual(self.model.max_iter, 1000)
        
    def test_fit_tol(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Lasso(tol=0.0001)
        self.model.fit(X, y)
        self.assertEqual(self.model.tol, 0.0001)
        
    def test_invalid_fit(self):
        with self.assertRaises(Exception):
            self.model.fit(None, None)
            
    def test_invalid_predict(self):
        with self.assertRaises(Exception):
            self.model.predict(None)
    
    def test_get_formula(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
    
    def test_get_formula_with_intercept(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
        
    def test_coef_(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        self.assertEqual(len(self.model.coef_), 5)
        
        
class TestBayesian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing Bayesian Regression Model")
    
    def setUp(self):
        self.model = Bayesian()
    
    def test_fit_predict(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.assertGreater(r2_score(y, y_pred), 0.5)
    
    def test_fit_intercept(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Bayesian(fit_intercept=True)
        self.model.fit(X, y)
        self.assertTrue(self.model.fit_intercept)
        
    def test_fit_max_iter(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Bayesian(max_iter=1000)
        self.model.fit(X, y)
        self.assertEqual(self.model.max_iter, 1000)
        
    def test_fit_tol(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = Bayesian(tol=0.0001)
        self.model.fit(X, y)
        self.assertEqual(self.model.tol, 0.0001)
        
    def test_tune(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        with suppress_print():
            alpha_1, alpha_2, lambda_1, lambda_2 = self.model.tune(X, y, beta1=0.9, beta2=0.999, iter=1000)
        self.assertIsNotNone(alpha_1)
        self.assertIsNotNone(alpha_2)
        self.assertIsNotNone(lambda_1)
        self.assertIsNotNone(lambda_2)
        
    def test_invalid_fit(self):
        with self.assertRaises(Exception):
            self.model.fit(None, None)
            
    def test_invalid_predict(self):
        with self.assertRaises(Exception):
            self.model.predict(None)
    
    def test_get_formula(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
    
    def test_get_formula_with_intercept(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
        
    def test_coef_(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        self.assertEqual(len(self.model.coef_), 5)
    
class TestRANSAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing RANSAC Regression Model")
    
    def setUp(self):
        self.model = RANSAC()
        
    def test_fit_predict(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.assertGreater(r2_score(y, y_pred), 0.5)

    def test_fit_n(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = RANSAC(n=100)
        self.model.fit(X, y)
        self.assertEqual(self.model.n, 100)
        
    def test_fit_k(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = RANSAC(k=10)
        self.model.fit(X, y)
        self.assertEqual(self.model.k, 10)
    
    def test_fit_t(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = RANSAC(t=2.0)
        self.model.fit(X, y)
        self.assertEqual(self.model.t, 2.0)
    
    def test_fit_d(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = RANSAC(d=10)
        self.model.fit(X, y)
        self.assertEqual(self.model.d, 10)
        
    def test_fit_auto_scale_t(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = RANSAC(auto_scale_t=True)
        self.model.fit(X, y)
        self.assertTrue(self.model.scale_threshold)
        
    def test_fit_auto_scale_n(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = RANSAC(auto_scale_n=True)
        self.model.fit(X, y)
        self.assertTrue(self.model.scale_n)

    def test_invalid_fit(self):
        with self.assertRaises(Exception):
            self.model.fit(None, None)
            
    def test_invalid_predict(self):
        with self.assertRaises(Exception):
            self.model.predict(None)
    
    def test_get_formula(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
    
    def test_get_formula_with_intercept(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
        

class TestPassiveAggressiveRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Testing Passive Aggressive Regressor Model")
    
    def setUp(self):
        self.model = PassiveAggressiveRegressor()
        
    def test_fit_predict(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        with suppress_print():
            self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.assertGreater(r2_score(y, y_pred), 0.5)
    
    def test_fit_all_steps(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        with suppress_print():
            self.model.fit(X, y, save_steps=True)
        self.assertTrue(len(self.model.steps_) > 0)
        
    def test_predict_all_steps(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        with suppress_print():
            self.model.fit(X, y, save_steps=True)
        y_preds = self.model.predict_all_steps(X)
        self.assertGreater(len(y_preds), 0)
        
    def test_fit_max_iter(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = PassiveAggressiveRegressor(max_iter=1000)
        with suppress_print():
            self.model.fit(X, y)
        self.assertEqual(self.model.max_iter, 1000)
        
    def test_fit_tol(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        self.model = PassiveAggressiveRegressor(tol=0.0001)
        with suppress_print():
            self.model.fit(X, y)
        self.assertEqual(self.model.tol, 0.0001)
                
    def test_invalid_fit(self):
        with self.assertRaises(Exception):
            self.model.fit(None, None)
            
    def test_invalid_predict(self):
        with self.assertRaises(Exception):
            self.model.predict(None)
    
    def test_get_formula(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        with suppress_print():
            self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
    
    def test_get_formula_with_intercept(self):
        X, y = synthetic_data_regression(n_samples=100, n_features=2, noise=0.1)
        with suppress_print():
            self.model.fit(X, y)
        formula = self.model.get_formula()
        self.assertIsNotNone(formula)
        self.assertIn('y = ', formula)
        
    def test_coef_(self):
        X, y = synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1)
        with suppress_print():
            self.model.fit(X, y)
        self.assertEqual(len(self.model.coef_), 5)
        
class TestPolynomialTransform(unittest.TestCase):
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
        
if __name__ == '__main__':
    unittest.main()