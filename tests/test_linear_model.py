import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared
accuracy_score = Metrics.accuracy

from sega_learn.linear_models import *
from sega_learn.linear_models import make_sample_data
from tests.utils import synthetic_data_regression, suppress_print

class TestOrdinaryLeastSquares(unittest.TestCase):
    """
    Unit test for the Ordinary Least Squares regression class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit_predict: Tests the fit and predict methods of the Ordinary Least Squares class.
    - test_fit_intercept: Tests the fit_intercept parameter of the Ordinary Least Squares class.
    - test_invalid_fit: Tests the fit method with invalid input.
    - test_invalid_predict: Tests the predict method with invalid input.
    - test_get_formula: Tests the get_formula method of the Ordinary Least Squares class.
    - test_get_formula_with_intercept: Tests the get_formula method with intercept of the Ordinary Least Squares class.
    - test_coef_: Tests the coef_ attribute of the Ordinary Least Squares class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Ordinary Least Squares Model", end="", flush=True)
    
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
    """
    Unit test for the Ridge regression class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit_predict: Tests the fit and predict methods of the Ridge class.
    - test_fit_intercept: Tests the fit_intercept parameter of the Ridge class.
    - test_fit_max_iter: Tests the max_iter parameter of the Ridge class.
    - test_fit_tol: Tests the tol parameter of the Ridge class.
    - test_invalid_fit: Tests the fit method with invalid input.
    - test_invalid_predict: Tests the predict method with invalid input.
    - test_get_formula: Tests the get_formula method of the Ridge class.
    - test_get_formula_with_intercept: Tests the get_formula method with intercept of the Ridge class.
    - test_coef_: Tests the coef_ attribute of the Ridge class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Ridge Regression Model", end="", flush=True)
    
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
    """
    Unit test for the Lasso regression class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit_predict: Tests the fit and predict methods of the Lasso class.
    - test_fit_intercept: Tests the fit_intercept parameter of the Lasso class.
    - test_fit_max_iter: Tests the max_iter parameter of the Lasso class.
    - test_fit_tol: Tests the tol parameter of the Lasso class.
    - test_invalid_fit: Tests the fit method with invalid input.
    - test_invalid_predict: Tests the predict method with invalid input.
    - test_get_formula: Tests the get_formula method of the Lasso class.
    - test_get_formula_with_intercept: Tests the get_formula method with intercept of the Lasso class.
    - test_coef_: Tests the coef_ attribute of the Lasso class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Lasso Regression Model", end="", flush=True)
    
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
    """
    Unit test for the Bayesian regression class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit_predict: Tests the fit and predict methods of the Bayesian class.
    - test_fit_intercept: Tests the fit_intercept parameter of the Bayesian class.
    - test_fit_max_iter: Tests the max_iter parameter of the Bayesian class.
    - test_fit_tol: Tests the tol parameter of the Bayesian class.
    - test_tune: Tests the tune method of the Bayesian class.
    - test_invalid_fit: Tests the fit method with invalid input.
    - test_invalid_predict: Tests the predict method with invalid input.
    - test_get_formula: Tests the get_formula method of the Bayesian class.
    - test_get_formula_with_intercept: Tests the get_formula method with intercept of the Bayesian class.
    - test_coef_: Tests the coef_ attribute of the Bayesian class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Bayesian Regression Model", end="", flush=True)
    
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
    """
    Unit test for the RANSAC regression class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit_predict: Tests the fit and predict methods of the RANSAC class.
    - test_fit_n: Tests the n parameter of the RANSAC class.
    - test_fit_k: Tests the k parameter of the RANSAC class.
    - test_fit_t: Tests the t parameter of the RANSAC class.
    - test_fit_d: Tests the d parameter of the RANSAC class.
    - test_fit_auto_scale_t: Tests the auto_scale_t parameter of the RANSAC class.
    - test_fit_auto_scale_n: Tests the auto_scale_n parameter of the RANSAC class.
    - test_invalid_fit: Tests the fit method with invalid input.
    - test_invalid_predict: Tests the predict method with invalid input.
    - test_get_formula: Tests the get_formula method of the RANSAC class.
    - test_get_formula_with_intercept: Tests the get_formula method with intercept of the RANSAC class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting RANSAC Regression Model", end="", flush=True)
    
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
    """
    Unit test for the Passive Aggressive Regressor class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_fit_predict: Tests the fit and predict methods of the Passive Aggressive Regressor class.
    - test_fit_all_steps: Tests the fit method with save_steps parameter of the Passive Aggressive Regressor class.
    - test_predict_all_steps: Tests the predict_all_steps method of the Passive Aggressive Regressor class.
    - test_fit_max_iter: Tests the max_iter parameter of the Passive Aggressive Regressor class.
    - test_fit_tol: Tests the tol parameter of the Passive Aggressive Regressor class.
    - test_invalid_fit: Tests the fit method with invalid input.
    - test_invalid_predict: Tests the predict method with invalid input.
    - test_get_formula: Tests the get_formula method of the Passive Aggressive Regressor class.
    - test_get_formula_with_intercept: Tests the get_formula method with intercept of the Passive Aggressive Regressor class.
    - test_coef_: Tests the coef_ attribute of the Passive Aggressive Regressor class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Passive Aggressive Regressor Model", end="", flush=True)
    
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
  
        
class TestLinearDiscriminantAnalysis(unittest.TestCase):
    """
    Unit test for the Linear Discriminant Analysis class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_lda: Tests the fit and predict methods of the Linear Discriminant Analysis class.
    - test_lda_svd: Tests the fit and predict methods with svd solver of the Linear Discriminant Analysis class.
    - test_lda_lsqr: Tests the fit and predict methods with lsqr solver of the Linear Discriminant Analysis class.
    - test_lda_eigen: Tests the fit and predict methods with eigen solver of the Linear Discriminant Analysis class.
    - test_lda_bad_solver: Tests the fit method with invalid solver of the Linear Discriminant Analysis class.
    - test_lda_no_solver: Tests the fit method with no solver of the Linear Discriminant Analysis class.
    - test_lda_no_data: Tests the fit method with invalid input.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Linear Discriminant Analysis", end="", flush=True)
        
    def setUp(self):
        self.cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0    # Covariance matrix for class 1, scaled by 2.0
        self.cov_class_2 = self.cov_class_1.T                           # Covariance matrix for class 2, same as class 1 but transposed
        
        # Generate data
        self.X, self.y = make_sample_data(n_samples=1000, n_features=2, cov_class_1=self.cov_class_1, cov_class_2=self.cov_class_2, shift=[4,1], seed=1)
        
    def test_lda(self):
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.X, self.y)
        y_pred = lda.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreaterEqual(acc, 0.50)
    
    def test_lda_svd(self):
        lda = LinearDiscriminantAnalysis(solver='svd')
        lda.fit(self.X, self.y)
        y_pred = lda.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreater(acc, 0)
    
    def test_lda_lsqr(self):
        lda = LinearDiscriminantAnalysis(solver='lsqr')
        lda.fit(self.X, self.y)
        y_pred = lda.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreater(acc, 0)
    
    def test_lda_eigen(self):
        lda = LinearDiscriminantAnalysis(solver='eigen')
        lda.fit(self.X, self.y)
        y_pred = lda.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreater(acc, 0)
        
    def test_lda_bad_solver(self):
        lda = LinearDiscriminantAnalysis(solver='bad_solver')
        with self.assertRaises(ValueError):
            lda.fit(self.X, self.y)
            
    def test_lda_no_solver(self):
        lda = LinearDiscriminantAnalysis(solver=None)
        with self.assertRaises(ValueError):
            lda.fit(self.X, self.y)
            
    def test_lda_no_data(self):
        lda = LinearDiscriminantAnalysis()
        with self.assertRaises(Exception):
            lda.fit(None, None)
    
class TestQuadraticDiscriminantAnalysis(unittest.TestCase):
    """
    Unit test for the Quadratic Discriminant Analysis class.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_qda: Tests the fit and predict methods of the Quadratic Discriminant Analysis class.
    - test_qda_prior: Tests the fit and predict methods with priors parameter of the Quadratic Discriminant Analysis class.
    - test_qda_reg_param: Tests the fit and predict methods with reg_param parameter of the Quadratic Discriminant Analysis class.
    - test_qda_bad_reg_param: Tests the fit method with invalid reg_param parameter of the Quadratic Discriminant Analysis class.
    - test_qda_no_data: Tests the fit method with invalid input.
    - test_qda_no_priors: Tests the fit method with no priors parameter of the Quadratic Discriminant Analysis class.
    - test_qda_bad_priors: Tests the fit method with invalid priors parameter of the Quadratic Discriminant Analysis class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Quadratic Discriminant Analysis", end="", flush=True)
        
    def setUp(self):
        self.cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0    # Covariance matrix for class 1, scaled by 2.0
        self.cov_class_2 = self.cov_class_1.T                           # Covariance matrix for class 2, same as class 1 but transposed
        
        # Generate data
        self.X, self.y = make_sample_data(n_samples=1000, n_features=2, cov_class_1=self.cov_class_1, cov_class_2=self.cov_class_2, shift=[4,1], seed=1)
        
    def test_qda(self):
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(self.X, self.y)
        y_pred = qda.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreaterEqual(acc, 0.50)
        
    def test_qda_prior(self):
        qda = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5])
        self.y = self.y.astype(int)  # Ensure class labels are integers
        qda.fit(self.X, self.y)
        y_pred = qda.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreaterEqual(acc, 0.50)
    
    def test_qda_reg_param(self):
        qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
        qda.fit(self.X, self.y)
        y_pred = qda.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        self.assertGreaterEqual(acc, 0.50)
        
    def test_qda_bad_reg_param(self):
        with self.assertRaises(Exception):
            qda = QuadraticDiscriminantAnalysis(reg_param=-0.1)
            qda.fit(self.X, self.y)
        
    def test_qda_no_data(self):
        qda = QuadraticDiscriminantAnalysis()
        with self.assertRaises(Exception):
            qda.fit(None, None)
            
    def test_qda_no_priors(self):
        qda = QuadraticDiscriminantAnalysis(priors=None)
        qda.fit(self.X, self.y)    
        self.assertEqual(qda.priors, None)   
        
    def test_qda_bad_priors(self):
        with self.assertRaises(Exception):
            qda = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5, 0.5])
            qda.fit(self.X, self.y)
        
if __name__ == '__main__':
    unittest.main()