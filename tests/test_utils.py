import unittest
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.utils import *
from sega_learn.linear_models import *
from sega_learn.trees import *
from tests.utils import synthetic_data_regression, suppress_print


class TestPolynomialTransform(unittest.TestCase):
    """
    Unit test for the Polynomial Transform class.
    Methods:
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
    - test_one_hot_encode: Tests the one_hot_encode method of the Data Prep class.
    - test_one_hot_encode_multiple: Tests the one_hot_encode method with multiple columns.
    - test_write_data: Tests the write_data method of the Data
    - test_df_to_ndarray: Tests the df_to_ndarray method of the Data Prep class.
    - test_k_split: Tests the k_split method of the Data Prep class.
    - test_k_split_invalid: Tests the k_split method with invalid input.
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
        
    def test_df_to_ndarray(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]})
        X, y = DataPrep.df_to_ndarray(df, y_col=2)
        self.assertEqual(X.shape[1], 2)
        self.assertEqual(y.shape[0], 4)
        
    def test_k_split(self):
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        X_folds, y_folds = DataPrep.k_split(X, y, k=5)
        self.assertEqual(len(X_folds), 5)
        self.assertEqual(len(y_folds), 5)
        self.assertEqual(X_folds[0].shape[0], 20)
        self.assertEqual(y_folds[0].shape[0], 20)
        
    def test_k_split_invalid(self):
        with self.assertRaises(Exception):
            DataPrep.k_split(None, None, k=5)
        
class TestVotingRegressor(unittest.TestCase):
    """
    Unit test for the Voting Regressor class.
    Methods:
    - setUp: Initializes a new instance of the Voting Regressor class before each test method is run.
    - test_init: Tests the initialization of the Voting Regressor class.
    - test_predict: Tests the predict method of the Voting Regressor class.
    - test_get_params: Tests the get_params method of the Voting Regressor class.
    - test_show_models: Tests the show_models method of the Voting Regressor class.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Voting Regressor")
    
    def setUp(self):
        self.X, self.y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
        ols = OrdinaryLeastSquares()
        ols.fit(self.X, self.y)
        lasso = Lasso()
        lasso.fit(self.X, self.y)
        ridge = Ridge()
        ridge.fit(self.X, self.y)
        self.voter = VotingRegressor(models=[ols, lasso, ridge], model_weights=[0.3, 0.3, 0.4])        
        
    def test_init(self):
        self.assertEqual(len(self.voter.models), 3)
        self.assertEqual(len(self.voter.model_weights), 3)
    
    def test_predict(self):
        y_pred = self.voter.predict(self.X)
        self.assertEqual(y_pred.shape[0], self.y.shape[0])
        
    def test_get_params(self):
        params = self.voter.get_params()
        self.assertEqual(len(params), 2)
        
    def test_show_models(self):
        with suppress_print():
            self.voter.show_models()
            self.voter.show_models(formula=True)
            

class TestGridSearchCV(unittest.TestCase):
    """
    Unit test for the GridSearchCV class.
    Methods:
    - setUp: Initializes a new instance of the GridSearchCV class before each test method is run.
    - test_init: Tests the initialization of the GridSearchCV class.
    - test_param_combinations: Tests the _get_param_combinations method of the GridSearchCV class.
    - test_ols: Tests the GridSearchCV class with the Ordinary Least Squares model.
    - test_ridge: Tests the GridSearchCV class with the Ridge model.
    - test_lasso: Tests the GridSearchCV class with the Lasso model.
    - test_bayesian: Tests the GridSearchCV class with the Bayesian Ridge model.
    - test_passiveAggReg: Tests the GridSearchCV class with the Passive Aggressive Regressor model.
    - test_randomForestClassifier: Tests the GridSearchCV class with the Random Forest Classifier model.
    - test_randomForestRegressor: Tests the GridSearchCV class with the Random Forest Regressor model.
    - test_gradientBoostiedRegressor: Tests the GridSearchCV class with the Gradient Boosted Regressor model
    """
    @classmethod
    def setUpClass(cls):
        print("Testing GridSearchCV")
    
    def setUp(self):
        self.X_reg, self.y_reg = make_regression(n_samples=100, n_features=5, noise=25, random_state=42)
        self.X_class, self.y_class = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    
    def test_init(self):
        grid_search = GridSearchCV(model=Ridge, param_grid=[{'alpha': [0.1, 1, 10]}])
        self.assertEqual(grid_search.model, Ridge)
        self.assertEqual(grid_search.param_grid, [{'alpha': [0.1, 1, 10]}])
        self.assertEqual(grid_search.cv, 5)
        self.assertEqual(grid_search.metric, 'mse')
        self.assertEqual(grid_search.direction, 'minimize')
    
    def test_param_combinations(self):
        grid_search = GridSearchCV(model=Ridge, param_grid=[{'alpha': [0.1, 1, 10], 'fit_intercept': [True, False]}])
        self.assertEqual(len(grid_search.param_combinations), 6)
    
    def test_ols(self):
        ols = OrdinaryLeastSquares
        param_grid = [{'fit_intercept': [True, False]}]
        grid_search = GridSearchCV(model=ols, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_reg, self.y_reg)
        
    def test_ridge(self):
        ridge = Ridge
        param_grid = [{'alpha': [0.1, 1, 10]}]
        grid_search = GridSearchCV(model=ridge, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_reg, self.y_reg)
        
    def test_lasso(self):
        lasso = Lasso
        param_grid = [{'alpha': [0.1, 1, 10]}]
        grid_search = GridSearchCV(model=lasso, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_reg, self.y_reg)
        
    def test_bayesian(self):
        bayesian_ridge = Bayesian
        param_grid = [{'max_iter': [100, 200, 300]}]
        grid_search = GridSearchCV(model=bayesian_ridge, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_reg, self.y_reg)
        
    def test_passiveAggReg(self):
        passive_agg = PassiveAggressiveRegressor
        param_grid = [{'C': [0.1, 1, 10]}]
        grid_search = GridSearchCV(model=passive_agg, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_reg, self.y_reg)
    
    def test_randomForestClassifier(self):
        decision_tree = RandomForestClassifier
        param_grid = [{'max_depth': [3, 5, 7]}]
        grid_search = GridSearchCV(model=decision_tree, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_class, self.y_class)
        
    def test_randomForestRegressor(self):
        decision_tree = RandomForestRegressor
        param_grid = [{'max_depth': [3, 5, 7]}]
        grid_search = GridSearchCV(model=decision_tree, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_reg, self.y_reg)
        
    def test_gradientBoostiedRegressor(self):
        decision_tree = GradientBoostedRegressor
        param_grid = [{'num_trees': [50, 100, 150]}]
        grid_search = GridSearchCV(model=decision_tree, param_grid=param_grid, cv=3)
        grid_search.fit(self.X_reg, self.y_reg)
        

class TestMetrics(unittest.TestCase):
    """
    Unit test for the Metrics class.
    Methods:
    - setUp: Initializes a new instance of the Metrics class before each test method is run.
    - test_mse: Tests the mean squared error method of the Metrics class.
    - test_r2: Tests the r squared method of the Metrics class.
    - test_accuracy: Tests the accuracy method of the Metrics class.
    - test_precision: Tests the precision method of the Metrics class.
    - test_recall: Tests the recall method of the Metrics class.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Metrics")
    
    def setUp(self):
        self.y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.y_pred = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        self.metrics = Metrics()
    
    def test_mse(self):
        mse = self.metrics.mean_squared_error(self.y_true, self.y_pred)
        self.assertEqual(mse, 0.1) # MSE = 1 / 10 = 0.1
    
    def test_r2(self):
        r2 = self.metrics.r_squared(self.y_true, self.y_pred)
        self.assertAlmostEqual(r2, 0.6, places=2) # R2 = 1 - (SS_res / SS_tot) = 1 - (0.1 / 0.25)
        
    def test_accuracy(self):
        accuracy = self.metrics.accuracy(self.y_true, self.y_pred)
        self.assertEqual(accuracy, 0.9)

    def test_precision(self):
        precision = self.metrics.precision(self.y_true, self.y_pred)
        self.assertAlmostEqual(precision, (5/6), places=2) # Precision = TP / (TP + FP)  = 5 / (5 + 1)
    
    def test_recall(self):
        recall = self.metrics.recall(self.y_true, self.y_pred)
        self.assertAlmostEqual(recall, (5/5), places=2) # Recall = TP / (TP + FN) = 5 / (5 + 0)
    

if __name__ == '__main__':
    unittest.main()