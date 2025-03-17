import random
import unittest
import sys
import os
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.utils import *
from sega_learn.linear_models import *
from sega_learn.trees import *
from tests.utils import synthetic_data_regression, suppress_print
from sega_learn.utils import make_regression, make_classification
from sega_learn.utils import train_test_split


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
        print("\nTesting Polynomial Transform", end="", flush=True)
    
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
        print("\nTesting Data Prep", end="", flush=True)
        
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
        
    def test_one_hot_encode_invalid(self):
        # DF with no categorical columns
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        df_encoded = DataPrep.one_hot_encode(df, [])
        self.assertEqual(df_encoded.shape[1], 2)
        
    def test_one_hot_encode_empty(self):
        # Empty DataFrame
        df = pd.DataFrame()
        df_encoded = DataPrep.one_hot_encode(df, [])
        self.assertEqual(df_encoded.shape[1], 0)
        
    def test_one_hot_encode_invalid_col(self):
        # DF with invalid column index
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        with self.assertRaises(Exception):
            DataPrep.one_hot_encode(df, [2])
            
    def test_one_hot_encode_dtype_pd(self):
        # DF with non-numeric column
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'a', 'b']})
        df_encoded = DataPrep.one_hot_encode(df, [1])
        self.assertIsInstance(df_encoded, pd.DataFrame)
    
    def test_one_hot_encode_dtype_np(self):
        # Numpy array with one categorical column (col 1)
        data = np.array([[1, 'a'], [2, 'b'], [3, 'a'], [4, 'b']])
        data_encoded = DataPrep.one_hot_encode(data, [1])
        self.assertIsInstance(data_encoded, np.ndarray)
                    
    def test_find_categorical_columns(self):
        # DF with one categorical column (col 2)
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'a', 'b'], 'C': [5, 6, 7, 8]})
        categorical_cols = DataPrep.find_categorical_columns(df)
        self.assertEqual(categorical_cols, [1])
        
    def test_find_categorical_columns_multiple(self):
        # DF with two categorical columns (col 1 and 2)
        df = pd.DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': ['x', 'y', 'x', 'y'], 'C': [5, 6, 7, 8]})
        categorical_cols = DataPrep.find_categorical_columns(df)
        self.assertEqual(categorical_cols, [0, 1])
        
    def test_find_categorical_columns_empty(self):
        # Empty DataFrame
        df = pd.DataFrame()
        categorical_cols = DataPrep.find_categorical_columns(df)
        self.assertEqual(categorical_cols, [])
        
    def test_find_categorical_columns_invalid(self):
        # DF with no categorical columns
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        categorical_cols = DataPrep.find_categorical_columns(df)
        self.assertEqual(categorical_cols, [])
        
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
        print("\nTesting Voting Regressor", end="", flush=True)
    
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

class TestModelSelectionUtils(unittest.TestCase):
    """
    Unit tests for the Utility functions in the Model Selection module.
    Methods:
    - setUp: Initializes a new instance of the Model Selection Utility class before each test method is run.
    - test_get_param_combinations: Tests the get_param_combinations method of the Model Selection Utility class.
    - test_get_param_combinations_invalid: Tests the get_param_combinations method with invalid input.
    - test_get_param_combinations_empty: Tests the get_param_combinations method with empty input.
    - test_get_param_combinations_single: Tests the get_param_combinations method with a single parameter.
    - test_cross_validate: Tests the cross_validate method of the Model Selection Utility class.
    - test_cross_validate_invalid: Tests the cross_validate method with invalid input.
    - test_cross_validate_invalid_cv: Tests the cross_validate method with invalid cv.
    - test_cross_validate_invalid_params: Tests the cross_validate method with invalid params.
    - test_cross_validate_invalid_params_type: Tests the cross_validate method with invalid params type.
    - test_cross_validate_cv_1: Tests the cross_validate method with cv=1.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Model Selection Utils", end="", flush=True)
        
    def setUp(self):
        self.X, self.y = make_regression(n_samples=100, n_features=5, noise=25, random_state=42)
        self.num_tests = 100
    
    def test_get_param_combinations(self):
        param_grid = [{'alpha': [0.1, 1, 10], 'fit_intercept': [True, False]}]
        param_combinations = ModelSelectionUtility.get_param_combinations(param_grid)
        self.assertEqual(len(param_combinations), 6)
        
    def test_get_param_combinations_invalid(self):
        with self.assertRaises(Exception):
            ModelSelectionUtility.get_param_combinations(None)
            
    def test_get_param_combinations_empty(self):
        with self.assertRaises(Exception):
            ModelSelectionUtility.get_param_combinations([])
            
    def test_get_param_combinations_single(self):
        param_grid = [{'alpha': [0.1, 1, 10]}]
        param_combinations = ModelSelectionUtility.get_param_combinations(param_grid)
        self.assertEqual(len(param_combinations), 3)
        
    def test_cross_validate(self):
        ols = OrdinaryLeastSquares
        mse_scores, _ = ModelSelectionUtility.cross_validate(ols, self.X, self.y, params={'fit_intercept': [True]}, cv=5)
        self.assertEqual(len(mse_scores), 5)
        for score in mse_scores:
            self.assertIsInstance(score, float)
    
    def test_cross_validate_invalid(self):
        with self.assertRaises(Exception):
            ModelSelectionUtility.cross_validate(None, None, None, params=None, cv=5)
            
    def test_cross_validate_invalid_cv(self):
        with self.assertRaises(Exception):
            ModelSelectionUtility.cross_validate(OrdinaryLeastSquares, self.X, self.y, params=None, cv=0)
            
    def test_cross_validate_invalid_params(self):
        with self.assertRaises(Exception):
            ModelSelectionUtility.cross_validate(OrdinaryLeastSquares, self.X, self.y, params=None, cv=5)
            
    def test_cross_validate_invalid_params_type(self):
        with self.assertRaises(Exception):
            ModelSelectionUtility.cross_validate(OrdinaryLeastSquares, self.X, self.y, params='params', cv=5)
            
    def test_cross_validate_cv_1(self):
        ols = OrdinaryLeastSquares
        with self.assertRaises(Exception):
            mse_scores, _ = ModelSelectionUtility.cross_validate(ols, self.X, self.y, params={'fit_intercept': [True]}, cv=1)
          
          
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
    - test_invalid_param_grid: Tests the GridSearchCV class with invalid param_grid
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting GridSearchCV", end="", flush=True)
    
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
        
    def test_invalid_param_grid(self):
        with self.assertRaises(Exception):
            grid_search = GridSearchCV(model=Ridge, param_grid=None)
            grid_search.fit(self.X_reg, self.y_reg)

class TestRandomSearchCV(unittest.TestCase):
    """
    Unit test for the RandomSearchCV class.
    Methods:
    - setUp: Initializes a new instance of the RandomSearchCV_ class before each test method is run.
    - test_init: Tests the initialization of the RandomSearchCV_ class.
    - test_param_combinations: Tests the _get_param_combinations method of the RandomSearchCV_ class.
    - test_ols: Tests the RandomSearchCV_ class with the Ordinary Least Squares model.
    - test_ridge: Tests the RandomSearchCV_ class with the Ridge model.
    - test_lasso: Tests the RandomSearchCV_ class with the Lasso model.
    - test_bayesian: Tests the RandomSearchCV_ class with the Bayesian Ridge model.
    - test_passiveAggReg: Tests the RandomSearchCV_ class with the Passive Aggressive Regressor model.
    - test_randomForestClassifier: Tests the RandomSearchCV_ class with the Random Forest Classifier model.
    - test_randomForestRegressor: Tests the RandomSearchCV_ class with the Random Forest Regressor model.
    - test_gradientBoostiedRegressor: Tests the RandomSearchCV_ class with the Gradient Boosted Regressor model
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting RandomSearchCV", end="", flush=True)
    
    def setUp(self):
        self.X_reg, self.y_reg = make_regression(n_samples=100, n_features=5, noise=25, random_state=42)
        self.X_class, self.y_class = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    
    def test_init(self):
        rand_search = RandomSearchCV(model=Ridge, param_grid=[{'alpha': [0.1, 1, 10]}], iter=3)
        self.assertEqual(rand_search.model, Ridge)
        self.assertEqual(rand_search.param_grid, [{'alpha': [0.1, 1, 10]}])
        self.assertEqual(rand_search.cv, 5)
        self.assertEqual(rand_search.metric, 'mse')
        self.assertEqual(rand_search.direction, 'minimize')
    
    def test_param_combinations(self):
        rand_search = RandomSearchCV(model=Ridge, param_grid=[{'alpha': [0.1, 1, 10], 'fit_intercept': [True, False]}], iter=2)
        self.assertEqual(len(rand_search.param_combinations), 6)
    
    def test_ols(self):
        ols = OrdinaryLeastSquares
        param_grid = [{'fit_intercept': [True, False]}]
        rand_search = RandomSearchCV(model=ols, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_reg, self.y_reg)
        
    def test_ridge(self):
        ridge = Ridge
        param_grid = [{'alpha': [0.1, 1, 10]}]
        rand_search = RandomSearchCV(model=ridge, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_reg, self.y_reg)
        
    def test_lasso(self):
        lasso = Lasso
        param_grid = [{'alpha': [0.1, 1, 10]}]
        rand_search = RandomSearchCV(model=lasso, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_reg, self.y_reg)
        
    def test_bayesian(self):
        bayesian_ridge = Bayesian
        param_grid = [{'max_iter': [100, 200, 300]}]
        rand_search = RandomSearchCV(model=bayesian_ridge, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_reg, self.y_reg)
        
    def test_passiveAggReg(self):
        passive_agg = PassiveAggressiveRegressor
        param_grid = [{'C': [0.1, 1, 10]}]
        rand_search = RandomSearchCV(model=passive_agg, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_reg, self.y_reg)
    
    def test_randomForestClassifier(self):
        decision_tree = RandomForestClassifier
        param_grid = [{'max_depth': [3, 5, 7]}]
        rand_search = RandomSearchCV(model=decision_tree, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_class, self.y_class)
        
    def test_randomForestRegressor(self):
        decision_tree = RandomForestRegressor
        param_grid = [{'max_depth': [3, 5, 7]}]
        rand_search = RandomSearchCV(model=decision_tree, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_reg, self.y_reg)
        
    def test_gradientBoostiedRegressor(self):
        decision_tree = GradientBoostedRegressor
        param_grid = [{'num_trees': [50, 100, 150]}]
        rand_search = RandomSearchCV(model=decision_tree, param_grid=param_grid, cv=3, iter=2)
        rand_search.fit(self.X_reg, self.y_reg)
        
    def test_invalid_iter(self):
        with self.assertRaises(Exception):
            rand_search = RandomSearchCV(model=Ridge, param_grid=[{'alpha': [0.1, 1, 10]}], iter=0)
            rand_search.fit(self.X_reg, self.y_reg)
    
    def test_iter_larger_than_param_combinations(self):
        with suppress_print():
            rand_search = RandomSearchCV(model=Ridge, param_grid=[{'alpha': [0.1, 1, 10]}], iter=100)
            rand_search.fit(self.X_reg, self.y_reg)
        
    def test_invalid_param_grid(self):
        with self.assertRaises(Exception):
            rand_search = RandomSearchCV(model=Ridge, param_grid=None, iter=3)
            rand_search.fit(self.X_reg, self.y_reg)

class TestMetrics(unittest.TestCase):
    """
    Unit test for the Metrics class. Runs 100 tests for each method. Each test generates random data.
    Methods:
    - setUp: Initializes a new instance of the Metrics class before each test method is run.
    - test_mse: Tests the mean squared error method of the Metrics class.
    - test_r2: Tests the r squared method of the Metrics class.
    - test_mae: Tests the mean absolute error method of the Metrics class.
    - test_rmse: Tests the root mean squared error method of the Metrics class.
    - test_mape: Tests the mean absolute percentage error method of the Metrics class.
    - test_mpe: Tests the mean percentage error method of the Metrics class.
    - test_accuracy: Tests the accuracy method of the Metrics class.
    - test_precision: Tests the precision method of the Metrics class.
    - test_recall: Tests the recall method of the Metrics class.
    - test_f1_score: Tests the f1 score method of the Metrics class.
    - test_log_loss: Tests the log loss method of the Metrics class.
    - test_confusion_matrix: Tests the confusion matrix method of the Metrics class.
    - test_show_confusion_matrix: Tests the show confusion matrix method of the Metrics class.
    - test_classification_report: Tests the classification report method of the Metrics class.
    - test_show_classification_report: Tests the show classification report method of the Metrics class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Metrics", end="", flush=True)
        cls.num_tests = 100  # Define the variable for the number of tests
    
    def setUp(self):
        self.metrics = Metrics()
    
    def generate_regression_data(self):
        y_true, y_pred = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=None)
        return y_true.flatten(), y_pred.flatten()
    
    def generate_classification_data(self):
        X, y_true = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=None)
        y_pred = np.random.randint(0, 2, size=y_true.shape)
        y_pred_prob = np.random.rand(100, 2)
        y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)
        return y_true, y_pred, y_pred_prob
    
    # Regression Metrics
    def test_mse(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred = self.generate_regression_data()
                mse = self.metrics.mean_squared_error(y_true, y_pred)
                sk_mse = sk_metrics.mean_squared_error(y_true, y_pred)
                self.assertEqual(mse, sk_mse)
    
    def test_r2(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred = self.generate_regression_data()
                r2 = self.metrics.r_squared(y_true, y_pred)
                sk_r2 = sk_metrics.r2_score(y_true, y_pred)
                self.assertAlmostEqual(r2, sk_r2, places=4) 
        
    def test_mae(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred = self.generate_regression_data()
                mae = self.metrics.mean_absolute_error(y_true, y_pred)
                sk_mae = sk_metrics.mean_absolute_error(y_true, y_pred)
                self.assertEqual(mae, sk_mae) 
        
    def test_rmse(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred = self.generate_regression_data()
                rmse = self.metrics.root_mean_squared_error(y_true, y_pred)
                sk_rmse = np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))
                self.assertAlmostEqual(rmse, sk_rmse, places=4)
        
    def test_mape(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred = self.generate_regression_data()
                mape = self.metrics.mean_absolute_percentage_error(y_true, y_pred)
                sk_mape = sk_metrics.mean_absolute_percentage_error(y_true, y_pred)
                self.assertEqual(mape, sk_mape) 
        
    def test_mpe(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred = self.generate_regression_data()
                mpe = self.metrics.mean_percentage_error(y_true, y_pred)
                sk_mpe = np.mean((y_true - y_pred) / y_true)
                self.assertEqual(mpe, sk_mpe) 
        
    # Classification Metrics
    def test_accuracy(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                accuracy = self.metrics.accuracy(y_true, y_pred)
                sk_accuracy = sk_metrics.accuracy_score(y_true, y_pred)
                self.assertEqual(accuracy, sk_accuracy)

    def test_precision(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                precision = self.metrics.precision(y_true, y_pred)
                sk_precision = sk_metrics.precision_score(y_true, y_pred)
                self.assertAlmostEqual(precision, sk_precision, places=4) 
    
    def test_recall(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                recall = self.metrics.recall(y_true, y_pred)
                sk_recall = sk_metrics.recall_score(y_true, y_pred)
                self.assertAlmostEqual(recall, sk_recall, places=4) 
    
    def test_f1_score(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                f1 = self.metrics.f1_score(y_true, y_pred)
                sk_f1 = sk_metrics.f1_score(y_true, y_pred)
                self.assertAlmostEqual(f1, sk_f1, places=4) 
    
    def test_log_loss(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, _, y_pred_prob = self.generate_classification_data()
                log_loss = self.metrics.log_loss(y_true, y_pred_prob)
                sk_log_loss = sk_metrics.log_loss(y_true, y_pred_prob)
                self.assertAlmostEqual(log_loss, sk_log_loss, places=4)

    def test_confusion_matrix(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                cm = self.metrics.confusion_matrix(y_true, y_pred)
                sk_cm = sk_metrics.confusion_matrix(y_true, y_pred)
                self.assertTrue(np.array_equal(cm, sk_cm))
                
    def test_show_confusion_matrix(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                with suppress_print():
                    self.metrics.show_confusion_matrix(y_true, y_pred)
                    
    def test_classification_report(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                report = self.metrics.classification_report(y_true, y_pred)
                sk_report = sk_metrics.classification_report(y_true, y_pred, output_dict=True)
                for cls in report.keys():
                    self.assertAlmostEqual(report[cls]['recall'], sk_report[str(cls)]['recall'], places=4)
                    self.assertAlmostEqual(report[cls]['precision'], sk_report[str(cls)]['precision'], places=4)
                    self.assertAlmostEqual(report[cls]['f1-score'], sk_report[str(cls)]['f1-score'], places=4)
                    self.assertAlmostEqual(report[cls]['support'], sk_report[str(cls)]['support'], places=4)
                                                        
    def test_show_classification_report(self):
        for _ in range(self.num_tests):
            with self.subTest(i=_):
                y_true, y_pred, _ = self.generate_classification_data()
                with suppress_print():
                    self.metrics.show_classification_report(y_true, y_pred)


class TestDataAugmentation(unittest.TestCase):
    """
    Unit test for the Data Augmentation class.
    Methods:
    - setUp: Initializes a new instance of the Data Augmentation class before each test method is run.
    - test_random_over_sampler: Tests the Random Over Sampler method of the Data Augmentation class.
    - test_smote: Tests the SMOTE method of the Data Augmentation class.
    - test_random_under_sampler: Tests the Random Under Sampler method of the Data Augmentation class.
    - test_smote_with_force_equal: Tests the SMOTE method with force_equal parameter.
    - test_augment_with_empty_techniques: Tests the augment method with an empty list of techniques.
    - test_augment_with_invalid_techniques: Tests the augment method with invalid techniques.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Data Augmentation", end="", flush=True)
    
    def setUp(self):
        self.X, self.y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.7, 0.3], random_state=42, class_sep=.5)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.ros = RandomOverSampler(random_state=42)
        self.smote = SMOTE(random_state=42)
        self.rus = RandomUnderSampler(random_state=42)
        self.augmenter = Augmenter(techniques=[self.ros, self.smote, self.rus], verbose=False)
        
    def test_random_over_sampler(self):       
        X_resampled, y_resampled = self.ros.fit_resample(self.X_train, self.y_train)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))
        self.assertEqual(X_resampled.shape[0], 2 * np.sum(y_resampled == 0))
    
    def test_random_over_sampler_invalid(self):
        with self.subTest("Invalid input"):
            with self.assertRaises(Exception):
                self.ros.fit_resample(None, None)
        with self.subTest("Invalid input type"):
            with self.assertRaises(Exception):
                self.ros.fit_resample("invalid_input", "invalid_input")
        with self.subTest("Invalid input shape"):
            with self.assertRaises(Exception):
                self.ros.fit_resample(np.random.rand(101, 20), np.random.rand(100, 20))
    
    def test_random_over_sampler_invalid_params(self):
        with self.assertRaises(Exception):
            ros = RandomOverSampler(random_state=42, sampling_strategy='invalid_strategy')
            ros.fit_resample(self.X_train, self.y_train)
            
    def test_fit_random_over_sampler_invalid(self):
        with self.assertRaises(Exception):
            ros = RandomOverSampler(random_state=42)
            ros.fit(None, None)
    
    def test_fit_random_over_sampler_invalid_params(self):
        with self.assertRaises(Exception):
            ros = RandomOverSampler(random_state=42)
            ros.fit(self.X_train, None)
    
    def test_smote(self):
        X_resampled, y_resampled = self.smote.fit_resample(self.X_train, self.y_train)
        self.assertGreaterEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))
        
    def test_smote_equal(self):
        X_resampled, y_resampled = self.smote.fit_resample(self.X_train, self.y_train, force_equal=True)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))

    def test_smote_invalid(self):
        with self.subTest("Invalid input"):
            with self.assertRaises(Exception):
                self.smote.fit_resample(None, None)
        with self.subTest("Invalid input type"):
            with self.assertRaises(Exception):
                self.smote.fit_resample("invalid_input", "invalid_input")
        with self.subTest("Invalid input shape"):
            with self.assertRaises(Exception):
                self.smote.fit_resample(np.random.rand(101, 20), np.random.rand(100, 20))
    
    def test_random_under_sampler(self):
        X_resampled, y_resampled = self.rus.fit_resample(self.X_train, self.y_train)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))
        self.assertEqual(X_resampled.shape[0], 2 * np.sum(y_resampled == 0))
    
    def test_random_under_sampler_invalid(self):
        with self.subTest("Invalid input"):
            with self.assertRaises(Exception):
                self.rus.fit_resample(None, None)
        with self.subTest("Invalid input type"):
            with self.assertRaises(Exception):
                self.rus.fit_resample("invalid_input", "invalid_input")
        with self.subTest("Invalid input shape"):
            with self.assertRaises(Exception):
                self.rus.fit_resample(np.random.rand(101, 20), np.random.rand(100, 20))
    
    def test_random_under_sampler_invalid_params(self):
        with self.assertRaises(Exception):
            rus = RandomUnderSampler(random_state=42, sampling_strategy='invalid_strategy')
            rus.fit_resample(self.X_train, self.y_train)
            
    def test_fit_random_under_sampler_invalid(self):
        with self.assertRaises(Exception):
            rus = RandomUnderSampler(random_state=42)
            rus.fit(None, None)
    
    def test_fit_random_under_sampler_invalid_params(self):
        with self.assertRaises(Exception):
            rus = RandomUnderSampler(random_state=42)
            rus.fit(self.X_train, None)
       
    def test_augment(self):
        X_resampled, y_resampled = self.augmenter.augment(self.X, self.y)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))
        
    def test_augment_with_multiple_techniques(self):
        augmenter = Augmenter(techniques=[self.ros, self.smote], verbose=False)
        X_resampled, y_resampled = augmenter.augment(self.X, self.y)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))
        
    def test_augment_with_one_technique(self):
        augmenter = Augmenter(techniques=[self.rus], verbose=False)
        X_resampled, y_resampled = augmenter.augment(self.X, self.y)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))
    
    def test_augment_with_invalid_technique(self):
        invalid_technique = "invalid_technique"
        with self.assertRaises(Exception):
            augmenter = Augmenter(techniques=[invalid_technique], verbose=False)
            augmenter.augment(self.X, self.y)
    
    def test_augment_with_invalid_input(self):
        with self.assertRaises(Exception):
            self.augmenter.augment(None, None)
        
    def test_augment_with_balanced_data(self):
        X_balanced, y_balanced = self.ros.fit_resample(self.X, self.y)
        X_resampled, y_resampled = self.augmenter.augment(X_balanced, y_balanced)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))
    
    def test_smote_with_force_equal(self):
        X_resampled, y_resampled = self.smote.fit_resample(self.X_train, self.y_train, force_equal=True)
        self.assertEqual(np.sum(y_resampled == 0), np.sum(y_resampled == 1))

    def test_augment_with_empty_techniques(self):
        augmenter = Augmenter(techniques=[], verbose=False)
        X_resampled, y_resampled = augmenter.augment(self.X, self.y)
        self.assertEqual(X_resampled.shape, self.X.shape)
        self.assertEqual(y_resampled.shape, self.y.shape)

    def test_augment_with_invalid_techniques(self):
        invalid_technique = "invalid_technique"
        with self.assertRaises(ValueError):
            augmenter = Augmenter(techniques=[self.rus, invalid_technique], verbose=False)
            augmenter.augment(self.X, self.y)

class TestDataDecomposition(unittest.TestCase):
    """
    Unit test for the Decomposition class.
    Methods:
    - setUp: Initializes a new instance of the Decomposition class before each test method is run.
    - test_pca_fit_transform: Tests the fit_transform method of the PCA class.
    - test_pca_inverse_transform: Tests the inverse_transform method of the PCA class.
    - test_svd_fit_transform: Tests the fit_transform method of the SVD class.
    - test_svd_get_singular_values: Tests the get_singular_values method of the SVD class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Decomposition", end="", flush=True)
    
    def setUp(self):
        self.X = np.random.rand(100, 5)
    
    def test_pca_fit_transform(self):
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(self.X)
        self.assertEqual(X_transformed.shape[1], 2)
        self.assertEqual(pca.get_components().shape[1], 2)
    
    def test_svd_fit_transform(self):
        svd = SVD(n_components=2)
        X_transformed = svd.fit_transform(self.X)
        self.assertEqual(X_transformed.shape[1], 2)
        self.assertEqual(svd.get_singular_values().shape[0], 2)
    
    def test_svd_get_singular_values(self):
        svd = SVD(n_components=2)
        svd.fit(self.X)
        singular_values = svd.get_singular_values()
        self.assertEqual(singular_values.shape[0], 2)

    def test_pca_invalid_input(self):
        pca = PCA(n_components=2)
        with self.assertRaises(ValueError):
            pca.fit("invalid_input")
        with self.assertRaises(ValueError):
            pca.fit(np.random.rand(100))
        with self.assertRaises(ValueError):
            pca.fit(np.random.rand(100, 1))
        with self.assertRaises(ValueError):
            pca.transform("invalid_input")
        with self.assertRaises(ValueError):
            pca.transform(np.random.rand(100))
        with self.assertRaises(ValueError):
            pca.inverse_transform("invalid_input")
        with self.assertRaises(ValueError):
            pca.inverse_transform(np.random.rand(100))

    def test_svd_invalid_input(self):
        svd = SVD(n_components=2)
        with self.assertRaises(ValueError):
            svd.fit("invalid_input")
        with self.assertRaises(ValueError):
            svd.fit(np.random.rand(100))
        with self.assertRaises(ValueError):
            svd.fit(np.random.rand(1, 100))
        with self.assertRaises(ValueError):
            svd.transform("invalid_input")
        with self.assertRaises(ValueError):
            svd.transform(np.random.rand(100))

if __name__ == '__main__':
    unittest.main()