import unittest
import os
import sys
import time

from utils import suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.linear_models import *
from sega_learn.linear_models import make_data
from utils import synthetic_data_regression, suppress_print

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

NUM_SAMPLES = 1_000_000
NUM_RUNS = 5
    
class TestLinearModelPerformance(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        self.X, self.y = synthetic_data_regression(n_samples=NUM_SAMPLES, n_features=5, noise=0.1, random_state=42)
        self.X_test, self.y_test = synthetic_data_regression(n_samples=NUM_SAMPLES, n_features=5, noise=0.1, random_state=42)

    def measure_performance(self, model_class_sklearn, model_class_sega):
        def model_performance(model_class):
            start_time = time.time()
            model = model_class()
            model.fit(self.X, self.y)
            y_pred = model.predict(self.X_test)
            end_time = time.time()
            return end_time - start_time

        sklearn_times = [model_performance(model_class_sklearn) for _ in range(NUM_RUNS)]
        sega_times = [model_performance(model_class_sega) for _ in range(NUM_RUNS)]

        avg_sklearn_time = sum(sklearn_times) / NUM_RUNS
        avg_sega_time = sum(sega_times) / NUM_RUNS

        return avg_sklearn_time, avg_sega_time

    def test_ols(self):
        avg_sklearn_time, avg_sega_time = self.measure_performance(linear_model.LinearRegression, OrdinaryLeastSquares)
        
        print(f"\nOLS Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn Linear Regression: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega Linear Regression: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10     
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
        
    
    def test_ridge(self):
        # Use only NUM_SAMPLES / 100 for Ridge test 
        original_X = self.X
        original_y = self.y
        reduced_sample_size = NUM_SAMPLES // 100
        self.X = self.X[:reduced_sample_size]
        self.y = self.y[:reduced_sample_size]
        
        avg_sklearn_time, avg_sega_time = self.measure_performance(linear_model.Ridge, Ridge)
        
        print(f"\nRidge Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn Ridge Regression: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega Ridge Regression: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10     
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
            
        # Restore original data
        self.X = original_X
        self.y = original_y
                        
    def test_lasso(self):
        # Use only NUM_SAMPLES / 100 for Lasso test
        original_X = self.X
        original_y = self.y
        reduced_sample_size = NUM_SAMPLES // 100
        self.X = self.X[:reduced_sample_size]
        self.y = self.y[:reduced_sample_size]
        
        avg_sklearn_time, avg_sega_time = self.measure_performance(linear_model.Lasso, Lasso)
        
        print(f"\nLasso Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn Lasso Regression: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega Lasso Regression: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10     
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
        
        # Restore original data
        self.X = original_X
        self.y = original_y
            
    def test_bayesian(self):
        avg_sklearn_time, avg_sega_time = self.measure_performance(linear_model.BayesianRidge, Bayesian)
        
        print(f"\nBayesian Ridge Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn Bayesian Ridge Regression: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega Bayesian Ridge Regression: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10     
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
            
    def test_ransac(self):
        avg_sklearn_time, avg_sega_time = self.measure_performance(linear_model.RANSACRegressor, RANSAC)
        
        print(f"\nRANSAC Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn RANSAC Regression: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega RANSAC Regression: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
            
    def test_passiveAggressiveRegressor(self):
        # Use only NUM_SAMPLES / 100 for Passive Aggressive test
        original_X = self.X
        original_y = self.y
        reduced_sample_size = NUM_SAMPLES // 100
        self.X = self.X[:reduced_sample_size]
        self.y = self.y[:reduced_sample_size]
        
        with suppress_print():
            avg_sklearn_time, avg_sega_time = self.measure_performance(linear_model.PassiveAggressiveRegressor, PassiveAggressiveRegressor)
        
        print(f"\nPassive Aggressive Performance comparison on {reduced_sample_size:,} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn Passive Aggressive Regression: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega Passive Aggressive Regression: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10     
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
        
        # Restore original data
        self.X = original_X
        self.y = original_y        
                
if __name__ == '__main__':
    unittest.main()