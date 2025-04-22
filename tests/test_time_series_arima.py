import os
import sys
import unittest
import warnings

import numpy as np

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.time_series.arima import ARIMA


class TestARIMA(unittest.TestCase):
    """Unit test suite for the ARIMA class."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting ARIMA", end="", flush=True)

    def setUp(self):  # NOQA D201
        self.time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.order = (2, 1, 2)
        self.arima = ARIMA(order=self.order)

    def test_initialization(self):
        """Test ARIMA initialization."""
        self.assertEqual(self.arima.order, self.order)
        self.assertEqual(self.arima.p, self.order[0])
        self.assertEqual(self.arima.d, self.order[1])
        self.assertEqual(self.arima.q, self.order[2])
        self.assertIsNone(self.arima.model)
        self.assertIsNone(self.arima.fitted_model)

    def test_invalid_order_negative_value(self):
        """Test ARIMA initialization with a negative value in the order."""
        with self.assertRaises(ValueError):
            ARIMA(order=(1, -1, 1))

    def test_invalid_order_missing_parameter(self):
        """Test ARIMA initialization with a missing parameter in the order."""
        with self.assertRaises(ValueError):
            ARIMA(order=(1, 1))

    def test_invalid_order_invalid_type(self):
        """Test ARIMA initialization with an invalid type for the order."""
        with self.assertRaises(ValueError):
            ARIMA(order="invalid")

    def test_invalid_order_extra_parameter(self):
        """Test ARIMA initialization with an extra parameter in the order."""
        with self.assertRaises(ValueError):
            ARIMA(order=(1, 1, 1, 1))

    def test_fit(self):
        """Test the fit method."""
        self.arima.fit(self.time_series)
        self.assertIsNotNone(self.arima.fitted_model)
        self.assertTrue("ar_coefficients" in self.arima.fitted_model)
        self.assertTrue("ma_coefficients" in self.arima.fitted_model)

    def test_fit_invalid_data(self):
        """Test fitting with invalid data."""
        with self.assertRaises(ValueError):
            self.arima.fit("invalid_data")

    def test_forecast(self):
        """Test the forecast method."""
        self.arima.fit(self.time_series)
        forecasted_values = self.arima.forecast(steps=3)
        self.assertEqual(len(forecasted_values), 3)
        self.assertTrue(isinstance(forecasted_values, np.ndarray))

    def test_forecast_without_fit(self):
        """Test forecasting without fitting the model."""
        with self.assertRaises(ValueError):
            self.arima.forecast(steps=3)

    def test_suggest_order(self):
        """Test the suggest_order static method."""
        suggested_order = ARIMA.suggest_order(self.time_series)
        self.assertEqual(len(suggested_order), 3)
        self.assertTrue(all(isinstance(i, int) for i in suggested_order))

    def test_suggest_order_invalid_data(self):
        """Test suggest_order with invalid data."""
        with self.assertRaises(ValueError):
            ARIMA.suggest_order("invalid_data")

    def test_suggest_order_empty_data(self):
        """Test suggest_order with empty data."""
        with self.assertRaises(ValueError):
            ARIMA.suggest_order([])

    def test_find_best_order(self):
        """Test the find_best_order static method."""
        train_series = self.time_series[:7]
        test_series = self.time_series[7:]
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        best_order = ARIMA.find_best_order(train_series, test_series)
        self.assertEqual(len(best_order), 3)
        self.assertTrue(all(isinstance(i, int) for i in best_order))

    def test_find_best_order_invalid_data(self):
        """Test find_best_order with invalid data."""
        with self.assertRaises(ValueError):
            ARIMA.find_best_order("invalid_data", "invalid_data")

    def test_find_best_order_empty_data(self):
        """Test find_best_order with empty data."""
        with self.assertRaises(ValueError):
            ARIMA.find_best_order([], [])

    def test_find_best_order_subset(self):
        """Test find_best_order with a subset of the time series."""
        train_series = self.time_series[:5]
        test_series = self.time_series[5:]
        best_order = ARIMA.find_best_order(train_series, test_series, subset_size=0.5)
        self.assertEqual(len(best_order), 3)
        self.assertTrue(all(isinstance(i, int) for i in best_order))

    def test_find_best_order_invalid_subset(self):
        """Test find_best_order with an invalid subset size."""
        train_series = self.time_series[:5]
        test_series = self.time_series[5:]
        with self.assertRaises(ValueError):
            ARIMA.find_best_order(train_series, test_series, subset_size=1.5)
        with self.assertRaises(ValueError):
            ARIMA.find_best_order(train_series, test_series, subset_size=-5)
        with self.assertRaises(ValueError):
            ARIMA.find_best_order(train_series, test_series, subset_size=0)


if __name__ == "__main__":
    unittest.main()
