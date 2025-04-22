import os
import sys
import unittest

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.time_series import *


class TestImportsTimeSeries(unittest.TestCase):
    """Tests that the time_series subpackage can be imported correctly.

    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting Imports - Time Series", end="", flush=True)

    def test_individual_imports(self):
        """Tests that each module in the segadb package can be imported individually."""
        from sega_learn.time_series import ARIMA as arima
        from sega_learn.time_series import SARIMA as sarima
        from sega_learn.time_series import SARIMAX as sarimax

        assert arima is not None
        assert sarima is not None
        assert sarimax is not None

    def test_wildcard_import(self):
        """Tests that the segadb package can be imported using a wildcard import."""
        assert ARIMA is not None
        assert SARIMA is not None
        assert SARIMAX is not None

    def test_ARIMA(self):  # NOQA D201
        from sega_learn.time_series import ARIMA

        assert ARIMA is not None


if __name__ == "__main__":
    unittest.main()
