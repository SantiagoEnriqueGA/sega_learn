import os
import sys
import unittest

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.nearest_neighbors import *
from sega_learn.nearest_neighbors import KNeighborsClassifier as knc
from sega_learn.nearest_neighbors import KNeighborsRegressor as knr


class TestImportsNearestNeighbors(unittest.TestCase):
    """
    Tests that the clustering subpackage can be imported correctly.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """

    @classmethod
    def setUpClass(cls):
        print("\nTesting Imports - Nearest Neighbors", end="", flush=True)

    def test_individual_imports(self):
        assert knc is not None
        assert knr is not None

    def test_wildcard_import(self):
        assert KNeighborsClassifier is not None
        assert KNeighborsRegressor is not None


if __name__ == "__main__":
    unittest.main()
