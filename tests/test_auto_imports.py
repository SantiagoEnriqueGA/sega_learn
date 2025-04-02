import unittest
import sys
import os

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.auto import AutoRegressor as reg
from sega_learn.auto import AutoClassifier as clf

from sega_learn.auto import *

class TestImportsLinear(unittest.TestCase):
    """
    Tests that the linear_models subpackage can be imported correctly.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Imports - Auto", end="", flush=True)
    
    def test_individual_imports(self):
        assert reg is not None
        assert clf is not None
       
    def test_wildcard_import(self):
        assert AutoRegressor is not None
        assert AutoClassifier is not None

if __name__ == '__main__':
    unittest.main()