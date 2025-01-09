import unittest
import sys
import os

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.trees import ClassifierTreeUtility as ctu
from sega_learn.trees import ClassifierTree as ct
from sega_learn.trees import RegressorTreeUtility as rtu
from sega_learn.trees import RegressorTree as rt
from sega_learn.trees import RandomForestClassifier as rfc
from sega_learn.trees import RandomForestRegressor as rfr
from sega_learn.trees import GradientBoostedRegressor as gbr

from sega_learn.trees import *

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
        print("Testing Imports - Trees")
    
    def test_individual_imports(self):
        assert ctu is not None
        assert ct is not None
        assert rtu is not None
        assert rt is not None
        assert rfc is not None
        assert rfr is not None
        assert gbr is not None        
       
    def test_wildcard_import(self):
        assert ClassifierTreeUtility is not None
        assert ClassifierTree is not None
        assert RegressorTreeUtility is not None
        assert RegressorTree is not None
        assert RandomForestClassifier is not None
        assert RandomForestRegressor is not None
        assert GradientBoostedRegressor is not None

if __name__ == '__main__':
    unittest.main()