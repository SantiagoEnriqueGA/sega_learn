import unittest
import sys
import os

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.linear_models import Utility as utl
from sega_learn.linear_models import OrdinaryLeastSquares as ols
from sega_learn.linear_models import Ridge as rdg
from sega_learn.linear_models import Lasso as lso
from sega_learn.linear_models import Bayesian as byn
from sega_learn.linear_models import RANSAC as rns
from sega_learn.linear_models import PassiveAggressiveRegressor as par
from sega_learn.linear_models import PolynomialTransform as plt
from sega_learn.linear_models import LinearDiscriminantAnalysis as lda
from sega_learn.linear_models import QuadraticDiscriminantAnalysis as qda
from sega_learn.linear_models import make_data as mdt

from sega_learn.linear_models import *

class TestImports(unittest.TestCase):
    """
    Tests that the segadb package can be imported.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Imports - Linear Models")
    
    def test_individual_imports(self):
        assert utl is not None
        assert ols is not None
        assert rdg is not None
        assert lso is not None
        assert byn is not None
        assert rns is not None
        assert par is not None
        assert plt is not None
        assert lda is not None
        assert qda is not None
        assert mdt is not None
       
    def test_wildcard_import(self):
        assert Utility is not None
        assert OrdinaryLeastSquares is not None
        assert Ridge is not None
        assert Lasso is not None
        assert Bayesian is not None
        assert RANSAC is not None
        assert PassiveAggressiveRegressor is not None
        assert PolynomialTransform is not None
        assert LinearDiscriminantAnalysis is not None
        assert QuadraticDiscriminantAnalysis is not None

if __name__ == '__main__':
    unittest.main()