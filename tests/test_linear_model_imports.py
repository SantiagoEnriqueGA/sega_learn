import os
import sys
import unittest

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.linear_models import *
from sega_learn.linear_models import RANSAC as rns
from sega_learn.linear_models import Bayesian as byn
from sega_learn.linear_models import Lasso as lso
from sega_learn.linear_models import LinearDiscriminantAnalysis as lda
from sega_learn.linear_models import OrdinaryLeastSquares as ols
from sega_learn.linear_models import PassiveAggressiveRegressor as par
from sega_learn.linear_models import QuadraticDiscriminantAnalysis as qda
from sega_learn.linear_models import Ridge as rdg
from sega_learn.linear_models import make_sample_data as mdt


class TestImportsLinear(unittest.TestCase):
    """Tests that the linear_models subpackage can be imported correctly.

    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting Imports - Linear Models", end="", flush=True)

    def test_individual_imports(self):  # NOQA D201
        assert ols is not None
        assert rdg is not None
        assert lso is not None
        assert byn is not None
        assert rns is not None
        assert par is not None
        assert lda is not None
        assert qda is not None
        assert mdt is not None

    def test_wildcard_import(self):  # NOQA D201
        assert OrdinaryLeastSquares is not None
        assert Ridge is not None
        assert Lasso is not None
        assert Bayesian is not None
        assert RANSAC is not None
        assert PassiveAggressiveRegressor is not None
        assert LinearDiscriminantAnalysis is not None
        assert QuadraticDiscriminantAnalysis is not None


if __name__ == "__main__":
    unittest.main()
