import os
import sys
import unittest

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.utils import *
from sega_learn.utils import PCA as pca
from sega_learn.utils import SMOTE as smote
from sega_learn.utils import SVD as svd
from sega_learn.utils import Augmenter as augmenter
from sega_learn.utils import DataPrep as dp
from sega_learn.utils import GridSearchCV as gscv
from sega_learn.utils import Metrics as metrics
from sega_learn.utils import ModelSelectionUtility as msu
from sega_learn.utils import PolynomialTransform as plt
from sega_learn.utils import RandomOverSampler as ros
from sega_learn.utils import RandomSearchCV as rscv
from sega_learn.utils import RandomUnderSampler as rus
from sega_learn.utils import Scaler as scaler
from sega_learn.utils import VotingRegressor as vr
from sega_learn.utils import make_blobs as mkblobs
from sega_learn.utils import make_classification as mkcls
from sega_learn.utils import make_regression as mkreg
from sega_learn.utils import normalize as norm
from sega_learn.utils import one_hot_encode as ohe
from sega_learn.utils import train_test_split as tts


class TestImportsSVM(unittest.TestCase):
    """Tests that the SVM subpackage can be imported correctly.

    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting Imports - SVM", end="", flush=True)

    def test_individual_imports(self):
        """Tests that each module in the segadb package can be imported individually."""
        assert plt is not None
        assert dp is not None
        assert vr is not None
        assert msu is not None
        assert gscv is not None
        assert rscv is not None
        assert metrics is not None
        assert ros is not None
        assert rus is not None
        assert smote is not None
        assert augmenter is not None
        assert pca is not None
        assert svd is not None
        assert mkreg is not None
        assert mkcls is not None
        assert mkblobs is not None
        assert tts is not None
        assert ohe is not None
        assert norm is not None
        assert scaler is not None

    def test_wildcard_import(self):
        """Tests that the segadb package can be imported using a wildcard import."""
        assert PolynomialTransform is not None
        assert DataPrep is not None
        assert VotingRegressor is not None
        assert ModelSelectionUtility is not None
        assert GridSearchCV is not None
        assert RandomSearchCV is not None
        assert Metrics is not None
        assert RandomOverSampler is not None
        assert RandomUnderSampler is not None
        assert SMOTE is not None
        assert Augmenter is not None
        assert PCA is not None
        assert SVD is not None
        assert make_regression is not None
        assert make_classification is not None
        assert make_blobs is not None
        assert train_test_split is not None
        assert one_hot_encode is not None
        assert normalize is not None
        assert Scaler is not None


if __name__ == "__main__":
    unittest.main()
