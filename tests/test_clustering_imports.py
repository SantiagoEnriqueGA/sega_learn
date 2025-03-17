import unittest
import sys
import os

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.clustering import DBSCAN as dbs
from sega_learn.clustering import KMeans as kms

from sega_learn.clustering import *

class TestImportsClustering(unittest.TestCase):
    """
    Tests that the clustering subpackage can be imported correctly.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting Imports - Clustering", end="", flush=True)
    
    def test_individual_imports(self):
        assert dbs is not None
        assert kms is not None
       
    def test_wildcard_import(self):
        assert DBSCAN is not None
        assert KMeans is not None

if __name__ == '__main__':
    unittest.main()