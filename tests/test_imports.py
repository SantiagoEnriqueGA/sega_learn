import unittest
import sys
import os

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sega_learn
from sega_learn import clustering
from sega_learn import linear_models
from sega_learn import utils
from sega_learn import trees
from sega_learn import neural_networks
from sega_learn import nearest_neighbors

class TestImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nTesting Imports - Main Package", end="", flush=True)
        
    def test_all_imports(self):
        assert sega_learn is not None
        assert sega_learn.linear_models is not None
        assert sega_learn.clustering is not None
        assert sega_learn.utils is not None
        assert sega_learn.trees is not None
        assert sega_learn.neural_networks is not None
        assert sega_learn.nearest_neighbors is not None
    
    def test_module_imports(self):
        assert clustering is not None
        assert linear_models is not None
        assert utils is not None
        assert neural_networks is not None
        assert trees is not None
        assert nearest_neighbors is not None
    
if __name__ == "__main__":
    unittest.main()