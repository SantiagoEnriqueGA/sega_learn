import os
import sys
import unittest

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sega_learn
from sega_learn import (
    auto,
    clustering,
    linear_models,
    nearest_neighbors,
    neural_networks,
    trees,
    utils,
)


class TestImports(unittest.TestCase):
    """Tests that the main package can be imported correctly."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting Imports - Main Package", end="", flush=True)

    def test_all_imports(self):  # NOQA D201
        assert sega_learn is not None
        assert sega_learn.linear_models is not None
        assert sega_learn.clustering is not None
        assert sega_learn.utils is not None
        assert sega_learn.trees is not None
        assert sega_learn.neural_networks is not None
        assert sega_learn.nearest_neighbors is not None
        assert sega_learn.auto is not None

    def test_module_imports(self):  # NOQA D201
        assert clustering is not None
        assert linear_models is not None
        assert utils is not None
        assert neural_networks is not None
        assert trees is not None
        assert nearest_neighbors is not None
        assert auto is not None


if __name__ == "__main__":
    unittest.main()
