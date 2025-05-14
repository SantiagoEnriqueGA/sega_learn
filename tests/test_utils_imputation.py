import os
import sys
import unittest

# import numpy as np
# import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sega_learn.utils.imputation import *
from tests.utils import BaseTest


class TestStatisticalImputer(BaseTest):
    """Unit test for the StatisticalImputer class in the imputation module."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting StatisticalImputer", end="", flush=True)

    def setUp(self):
        """Set up test data and initialize StatisticalImputer instances for testing."""
        pass


class TestDirectionalImputer(BaseTest):
    """Unit test for the DirectionalImputer class in the imputation module."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting DirectionalImputer", end="", flush=True)

    def setUp(self):
        """Set up test data and initialize DirectionalImputer instances for testing."""
        pass


class TestInterpolationImputer(BaseTest):
    """Unit test for the InterpolationImputer class in the imputation module."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting InterpolationImputer", end="", flush=True)

    def setUp(self):
        """Set up test data and initialize InterpolationImputer instances for testing."""
        pass


class TestKNNImputer(BaseTest):
    """Unit test for the KNNImputer class in the imputation module."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting KNNImputer", end="", flush=True)

    def setUp(self):
        """Set up test data and initialize KNNImputer instances for testing."""
        pass


class TestCustomImputer(BaseTest):
    """Unit test for the CustomImputer class in the imputation module."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting CustomImputer", end="", flush=True)

    def setUp(self):
        """Set up test data and initialize CustomImputer instances for testing."""
        pass


if __name__ == "__main__":
    unittest.main()
