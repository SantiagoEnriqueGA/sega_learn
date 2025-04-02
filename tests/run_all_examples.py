import os
import sys
import unittest
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for all_test_suite in unittest.defaultTestLoader.discover(
        "tests", pattern="test*_examples.py"
    ):
        for test_suite in all_test_suite:
            suite.addTest(test_suite)
    return suite


if __name__ == "__main__":
    # Discover and run selected tests in the 'tests' directory
    loader = unittest.TestLoader()
    tests = load_tests(loader, None, "test*.py")
    testRunner = unittest.TextTestRunner()
    testRunner.run(tests)
