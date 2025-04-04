import os
import sys
import unittest
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_tests(loader, standard_tests, pattern):
    """Dynamically load test cases for each example file.

    Args:
        loader: The test loader instance.
        standard_tests: The standard test cases to load.
        pattern: The pattern to match test files.
    """
    suite = unittest.TestSuite()
    for all_test_suite in unittest.defaultTestLoader.discover(
        "tests", pattern="test*.py"
    ):
        for test_suite in all_test_suite:
            if isinstance(test_suite, unittest.TestSuite):
                for test_case in test_suite:
                    # Doesnt contain "TestExamples" or "TestExampleExceptions"
                    if "TestExamples" not in str(
                        test_case
                    ) and "TestExampleExceptions" not in str(test_case):
                        suite.addTest(test_case)
    return suite


if __name__ == "__main__":
    # Discover and run selected tests in the 'tests' directory
    loader = unittest.TestLoader()
    tests = load_tests(loader, None, "test*.py")
    testRunner = unittest.TextTestRunner()
    testRunner.run(tests)
