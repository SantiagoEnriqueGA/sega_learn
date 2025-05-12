import csv
import os
import sys
import unittest
from functools import partialmethod

from tqdm import tqdm

from _default_tests import DEFAULT_TESTS

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


CUTOFF = 1.0  # Elapsed time cutoff in seconds
TEST_LOG_FILE = "tests/_test_log.csv"
FILTER_NUMBA = True


def get_quick_tests():
    """Retrieve test names with elapsed time less than the cutoff or use DEFAULT_TESTS if no log file exists."""
    quick_tests = set()
    if os.path.exists(TEST_LOG_FILE):
        with open(TEST_LOG_FILE) as file:
            reader = csv.DictReader(file)
            for row in reader:
                if float(row["Elapsed Time (s)"]) < CUTOFF:
                    quick_tests.add(row["Test Name"])
    else:
        quick_tests.update(DEFAULT_TESTS)
    return quick_tests


def load_tests(loader, standard_tests, pattern):
    """Dynamically load test cases for each example file.

    Args:
        loader: The test loader instance.
        standard_tests: The standard test cases to load.
        pattern: The pattern to match test files.
    """
    quick_tests = get_quick_tests()
    suite = unittest.TestSuite()
    for all_test_suite in unittest.defaultTestLoader.discover(
        "tests", pattern="test*.py"
    ):
        for test_suite in all_test_suite:
            if isinstance(test_suite, unittest.TestSuite):
                for test_case in test_suite:
                    test_name = str(test_case)
                    # Exclude tests not in quick_tests or containing "numba" or "JIT" if FILTER_NUMBA is True
                    if (
                        test_name in quick_tests
                        and not (
                            FILTER_NUMBA
                            and (
                                "numba" in test_name.lower()
                                or "jit" in test_name.lower()
                            )
                        )
                        and "TestExamples" not in str(test_case)
                        and "TestExampleExceptions" not in str(test_case)
                        and "_imports" not in str(test_case)
                    ):
                        suite.addTest(test_case)
    return suite


if __name__ == "__main__":
    # Discover and run selected tests in the 'tests' directory
    loader = unittest.TestLoader()
    tests = load_tests(loader, None, "test*.py")
    testRunner = unittest.TextTestRunner()
    testRunner.run(tests)
