import csv
import os
import sys
import time
import unittest
import warnings
from functools import partialmethod

from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
warnings.filterwarnings("ignore")

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def reorder_tests(tests):
    """Reorder tests to run import tests first, then normal tests, and finally example file tests."""
    import_tests = []
    example_tests = []
    normal_tests = []
    for test in tests:
        if "import" in str(test).lower():
            import_tests.append(test)
        elif "example" in str(test).lower():
            example_tests.append(test)
        else:
            normal_tests.append(test)
    return import_tests + normal_tests + example_tests


class LoggingTestResult(unittest.TextTestResult):
    """Custom test result class to log test results with elapsed time."""

    def __init__(self, *args, **kwargs):
        """Initialize the LoggingTestResult."""
        super().__init__(*args, **kwargs)
        self.test_log = []

    def startTest(self, test):
        """Start logging the test with the current time."""
        self._start_time = time.time()
        super().startTest(test)

    def stopTest(self, test):
        """Log the test result with elapsed time."""
        elapsed_time = time.time() - self._start_time
        status = (
            "PASSED"
            if test not in self.failures and test not in self.errors
            else "FAILED"
        )
        self.test_log.append((str(test), status, elapsed_time))
        super().stopTest(test)


if __name__ == "__main__":
    # Discover and run all tests in the 'tests' directory
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    # Reorder tests
    ordered_tests = reorder_tests(list(suite))
    ordered_suite = unittest.TestSuite(ordered_tests)

    # Run tests with logging
    with open("tests/_test_log.csv", "w", newline="") as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["Test Name", "Status", "Elapsed Time (s)"])

        testRunner = unittest.TextTestRunner(resultclass=LoggingTestResult)
        result = testRunner.run(ordered_suite)

        # Write test results to the CSV file
        for test_name, status, elapsed_time in result.test_log:
            csv_writer.writerow([test_name, status, f"{elapsed_time:.2f}"])
