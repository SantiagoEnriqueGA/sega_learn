import unittest
import os
import sys

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    for all_test_suite in unittest.defaultTestLoader.discover('tests', pattern='test*.py'):
        for test_suite in all_test_suite:
            if isinstance(test_suite, unittest.TestSuite):
                for test_case in test_suite:
                    # Doesnt contain "TestExamples" or "TestExampleExceptions" 
                    if "TestExamples" not in str(test_case) and "TestExampleExceptions" not in str(test_case): 
                        suite.addTest(test_case)
    return suite

if __name__ == '__main__':
    # Discover and run selected tests in the 'tests' directory
    loader = unittest.TestLoader()
    tests = load_tests(loader, None, 'test*.py')
    testRunner = unittest.TextTestRunner()
    testRunner.run(tests)
