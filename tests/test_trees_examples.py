
import unittest
import glob
import os
import sys
import importlib.util
import contextlib
import io

from matplotlib.pylab import f

from utils import suppress_print, strip_file_path

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestExampleExceptions(unittest.TestCase):
    """
    Test cases to check for exceptions in example files.
    """
    def test_main(self, example_file):
        if 'gradientBoostedRegressor.py' in example_file:
            from examples.trees.gradientBoostedRegressor import basic_example
            print(f"Testing file: {strip_file_path(example_file)}")
            with suppress_print():
                basic_example(num_trees=5, max_depth=3)
        
        elif 'randomForestClassifier.py' in example_file:
            from examples.trees.randomForestClassifier import basic_example
            print(f"Testing file: {strip_file_path(example_file)}")
            with suppress_print():
                basic_example(num_trees=5, max_depth=3)
                
        elif 'randomForestRegressor.py' in example_file:
            from examples.trees.randomForestRegressor import basic_example
            print(f"Testing file: {strip_file_path(example_file)}")
            with suppress_print():
                basic_example(num_trees=5, max_depth=3)

class TestExamplesTrees(unittest.TestCase):
    """
    Test cases for the example files.  
    Holds dynamically generated test cases for each example file.
    """
    pass

def load_tests(loader, tests, pattern):
    """
    Dynamically load test cases for each example file.
    args:
        loader: The test loader instance.
        tests: The test cases to load.
        pattern: The pattern to match test files.
    """
    # Find all example files in the examples directory. (Files starting with 'example_')
    example_files = glob.glob(os.path.join(os.path.dirname(__file__), '..\\examples\\trees\\*.py'))
    
    # Raise an error if no example files are found.
    if not example_files:
        raise FileNotFoundError("No example files found.")
    
    # Dynamically generate test cases for each example file.
    for example_file in example_files:
        test_name = f'test_{os.path.basename(example_file)}'
        
        def test_func(self, example_file=example_file):
            """Tests the functionality of a given example file by importing it as a module and executing it."""
            print(f"Testing file: {strip_file_path(example_file)}")
            
            # Import the example file as a module and execute it.
            spec = importlib.util.spec_from_file_location("module.name", example_file)
            example_module = importlib.util.module_from_spec(spec)
            
            # Redirect stdout to suppress output from the example file.
            with contextlib.redirect_stdout(io.StringIO()):
                spec.loader.exec_module(example_module)
        
        # Dynamically add the test function to the TestExamples class.
        # If example file runs in __main__, use a lambda function to call the test function.
        test_exeptions = ['gradientBoostedRegressor.py', 'randomForestClassifier.py', 'randomForestRegressor.py']
        test_exeptions = [f'test_{name}' for name in test_exeptions]
        if test_name in test_exeptions:
            setattr(TestExamplesTrees, test_name, lambda self, example_file=example_file: TestExampleExceptions().test_main(example_file))
        else:
            setattr(TestExamplesTrees, test_name, test_func)
    
    # Load the dynamically generated test cases.
    return loader.loadTestsFromTestCase(TestExamplesTrees)

if __name__ == '__main__':
    unittest.main()