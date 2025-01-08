import unittest
import sys
import os

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.neural_networks import AdamOptimizer as adam
from sega_learn.neural_networks import SGDOptimizer as sgd
from sega_learn.neural_networks import AdadeltaOptimizer as adadelta
from sega_learn.neural_networks import lr_scheduler_exp as lrExp
from sega_learn.neural_networks import lr_scheduler_plateau as lrPlateau
from sega_learn.neural_networks import lr_scheduler_step as lrStep
from sega_learn.neural_networks import CrossEntropyLoss as cel
from sega_learn.neural_networks import BCEWithLogitsLoss as bcel
from sega_learn.neural_networks import NeuralNetwork as nn
from sega_learn.neural_networks import Layer as lay
from sega_learn.neural_networks import Activation as act

from sega_learn.neural_networks import *

class TestImportsNeuralNetworks(unittest.TestCase):
    """
    Tests that the neural_networks package can be imported correctly.
    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing Imports - Neural Networks")
    
    def test_individual_imports(self):
        assert adam is not None
        assert sgd is not None
        assert adadelta is not None
        assert lrExp is not None
        assert lrPlateau is not None
        assert lrStep is not None
        assert cel is not None
        assert bcel is not None
        assert nn is not None
        assert lay is not None
        assert act is not None
   
    def test_wildcard_import(self):
        assert AdamOptimizer is not None
        assert SGDOptimizer is not None
        assert AdadeltaOptimizer is not None
        assert lr_scheduler_exp is not None
        assert lr_scheduler_plateau is not None
        assert lr_scheduler_step is not None
        assert CrossEntropyLoss is not None
        assert BCEWithLogitsLoss is not None
        assert NeuralNetwork is not None
        assert Layer is not None
        assert Activation is not None

if __name__ == '__main__':
    unittest.main()