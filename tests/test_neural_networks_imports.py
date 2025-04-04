import os
import sys
import unittest

# Change the working directory to the parent directory to allow importing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.neural_networks import *
from sega_learn.neural_networks import Activation as act
from sega_learn.neural_networks import AdadeltaOptimizer as adadelta
from sega_learn.neural_networks import AdamOptimizer as adam
from sega_learn.neural_networks import BCEWithLogitsLoss as bcel
from sega_learn.neural_networks import CrossEntropyLoss as cel
from sega_learn.neural_networks import DenseLayer as lay
from sega_learn.neural_networks import JITAdadeltaOptimizer as jitadadelta
from sega_learn.neural_networks import JITAdamOptimizer as jitadam
from sega_learn.neural_networks import JITBCEWithLogitsLoss as jitbcel
from sega_learn.neural_networks import JITCrossEntropyLoss as jitcel
from sega_learn.neural_networks import JITDenseLayer as jitlay
from sega_learn.neural_networks import JITSGDOptimizer as jitsgd
from sega_learn.neural_networks import NeuralNetworkBase as nn
from sega_learn.neural_networks import SGDOptimizer as sgd
from sega_learn.neural_networks import lr_scheduler_exp as lrExp
from sega_learn.neural_networks import lr_scheduler_plateau as lrPlateau
from sega_learn.neural_networks import lr_scheduler_step as lrStep
from sega_learn.neural_networks import neuralNetworkBaseBackend as nnBackend
from sega_learn.neural_networks import neuralNetworkNumbaBackend as nnNumba


class TestImportsNeuralNetworks(unittest.TestCase):
    """Tests that the neural_networks package can be imported correctly.

    Methods:
    - setUpClass: Initializes a new instance of the Index class before each test method is run.
    - test_individual_imports: Tests that each module in the segadb package can be imported individually.
    - test_wildcard_import: Tests that the segadb package can be imported using a wildcard import.
    """

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting Imports - Neural Networks", end="", flush=True)

    def test_base_method_imports(self):  # NOQA D201
        assert lrExp is not None
        assert lrPlateau is not None
        assert lrStep is not None
        assert nn is not None

    def test_base_method_wildcard_import(self):  # NOQA D201
        assert lr_scheduler_exp is not None
        assert lr_scheduler_plateau is not None
        assert lr_scheduler_step is not None
        assert NeuralNetworkBase is not None

    def test_base_backend_imports(self):  # NOQA D201
        assert adam is not None
        assert sgd is not None
        assert adadelta is not None
        assert cel is not None
        assert bcel is not None
        assert nn is not None
        assert lay is not None
        assert act is not None
        assert nnBackend is not None

    def test_base_backend_wildcard_import(self):  # NOQA D201
        assert AdamOptimizer is not None
        assert SGDOptimizer is not None
        assert AdadeltaOptimizer is not None
        assert lr_scheduler_exp is not None
        assert lr_scheduler_plateau is not None
        assert lr_scheduler_step is not None
        assert CrossEntropyLoss is not None
        assert BCEWithLogitsLoss is not None
        assert DenseLayer is not None
        assert Activation is not None
        assert BaseBackendNeuralNetwork is not None

    def test_numba_method_imports(self):  # NOQA D201
        assert jitadam is not None
        assert jitsgd is not None
        assert jitadadelta is not None
        assert jitcel is not None
        assert jitbcel is not None
        assert jitlay is not None
        assert nnNumba is not None

    def test_numba_method_wildcard_import(self):  # NOQA D201
        assert JITAdamOptimizer is not None
        assert JITSGDOptimizer is not None
        assert JITAdadeltaOptimizer is not None
        assert JITCrossEntropyLoss is not None
        assert JITBCEWithLogitsLoss is not None
        assert JITDenseLayer is not None
        assert NumbaBackendNeuralNetwork is not None


if __name__ == "__main__":
    unittest.main()
