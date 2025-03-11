import unittest
import sys
import os
import numpy as np
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.neural_networks import *
from tests.utils import suppress_print


class MockLayer:
    """ Mock layer class for testing optimizers. """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

class TestAdamOptimizer(unittest.TestCase):
    """
    Unit tests for the AdamOptimizer class.
    Methods:
    - setUpClass: Print message before running tests.
    - setUp: Initialize the AdamOptimizer class and layers.
    - test_initialize: Test the initialize method.
    - test_update: Test the update method.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing the AdamOptimizer Class")

    def setUp(self):
        self.optimizer = AdamOptimizer()
        self.layers = [MockLayer(3, 2), MockLayer(2, 1)]
        self.optimizer.initialize(self.layers)

    def test_initialize(self):
        self.assertEqual(len(self.optimizer.m), len(self.layers))
        self.assertEqual(len(self.optimizer.v), len(self.layers))
        for m, v, layer in zip(self.optimizer.m, self.optimizer.v, self.layers):
            np.testing.assert_array_equal(m, np.zeros_like(layer.weights))
            np.testing.assert_array_equal(v, np.zeros_like(layer.weights))

    def test_update(self):
        layer = self.layers[0]
        dW = np.random.randn(*layer.weights.shape)
        db = np.random.randn(*layer.biases.shape)
        initial_weights = layer.weights.copy()
        initial_biases = layer.biases.copy()
        self.optimizer.update(layer, dW, db, 0)
        self.assertFalse(np.array_equal(layer.weights, initial_weights))
        self.assertFalse(np.array_equal(layer.biases, initial_biases))

class TestSGDOptimizer(unittest.TestCase):
    """
    Unit tests for the SGDOptimizer class.
    Methhods:
    - setUpClass: Print message before running tests.
    - setUp: Initialize the SGDOptimizer class and layers.
    - test_initialize: Test the initialize method.
    - test_update: Test the update method.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing the SGDOptimizer Class")

    def setUp(self):
        self.optimizer = SGDOptimizer()
        self.layers = [MockLayer(3, 2), MockLayer(2, 1)]
        self.optimizer.initialize(self.layers)

    def test_initialize(self):
        self.assertEqual(len(self.optimizer.velocity), len(self.layers))
        for v, layer in zip(self.optimizer.velocity, self.layers):
            np.testing.assert_array_equal(v, np.zeros_like(layer.weights))

    def test_update(self):
        layer = self.layers[0]
        dW = np.random.randn(*layer.weights.shape)
        db = np.random.randn(*layer.biases.shape)
        initial_weights = layer.weights.copy()
        initial_biases = layer.biases.copy()

        self.optimizer.update(layer, dW, db, 0)

        self.assertFalse(np.array_equal(layer.weights, initial_weights))
        self.assertFalse(np.array_equal(layer.biases, initial_biases))

class TestAdadeltaOptimizer(unittest.TestCase):
    """
    Unit tests for the AdadeltaOptimizer class.
    Methods:
    - setUpClass: Print message before running tests.
    - setUp: Initialize the AdadeltaOptimizer class and layers.
    - test_initialize: Test the initialize method.
    - test_update: Test the update method.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing the AdadeltaOptimizer Class")

    def setUp(self):
        self.optimizer = AdadeltaOptimizer()
        self.layers = [MockLayer(3, 2), MockLayer(2, 1)]
        self.optimizer.initialize(self.layers)

    def test_initialize(self):
        self.assertEqual(len(self.optimizer.E_g2), len(self.layers))
        self.assertEqual(len(self.optimizer.E_delta_x2), len(self.layers))
        for E_g2, E_delta_x2, layer in zip(self.optimizer.E_g2, self.optimizer.E_delta_x2, self.layers):
            np.testing.assert_array_equal(E_g2, np.zeros_like(layer.weights))
            np.testing.assert_array_equal(E_delta_x2, np.zeros_like(layer.weights))

    def test_update(self):
        layer = self.layers[0]
        dW = np.random.randn(*layer.weights.shape)
        db = np.random.randn(*layer.biases.shape)
        initial_weights = layer.weights.copy()
        initial_biases = layer.biases.copy()

        self.optimizer.update(layer, dW, db, 0)

        self.assertFalse(np.array_equal(layer.weights, initial_weights))
        self.assertFalse(np.array_equal(layer.biases, initial_biases))


if __name__ == '__main__':
    unittest.main()
