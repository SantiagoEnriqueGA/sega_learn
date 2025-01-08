import unittest
import sys
import os
import numpy as np
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.neural_networks import *
from tests.utils import suppress_print

class TestNeuralNetwork(unittest.TestCase):
    """
    Unit tests for the NeuralNetwork class.
    Methods:
    - setUpClass: Print message before running tests.
    - setUp: Initialize the NeuralNetwork class and optimizer.
    - test_apply_dropout: Test the apply_dropout method.
    - test_forward: Test the forward method.
    - test_backward: Test the backward method.
    - test_train: Test the train method.
    - test_evaluate: Test the evaluate method.
    - test_calculate_loss: Test the calculate_loss method.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing the NeuralNetwork class")
    
    def setUp(self):
        self.nn = NeuralNetwork([2, 100, 25, 1], dropout_rate=0.2, reg_lambda=0.01)             # 2 input, 1 output, 2 hidden layers
        self.optimizer = AdamOptimizer(learning_rate=0.01)                                      # Adam optimizer
        np.random.seed(42) 
        self.X = np.random.randn(100, 2)                                                        # Example data
        self.y = np.random.randint(0, 2, (100, 1))                                              # Example labels
        with suppress_print():
            self.nn.train(self.X, self.y, self.X, self.y, self.optimizer, epochs=1, batch_size=32)  # Train the model
     
    def test_apply_dropout(self):
        A = np.ones((10, 10))
        A_dropped = self.nn.apply_dropout(A)
        self.assertEqual(A.shape, A_dropped.shape)                              # Check that the output shape is the same as the input shape
        self.assertTrue(np.any(A_dropped == 0))                                 # Check that the dropout has been applied
        non_zero_elements = A_dropped[A_dropped != 0]                           # Check that the scaling factor has been applied correctly
        self.assertAlmostEqual(np.mean(non_zero_elements), 1/(1-self.nn.dropout_rate), delta=0.1)
        
    def test_forward(self):
        outputs = self.nn.forward(self.X)                               # Forward pass
        self.assertEqual(outputs.shape, (100, 1))                       # Check output, shape
        self.assertTrue(np.all(outputs >= 0) and np.all(outputs <= 1))  # Check output, range

    def test_backward(self):
        self.nn.forward(self.X)                                 # Forward pass
        self.nn.backward(self.y)                                # Backward pass
        for layer in self.nn.layers:                            # Check gradients                
            dW, db = layer.gradients                                # Get gradients
            self.assertEqual(dW.shape, layer.weights.shape)         # Check shape, weights
            self.assertEqual(db.shape, layer.biases.shape)          # Check shape, biases

    def test_train(self):
        with suppress_print():
            self.nn.train(self.X, self.y, self.X, self.y, self.optimizer, epochs=1, batch_size=32)
        accuracy, predicted = self.nn.evaluate(self.X, self.y)      # Evaluate the model
        loss = self.nn.calculate_loss(self.X, self.y)               # Calculate loss
        self.assertTrue(0 <= accuracy <= 1)                         # Check accuracy, range
        self.assertGreater(loss, 0)                                 # Check loss, positive
        
    def test_evaluate(self):
        accuracy, predicted = self.nn.evaluate(self.X, self.y)      # Evaluate the model
        self.assertTrue(0 <= accuracy <= 1)                         # Check accuracy, range
        self.assertEqual(predicted.shape, self.y.shape)             # Check predicted, shape

    def test_calculate_loss(self):
        loss = self.nn.calculate_loss(self.X, self.y)               # Calculate loss
        self.assertGreater(loss, 0)                                 # Check loss, positivedA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15)) 



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
        print("Testing the AdamOptimizer class")

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
        print("Testing the SGDOptimizer class")

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
        print("Testing the AdadeltaOptimizer class")

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




class TestCrossEntropyLoss(unittest.TestCase):
    """
    Unit tests for the CrossEntropyLoss class.
    Methods:
    - setUpClass: Print message before running tests.
    - test_cross_entropy_loss_single_class: Test the cross entropy loss for a single class.
    - test_cross_entropy_loss_multi_class: Test the cross entropy loss for multiple classes.
    """
    @classmethod
    def setUpClass(cls):
        print("Testing the CrossEntropyLoss class")
    
    def test_cross_entropy_loss_single_class(self):
        loss_fn = CrossEntropyLoss()
        logits = np.array([[2.0, 1.0, 0.1]])
        targets = np.array([[1, 0, 0]])
        loss = loss_fn(logits, targets)
        
        # Correct the expected loss calculation
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        expected_loss = -np.sum(targets * np.log(softmax_probs + 1e-15)) / logits.shape[0]
        
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_cross_entropy_loss_multi_class(self):
        loss_fn = CrossEntropyLoss()
        logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        loss = loss_fn(logits, targets)
        
        # Correct the expected loss calculation
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        expected_loss = -np.sum(targets * np.log(softmax_probs + 1e-15)) / logits.shape[0]
        
        self.assertAlmostEqual(loss, expected_loss, places=5)

class TestBCEWithLogitsLoss(unittest.TestCase):
    """
    Unit tests for the BCEWithLogitsLoss class.
    Methods:
    - setUpClass: Print message before running tests.
    - test_bce_with_logits_loss: Test the binary cross entropy loss with logits.
    - test_bce_with_logits_loss_edge_cases: Test the binary cross entropy loss with logits for edge cases (large logits).
    """
    @classmethod
    def setUpClass(cls):
        print("Testing the BCEWithLogitsLoss class")
        
    def test_bce_with_logits_loss(self):
        loss_fn = BCEWithLogitsLoss()
        logits = np.array([0.0, 2.0, -2.0])
        targets = np.array([0, 1, 0])
        loss = loss_fn(logits, targets)
        expected_loss = -np.mean(
            targets * np.log(1 / (1 + np.exp(-logits)) + 1e-15) +
            (1 - targets) * np.log(1 - 1 / (1 + np.exp(-logits)) + 1e-15)
        )
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_bce_with_logits_loss_edge_cases(self):      
        warnings.filterwarnings('ignore')       # Suppress warnings, large logits will trigger overflow warnings
        loss_fn = BCEWithLogitsLoss()
        logits = np.array([1000.0, -1000.0])
        targets = np.array([1, 0])
        loss = loss_fn(logits, targets)
        expected_loss = -np.mean(
            targets * np.log(1 / (1 + np.exp(-logits)) + 1e-15) +
            (1 - targets) * np.log(1 - 1 / (1 + np.exp(-logits)) + 1e-15)
        )
        self.assertAlmostEqual(loss, expected_loss, places=5)

if __name__ == '__main__':
    unittest.main()
