import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.neural_networks import *


class TestDenseLayer(unittest.TestCase):
    """
    Comprehensive test suite for DenseLayer class.
    Tests initialization, forward pass, and backward pass functionalities.
    """

    @classmethod
    def setUpClass(cls):
        print("\nTesting DenseLayer class", end="", flush=True)

    def setUp(self):
        """Initialize test fixtures."""
        self.input_size = 10
        self.output_size = 5
        self.layer = DenseLayer(self.input_size, self.output_size, activation="relu")
        np.random.seed(42)

    ### Initialization Tests ###
    def test_initialization(self):
        """Test layer initialization with correct shapes and He initialization."""
        self.assertEqual(self.layer.weights.shape, (self.input_size, self.output_size))
        self.assertEqual(self.layer.biases.shape, (1, self.output_size))
        # Check He initialization for ReLU
        expected_scale = np.sqrt(2.0 / self.input_size)
        weights = self.layer.weights
        self.assertTrue(np.all(weights < 3 * expected_scale))
        self.assertTrue(np.all(weights > -3 * expected_scale))
        # Verify biases are initialized to zeros
        np.testing.assert_array_equal(
            self.layer.biases, np.zeros((1, self.output_size))
        )

    ### Forward Pass Tests ###
    def test_forward(self):
        """Test the forward pass with ReLU activation."""
        X = np.random.randn(3, self.input_size)
        output = self.layer.forward(X)
        self.assertEqual(output.shape, (3, self.output_size))
        # Manually compute expected output
        Z = np.dot(X, self.layer.weights) + self.layer.biases
        expected_output = np.maximum(0, Z)  # ReLU activation
        np.testing.assert_array_almost_equal(output, expected_output)

    ### Backward Pass Tests ###
    def test_backward(self):
        """Test the backward pass with gradient computations."""
        X = np.random.randn(3, self.input_size)
        self.layer.forward(X)  # Cache input and output
        dA = np.random.randn(3, self.output_size)
        dA_prev = self.layer.backward(dA, reg_lambda=0)
        self.assertEqual(dA_prev.shape, (3, self.input_size))
        # Manually compute expected gradients
        Z = np.dot(X, self.layer.weights) + self.layer.biases
        _output = np.maximum(0, Z)
        dZ = dA * (Z > 0)  # ReLU derivative
        dW = np.dot(X.T, dZ) / 3  # m=3 (batch size)
        db = np.sum(dZ, axis=0, keepdims=True) / 3
        dA_prev_expected = np.dot(dZ, self.layer.weights.T)
        np.testing.assert_array_almost_equal(self.layer.weight_gradients, dW)
        np.testing.assert_array_almost_equal(self.layer.bias_gradients, db)
        np.testing.assert_array_almost_equal(dA_prev, dA_prev_expected)


class TestFlattenLayer(unittest.TestCase):
    """
    Comprehensive test suite for FlattenLayer class.
    Tests forward and backward pass functionalities for flattening multi-dimensional inputs.
    """

    @classmethod
    def setUpClass(cls):
        print("\nTesting FlattenLayer class", end="", flush=True)

    def setUp(self):
        """Initialize test fixtures."""
        self.layer = FlattenLayer()
        np.random.seed(42)

    ### Forward Pass Tests ###
    def test_forward(self):
        """Test the forward pass flattens input correctly."""
        X = np.random.randn(2, 3, 4, 4)  # batch_size=2, channels=3, height=4, width=4
        output = self.layer.forward(X)
        self.assertEqual(output.shape, (2, 3 * 4 * 4))  # (2, 48)
        expected_output = X.reshape(2, -1)
        np.testing.assert_array_equal(output, expected_output)
        # Verify input shape is cached
        self.assertEqual(self.layer.input_shape, (3, 4, 4))

    ### Backward Pass Tests ###
    def test_backward(self):
        """Test the backward pass reshapes gradients correctly."""
        X = np.random.randn(2, 3, 4, 4)
        self.layer.forward(X)  # Cache input shape
        dA = np.random.randn(2, 3 * 4 * 4)
        dA_prev = self.layer.backward(dA)
        self.assertEqual(dA_prev.shape, (2, 3, 4, 4))
        expected_dA_prev = dA.reshape(2, 3, 4, 4)
        np.testing.assert_array_equal(dA_prev, expected_dA_prev)


class TestConvLayer(unittest.TestCase):
    """
    Comprehensive test suite for ConvLayer class.
    Tests initialization, forward pass, and backward pass with convolution operations.
    """

    @classmethod
    def setUpClass(cls):
        print("\nTesting ConvLayer class", end="", flush=True)

    def setUp(self):
        """Initialize test fixtures."""
        self.in_channels = 1
        self.out_channels = 1
        self.kernel_size = 3
        self.stride = 1
        self.padding = 0
        self.layer = ConvLayer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            activation="none",
        )
        # Set predictable weights and biases
        self.layer.weights = np.ones((1, 1, 3, 3))  # All weights are 1
        self.layer.biases = np.zeros((1, 1))
        np.random.seed(42)

    ### Initialization Tests ###
    def test_initialization(self):
        """Test layer initialization with correct shapes."""
        self.assertEqual(
            self.layer.weights.shape,
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
        )
        self.assertEqual(self.layer.biases.shape, (self.out_channels, 1))

    ### Forward Pass Tests ###
    def test_forward(self):
        """Test the forward pass with a simple convolution."""
        X = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])  # 1 batch, 1 channel, 3x3
        output = self.layer.forward(X)
        self.assertEqual(output.shape, (1, 1, 1, 1))  # Output size: (1, 1, 1, 1)
        expected_output = np.array([[[[45]]]])  # Sum of all elements with 3x3 kernel
        np.testing.assert_array_equal(output, expected_output)

    ### Backward Pass Tests ###
    def test_backward(self):
        """Test the backward pass with gradient computations."""
        X = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
        self.layer.forward(X)  # Cache input and intermediates
        d_out = np.array([[[[1]]]])  # Gradient from next layer
        dX = self.layer.backward(d_out)
        self.assertEqual(dX.shape, (1, 1, 3, 3))
        # With all-one weights and d_out=1, dX should be all ones
        expected_dX = np.ones((1, 1, 3, 3))
        np.testing.assert_array_equal(dX, expected_dX)
        # Weight gradients should equal the input X
        expected_dW = X
        np.testing.assert_array_equal(self.layer.weight_gradients, expected_dW)
        # Bias gradient should be the sum of d_out
        expected_db = np.array([[1]])
        np.testing.assert_array_equal(self.layer.bias_gradients, expected_db)


if __name__ == "__main__":
    unittest.main()
