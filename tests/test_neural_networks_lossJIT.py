import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sega_learn.neural_networks import *


class TestJITCrossEntropyLoss(unittest.TestCase):
    """Unit tests for the JITCrossEntropyLoss class."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting the JITCrossEntropyLoss class", end="", flush=True)

    def test_cross_entropy_loss_single_class(self):
        """Test the cross entropy loss for a single class."""
        loss_fn = JITCrossEntropyLoss()
        logits = np.array([[2.0, 1.0, 0.1]])
        targets = np.array([[1, 0, 0]])
        loss = loss_fn.calculate_loss(logits, targets)

        # Correct the expected loss calculation
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        expected_loss = (
            -np.sum(targets * np.log(softmax_probs + 1e-15)) / logits.shape[0]
        )

        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_cross_entropy_loss_multi_class(self):
        """Test the cross entropy loss for multiple classes."""
        loss_fn = JITCrossEntropyLoss()
        logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        loss = loss_fn.calculate_loss(logits, targets)

        # Correct the expected loss calculation
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        expected_loss = (
            -np.sum(targets * np.log(softmax_probs + 1e-15)) / logits.shape[0]
        )

        self.assertAlmostEqual(loss, expected_loss, places=5)


class TestJITBCEWithLogitsLoss(unittest.TestCase):
    """Unit tests for the JITBCEWithLogitsLoss class."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting the JITBCEWithLogitsLoss class", end="", flush=True)

    def test_bce_with_logits_loss(self):
        """Test the binary cross entropy loss with logits."""
        loss_fn = JITBCEWithLogitsLoss()
        logits = np.array([0.0, 2.0, -2.0])
        targets = np.array([0, 1, 0])
        loss = loss_fn.calculate_loss(logits, targets)
        expected_loss = -np.mean(
            targets * np.log(1 / (1 + np.exp(-logits)) + 1e-15)
            + (1 - targets) * np.log(1 - 1 / (1 + np.exp(-logits)) + 1e-15)
        )
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_bce_with_logits_loss_edge_cases(self):
        """Test the binary cross entropy loss with logits for edge cases (large logits)."""
        warnings.filterwarnings(
            "ignore"
        )  # Suppress warnings, large logits will trigger overflow warnings
        loss_fn = JITBCEWithLogitsLoss()
        logits = np.array([1000.0, -1000.0])
        targets = np.array([1, 0])
        loss = loss_fn.calculate_loss(logits, targets)
        expected_loss = -np.mean(
            targets * np.log(1 / (1 + np.exp(-logits)) + 1e-15)
            + (1 - targets) * np.log(1 - 1 / (1 + np.exp(-logits)) + 1e-15)
        )
        self.assertAlmostEqual(loss, expected_loss, places=5)


class TestMeanSquaredErrorLoss(unittest.TestCase):
    """Unit tests for the JITMeanSquaredErrorLoss class."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting the JITMeanSquaredErrorLoss class", end="", flush=True)

    def test_mean_squared_error_loss(self):
        """Test the mean squared error loss."""
        loss_fn = JITMeanSquaredErrorLoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        loss = loss_fn.calculate_loss(y_true, y_pred)
        expected_loss = np.mean((y_true - y_pred) ** 2)
        self.assertAlmostEqual(loss, expected_loss, places=5)


class TestMeanAbsoluteErrorLoss(unittest.TestCase):
    """Unit tests for the JITMeanAbsoluteErrorLoss class."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting the JITMeanAbsoluteErrorLoss class", end="", flush=True)

    def test_mean_absolute_error_loss(self):
        """Test the mean absolute error loss."""
        loss_fn = JITMeanAbsoluteErrorLoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        loss = loss_fn.calculate_loss(y_true, y_pred)
        expected_loss = np.mean(np.abs(y_true - y_pred))
        self.assertAlmostEqual(loss, expected_loss, places=5)


class TestHuberLoss(unittest.TestCase):
    """Unit tests for the HuberLoss class."""

    @classmethod
    def setUpClass(cls):  # NOQA D201
        print("\nTesting the JITHuberLoss class", end="", flush=True)

    def test_huber_loss_small_error(self):
        """Test the Huber loss for small errors."""
        delta = 1.0
        loss_fn = JITHuberLoss(delta=delta)
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        loss = loss_fn.calculate_loss(y_true, y_pred)
        error = y_true - y_pred
        expected_loss = np.mean(0.5 * error**2)
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_huber_loss_large_error(self):
        """Test the Huber loss for large errors."""
        delta = 1.0
        loss_fn = JITHuberLoss(delta=delta)
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        loss = loss_fn.calculate_loss(y_true, y_pred)
        error = y_true - y_pred
        expected_loss = np.mean(delta * (np.abs(error) - 0.5 * delta))
        self.assertAlmostEqual(loss, expected_loss, places=5)


if __name__ == "__main__":
    unittest.main()
