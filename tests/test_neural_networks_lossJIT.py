import unittest
import sys
import os
import numpy as np
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.neural_networks import *
from tests.utils import suppress_print

class TestJITCrossEntropyLoss(unittest.TestCase):
    """
    Unit tests for the JITCrossEntropyLoss class.
    """
    @classmethod
    def setUpClass(cls):
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
        expected_loss = -np.sum(targets * np.log(softmax_probs + 1e-15)) / logits.shape[0]
        
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
        expected_loss = -np.sum(targets * np.log(softmax_probs + 1e-15)) / logits.shape[0]
        
        self.assertAlmostEqual(loss, expected_loss, places=5)

class TestJITBCEWithLogitsLoss(unittest.TestCase): 
    """
    Unit tests for the JITBCEWithLogitsLoss class.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting the BCEWithLogitsLoss class", end="", flush=True)
        
    def test_bce_with_logits_loss(self):
        """Test the binary cross entropy loss with logits."""
        loss_fn = JITBCEWithLogitsLoss()
        logits = np.array([0.0, 2.0, -2.0])
        targets = np.array([0, 1, 0])
        loss = loss_fn.calculate_loss(logits, targets)
        expected_loss = -np.mean(
            targets * np.log(1 / (1 + np.exp(-logits)) + 1e-15) +
            (1 - targets) * np.log(1 - 1 / (1 + np.exp(-logits)) + 1e-15)
        )
        self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_bce_with_logits_loss_edge_cases(self):      
        """Test the binary cross entropy loss with logits for edge cases (large logits)."""
        warnings.filterwarnings('ignore')       # Suppress warnings, large logits will trigger overflow warnings
        loss_fn = JITBCEWithLogitsLoss()
        logits = np.array([1000.0, -1000.0])
        targets = np.array([1, 0])
        loss = loss_fn.calculate_loss(logits, targets)
        expected_loss = -np.mean(
            targets * np.log(1 / (1 + np.exp(-logits)) + 1e-15) +
            (1 - targets) * np.log(1 - 1 / (1 + np.exp(-logits)) + 1e-15)
        )
        self.assertAlmostEqual(loss, expected_loss, places=5)

if __name__ == '__main__':
    unittest.main()
