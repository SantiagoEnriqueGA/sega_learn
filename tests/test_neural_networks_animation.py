import unittest
import sys
import os
import io
import tempfile
import warnings
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.neural_networks.animation import TrainingAnimator
from tests.utils import suppress_print

# Dummy writer to simulate FFMpegWriter / PillowWriter behavior.
class DummyWriter:
    def __init__(self, *args, **kwargs):
        self.setup_called = False
        self.grab_called = False
        self.finish_called = False
    def setup(self, fig, filepath, dpi):
        self.setup_called = True
        self.fig = fig
        self.filepath = filepath
        self.dpi = dpi
    def grab_frame(self):
        self.grab_called = True
    def finish(self):
        self.finish_called = True

class TestTrainingAnimator(unittest.TestCase):
    """
     Unit tests for the TrainingAnimator class.
     Methods:
     - setUpClass: Print message before running tests.
     - setUp: Initialize the TrainingAnimator class and metrics.
     - test_initialization: Test the initialization method.
     - test_update_metrics: Test the update_metrics method.
     - test_animate_training_metrics: Test the animate_training_metrics method.
     - test_animate_training_metrics_no_initialize: Test the animate_training_metrics method with no initialization.
     - test_setup_training_video_success: Test the setup_training_video method with a successful setup.
     - test_setup_training_video_no_initialize: Test the setup_training_video method with no initialization.
     - test_add_training_frame: Test the add_training_frame method.
     - test_finish_training_video: Test the finish_training_video method.
     - test_setup_training_video_fallback: Test the fallback mechanism in setup_training_video.
    """
    @classmethod
    def setUpClass(cls):
        print("\nTesting the TrainingAnimator Class", end="", flush=True)

    def setUp(self):
        # Create an instance with specific figure size and dpi for testing.
        self.animator = TrainingAnimator(figure_size=(8, 6), dpi=80)

    def test_initialization(self):
        """Test that the __init__ sets the default attributes correctly."""
        self.assertEqual(self.animator.figure_size, (8, 6))
        self.assertEqual(self.animator.dpi, 80)
        self.assertIsNone(self.animator.fig)
        self.assertIsNone(self.animator.axes)
        self.assertEqual(self.animator.lines, {})
        self.assertEqual(self.animator.metrics, {})
        self.assertIsNone(self.animator.writer)
        self.assertEqual(self.animator.frame_count, 0)
        self.assertEqual(self.animator.metric_to_ax, {})

    def test_initialize_no_validation(self):
        """Test initialize without validation metrics."""
        metrics = ["loss", "accuracy"]
        self.animator.initialize(metrics_to_track=metrics, has_validation=False)
        # Check that the metrics dictionary contains only the training keys.
        self.assertEqual(set(self.animator.metrics.keys()), set(metrics))
        for metric in metrics:
            self.assertEqual(self.animator.metrics[metric], [])
            self.assertIn(metric, self.animator.lines)
            self.assertIn(metric, self.animator.metric_to_ax)
        # With 2 metrics, axes should be reshaped to (1,2)
        self.assertIsNotNone(self.animator.fig)
        self.assertIsNotNone(self.animator.axes)
        self.assertEqual(self.animator.axes.shape, (1, 2))

    def test_initialize_with_validation(self):
        """Test initialize when validation metrics are requested."""
        metrics = ["loss", "accuracy", "learning_rate"]
        self.animator.initialize(metrics_to_track=metrics, has_validation=True)
        # Expected keys: all training keys plus validation keys (except for learning_rate)
        expected_keys = set(metrics)
        expected_keys.update(["val_loss", "val_accuracy"])
        self.assertEqual(set(self.animator.metrics.keys()), expected_keys)
        # Check that training lines exist for all metrics and validation lines exist where applicable.
        for metric in metrics:
            self.assertIn(metric, self.animator.lines)
            if metric != "learning_rate":
                self.assertIn(f"val_{metric}", self.animator.lines)
            else:
                self.assertNotIn(f"val_{metric}", self.animator.lines)

    def test_update_metrics(self):
        """Test updating metrics for both training and validation."""
        metrics = ["loss", "accuracy"]
        self.animator.initialize(metrics_to_track=metrics, has_validation=True)
        # Update training metrics.
        self.animator.update_metrics({"loss": 0.5, "accuracy": 0.8}, validation=False)
        self.assertEqual(self.animator.metrics["loss"], [0.5])
        self.assertEqual(self.animator.metrics["accuracy"], [0.8])
        # Update validation metrics.
        self.animator.update_metrics({"loss": 0.4, "accuracy": 0.85}, validation=True)
        self.assertEqual(self.animator.metrics["val_loss"], [0.4])
        self.assertEqual(self.animator.metrics["val_accuracy"], [0.85])

    def test_animate_training_metrics(self):
        """
        Test that animate_training_metrics creates a FuncAnimation correctly and saves it to a temp file.
        If ffmpeg is not available, it should fall back to PillowWriter and save the animation as a .gif file.
        """                
        metrics = ["loss", "accuracy"]
        self.animator.initialize(metrics_to_track=metrics, has_validation=False)
        # Simulate training data for 3 epochs.
        self.animator.metrics["loss"] = [1.0, 0.8, 0.6]
        self.animator.metrics["accuracy"] = [0.5, 0.6, 0.7]
        # Create animation
        anim = self.animator.animate_training_metrics(interval=100, blit=False)
        self.assertIsInstance(anim, FuncAnimation)
        # Check that the x-axis limits for one of the axes are set correctly.
        ax = self.animator.metric_to_ax["loss"]
        xlim = ax.get_xlim()
        self.assertEqual(xlim, (0, 4))
        # Save animation to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            try:
                anim.save(tmp_path, fps=10, dpi=100, writer="ffmpeg")
            except ValueError:
                anim.save(tmp_path, fps=10, dpi=100, writer="pillow")
            self.assertTrue(os.path.exists(tmp_path))  # Ensure the file is created
        finally:
            os.remove(tmp_path)  # Clean up after test
            

    def test_animate_training_metrics_no_initialize(self):
        """Test that animate_training_metrics raises an error if initialize() wasn't called."""
        with self.assertRaises(ValueError):
            self.animator.animate_training_metrics()

    def test_setup_training_video_success(self):
        """Test successful setup of the training video writer using a dummy FFMpegWriter."""
        self.animator.initialize(metrics_to_track=["loss"], has_validation=False)
        # Use a temporary file for the output.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            # Patch FFMpegWriter from matplotlib.animation to return our DummyWriter.
            with patch("matplotlib.animation.FFMpegWriter", return_value=DummyWriter()):
                # Suppress any print output during setup.
                with suppress_print():
                    self.animator.setup_training_video(tmp.name, fps=10, dpi=80)
            self.assertIsNotNone(self.animator.writer)
            self.assertTrue(hasattr(self.animator.writer, "setup_called"))
            self.assertEqual(self.animator.frame_count, 0)
            self.assertIsNotNone(self.animator.bg_cache)

    def test_setup_training_video_no_initialize(self):
        """Test that setup_training_video raises an error if initialize() wasn't called."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            with self.assertRaises(ValueError):
                with suppress_print():
                    self.animator.setup_training_video(tmp.name, fps=10)

    def test_add_training_frame(self):
        """Test that add_training_frame updates frame_count and calls writer.grab_frame."""
        metrics = ["loss"]
        self.animator.initialize(metrics_to_track=metrics, has_validation=False)
        # Simulate some metric data.
        self.animator.metrics["loss"] = [1.0, 0.8]
        # Use a dummy writer.
        dummy_writer = DummyWriter()
        self.animator.writer = dummy_writer
        initial_frame_count = self.animator.frame_count
        self.animator.add_training_frame()
        self.assertEqual(self.animator.frame_count, initial_frame_count + 1)
        self.assertTrue(dummy_writer.grab_called)
        # Verify that if writer is None, a ValueError is raised.
        self.animator.writer = None
        with self.assertRaises(ValueError):
            self.animator.add_training_frame()

    def test_finish_training_video(self):
        """Test that finish_training_video calls writer.finish and resets state."""
        metrics = ["loss"]
        self.animator.initialize(metrics_to_track=metrics, has_validation=False)
        dummy_writer = DummyWriter()
        self.animator.writer = dummy_writer
        self.animator.frame_count = 5
        self.animator.finish_training_video()
        self.assertTrue(dummy_writer.finish_called)
        self.assertIsNone(self.animator.writer)
        self.assertEqual(self.animator.frame_count, 0)
        # Verify that if writer is None, a ValueError is raised.
        with self.assertRaises(ValueError):
            self.animator.finish_training_video()

    def test_setup_training_video_fallback(self):
        """
        Test the fallback mechanism in setup_training_video when FFMpegWriter is unavailable.
        It simulates a FileNotFoundError and checks that PillowWriter is used instead.
        """
        self.animator.initialize(metrics_to_track=["loss"], has_validation=False)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            # Patch FFMpegWriter from matplotlib.animation to raise FileNotFoundError.
            with patch("matplotlib.animation.FFMpegWriter", side_effect=FileNotFoundError("Not found")):
                # Patch PillowWriter from matplotlib.animation to return DummyWriter.
                with patch("matplotlib.animation.PillowWriter", return_value=DummyWriter()) as mock_pillow:
                    with suppress_print():
                        self.animator.setup_training_video(tmp.name, fps=10, dpi=80)
                    self.assertIsNotNone(self.animator.writer)
                    self.assertTrue(self.animator.writer.setup_called)

if __name__ == '__main__':
    unittest.main()