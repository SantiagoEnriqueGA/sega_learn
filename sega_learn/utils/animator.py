from abc import ABC, abstractmethod

# Animation Class
# The goal is to create a reusable and modular animation class that can handle animations for any model and dataset.

# Requirements
#   Modularity:          The class should be reusable for different models and datasets.
#                        Should have base class and subclasses for specific models types (regression, classification, forcasting).
#   Customizability:     Allow users to customize plot elements (e.g., colors, labels, titles).
#   Ease of Use:         Provide a simple interface for creating animations.
#   Support for Metrics: Include functionality to calculate and display metrics like MSE.
#   Saving Options:      Allow saving animations in different formats (e.g., GIF, MP4).
#   Dynamic Updates:     Support dynamic updates of model parameters (e.g., window size).
#   Plot Styling:        Provide options for grid, legends, axis limits, etc.


# High-level Design
#   Base Class:         AnimationBase
#     - Common attributes and methods for all animations.
#     - Methods for setting up the plot, updating the plot, and saving the animation.
#     - Abstract methods for model-specific updates (e.g., update_model, update_plot).
#   Subclasses:         RegressionAnimation, ClassificationAnimation, ForecastingAnimation
#     - Inherit from AnimationBase and implement model-specific updates.
#     - Each subclass can have its own attributes and methods specific to the model type.


class AnimationBase(ABC):
    """Base class for creating animations of machine learning models."""

    def __init__(  # noqa: B027
        self,
        model,
        train_series,
        test_series,
        dynamic_parameter=None,
        static_parameters=None,
        keep_previous=None,
        **kwargs,
    ):
        """Initialize the animation base class."""

    @abstractmethod
    def update_model(self, frame):
        """Abstract method to update the model for a given frame. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def update_plot(self, frame):
        """Abstract method to update the plot for a given frame.Must be implemented by subclasses."""
        pass
