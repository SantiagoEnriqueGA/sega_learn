import sys
from abc import ABC, abstractmethod

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .dataSplitting import train_test_split
from .decomposition import PCA

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

    def __init__(
        self,
        model,
        train_series,
        test_series,
        dynamic_parameter=None,
        static_parameters=None,
        keep_previous=None,
        **kwargs,
    ):
        """Initialize the animation base class.

        Args:
            model: The forecasting model or any machine learning model.
            train_series: Training time series data.
            test_series: Testing time series data.
            dynamic_parameter: The parameter to update dynamically (e.g., 'window', 'alpha', 'beta').
            static_parameters: Static parameters for the model.
                Should be a dictionary with parameter names as keys and their values.
            keep_previous: Whether to keep all previous lines with reduced opacity.
            **kwargs: Additional customization options (e.g., colors, line styles).
        """
        if dynamic_parameter is None:
            raise ValueError("dynamic_parameter must be provided.")

        self.model = model
        self.train_data = train_series
        self.test_data = test_series
        self.dynamic_parameter = dynamic_parameter  # Parameter to update dynamically
        self.static_parameters = (
            static_parameters if static_parameters is not None else {}
        )
        self.keep_previous = keep_previous
        self.kwargs = kwargs

        # Optional metric function (e.g., MSE)
        self.metric_fn = kwargs.get("metric_fn")
        # If self.metric_fn is not a list, convert it to a list
        if self.metric_fn and not isinstance(self.metric_fn, list):
            self.metric_fn = [self.metric_fn]

        # Plot elements
        self.fig, self.ax = None, None
        self.lines = {}
        self.title = None

    def setup_plot(
        self, title, xlabel, ylabel, legend_loc="upper left", grid=True, figsize=(12, 6)
    ):
        """Set up the plot for the animation.

        Args:
            title: Title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            legend_loc: Location of the legend.
            grid: Whether to show grid lines.
            figsize: Size of the figure.
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if legend_loc is not None:
            # self.ax.legend(loc=legend_loc)
            # Will call legend() in update_plot() to update the legend
            self.add_legend = True
        else:
            self.add_legend = False
        self.ax.grid(grid)
        plt.tight_layout()

    @abstractmethod
    def update_model(self, frame):
        """Abstract method to update the model for a given frame. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def update_plot(self, frame):
        """Abstract method to update the plot for a given frame.Must be implemented by subclasses."""
        pass

    def animate(self, frames, interval=150, blit=True, repeat=False):
        """Create the animation.

        Args:
            frames: Range of frames (e.g., window sizes).
            interval: Delay between frames in milliseconds.
            blit: Whether to use blitting for faster rendering.
            repeat: Whether to repeat the animation.
        """

        def _update(frame):
            self.update_model(frame)
            return self.update_plot(frame)

        self.ani = animation.FuncAnimation(
            self.fig,
            _update,
            frames=frames,
            interval=interval,
            blit=blit,
            repeat=repeat,
        )

        return self.ani

    def save(self, filename, writer="pillow", fps=5, dpi=100):
        """Save the animation to a file.

        Args:
            filename: Path to save the animation.
            writer: Writer to use (e.g., 'pillow' for GIF).
            fps: Frames per second.
            dpi: Dots per inch for the saved figure.
        """
        if not hasattr(self, "ani"):
            raise RuntimeError("Animation has not been created. Call `animate` first.")
        self.ani.save(filename, writer=writer, fps=fps, dpi=dpi)

        sys.stdout.write("\033[K")  # Clear the line
        print(f"Animation saved successfully to {filename}.")

    def show(self):
        """Display the animation."""
        if not hasattr(self, "ani"):
            raise RuntimeError("Animation has not been created. Call `animate` first.")
        plt.show()
        print("Animation displayed successfully.")


class ForcastingAnimation(AnimationBase):
    """Class for creating animations of forecasting models."""

    def __init__(
        self,
        model,
        train_series,
        test_series,
        forecast_steps,
        dynamic_parameter=None,
        static_parameters=None,
        keep_previous=False,
        **kwargs,
    ):
        """Initialize the forecasting animation class.

        Args:
            model: The forecasting model.
            train_series: Training time series data.
            test_series: Testing time series data.
            forecast_steps: Number of steps to forecast.
            dynamic_parameter: The parameter to update dynamically (e.g., 'window', 'alpha', 'beta').
            static_parameters: Static parameters for the model.
                Should be a dictionary with parameter names as keys and their values.
            keep_previous: Whether to keep all previous lines with reduced opacity.
            **kwargs: Additional customization options (e.g., colors, line styles).
        """
        super().__init__(
            model,
            train_series,
            test_series,
            dynamic_parameter,
            static_parameters,
            keep_previous,
            **kwargs,
        )
        self.forecast_steps = forecast_steps
        if self.keep_previous:
            self.previous_forecast_lines = []  # List to store previous forecast lines
            self.previous_fitted_lines = []  # List to store previous fitted lines

        # Initialize plot elements
        self.train_indices = range(len(train_series))
        self.forecast_indices = range(
            len(train_series), len(train_series) + forecast_steps
        )
        self.fitted_line = None
        self.forecast_line = None

    def setup_plot(
        self, title, xlabel, ylabel, legend_loc="upper left", grid=True, figsize=(12, 6)
    ):
        """Set up the plot for forecasting animation."""
        super().setup_plot(title, xlabel, ylabel, legend_loc, grid, figsize)

        # Plot static elements
        self.ax.plot(
            self.train_indices, self.train_data, label="Training Data", color="blue"
        )
        self.ax.axvline(
            x=len(self.train_data),
            color="black",
            linestyle="--",
            label="Forecast Start",
        )

        # Create placeholders for dynamic lines
        (self.fitted_line,) = self.ax.plot([], [], label="Fitted Values", color="green")
        (self.forecast_line,) = self.ax.plot(
            [], [], label="Forecast", linestyle="--", color="red"
        )

        # Auto-adjust y-limits based on the training data range
        min_y = min(self.train_data) - 0.5 * (
            max(self.train_data) - min(self.train_data)
        )
        max_y = max(self.train_data) + 0.5 * (
            max(self.train_data) - min(self.train_data)
        )
        self.ax.set_ylim(min_y, max_y)

        if self.add_legend:
            # Add legend to the plot
            self.ax.legend(loc=legend_loc)

    def update_model(self, frame):
        """Update the model for the current frame.

        Args:
            frame: The current frame (e.g., parameter value).
        """
        # Dynamically update the model with the current frame and include static parameters
        self.model_instance = self.model(
            **{self.dynamic_parameter: frame}, **self.static_parameters
        )
        self.fitted_values = self.model_instance.fit(self.train_data)
        self.forecast_values = self.model_instance.forecast(steps=self.forecast_steps)

    def update_plot(self, frame):
        """Update the plot for the current frame.

        Args:
            frame: The current frame (e.g., parameter value).
        """
        if self.keep_previous and self.forecast_line and self.fitted_line:
            # For all previous forecast lines, set alpha from 0.1 to 0.5 based on the number of lines
            self.previous_forecast_lines.append(self.forecast_line)
            for i, line in enumerate(self.previous_forecast_lines):
                line.set_alpha(0.1 + (0.4 / len(self.previous_forecast_lines)) * i)
                line.set_color("lightcoral")

            # Add a new forecast line
            (self.forecast_line,) = self.ax.plot(
                [], [], label="Forecast", linestyle="--", color="red"
            )

            # For all previous fitted lines, set alpha from 0.1 to 0.5 based on the number of lines
            self.previous_fitted_lines.append(self.fitted_line)
            for i, line in enumerate(self.previous_fitted_lines):
                line.set_alpha(0.1 + (0.4 / len(self.previous_fitted_lines)) * i)
                line.set_color("lightgreen")

            # Add a new fitted line
            (self.fitted_line,) = self.ax.plot(
                [], [], label="Fitted Values", color="green"
            )

        # Update the dynamic lines with the latest fitted and forecasted values
        self.fitted_line.set_data(self.train_indices, self.fitted_values)
        self.forecast_line.set_data(self.forecast_indices, self.forecast_values)

        # Update the title with the current frame and optional metric
        if self.metric_fn:
            if len(self.metric_fn) == 1:
                # If only one metric function is provided, use it directly
                metric_value = self.metric_fn[0](self.test_data, self.forecast_values)

                # Trim values
                metric_value = round(metric_value, 4)
                frame = round(frame, 2)

                self.ax.set_title(
                    f"Forecast ({self.dynamic_parameter}={frame}) - {self.metric_fn[0].__name__.capitalize()}: {metric_value:.4f}"
                )
                print(
                    f"{self.dynamic_parameter}: {frame}, {self.metric_fn[0].__name__.capitalize()}: {metric_value:.4f}",
                    end="\r",
                )

            else:
                # If multiple metric functions are provided, calculate and display each one
                metrics = [
                    metric_fn(self.test_data, self.forecast_values)
                    for metric_fn in self.metric_fn
                ]
                frame = round(frame, 2)

                self.ax.set_title(
                    f"Forecast ({self.dynamic_parameter}={frame}) - {', '.join([f'{fn.__name__.capitalize()}: {metric:.4f}' for fn, metric in zip(self.metric_fn, metrics, strict=False)])}"
                )
                print(
                    f"{self.dynamic_parameter}: {frame}, {', '.join([f'{fn.__name__.capitalize()}: {metric:.4f}' for fn, metric in zip(self.metric_fn, metrics, strict=False)])}",
                    end="\r",
                )

        else:
            self.ax.set_title(f"Forecast ({self.dynamic_parameter}={frame})")
            print(f"{self.dynamic_parameter}: {frame}", end="\r")

        return [self.fitted_line, self.forecast_line] + self.previous_forecast_lines


class RegressionAnimation(AnimationBase):
    """Class for creating animations of regression models."""

    def __init__(
        self,
        model,
        X,
        y,
        test_size=0.3,
        dynamic_parameter=None,
        static_parameters=None,
        keep_previous=False,
        **kwargs,
    ):
        """Initialize the regression animation class.

        Args:
            model: The regression model.
            X: Feature matrix (input data).
            y: Target vector (output data).
            test_size: Proportion of the dataset to include in the test split.
            dynamic_parameter: The parameter to update dynamically (e.g., 'alpha', 'beta').
            static_parameters: Additional static parameters for the model.
                Should be a dictionary with parameter names as keys and their values.
            keep_previous: Whether to keep all previous lines with reduced opacity.
            **kwargs: Additional customization options (e.g., colors, line styles).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        super().__init__(
            model,
            (X_train, y_train),
            (X_test, y_test),
            dynamic_parameter,
            static_parameters,
            keep_previous,
            **kwargs,
        )

        # Split training and testing data into features and target
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        # Initialize plot elements
        self.scatter_points = None
        self.predicted_line = None

        if self.keep_previous:
            self.previous_predicted_lines = []  # List to store previous predicted lines

    def setup_plot(
        self, title, xlabel, ylabel, legend_loc="upper left", grid=True, figsize=(12, 6)
    ):
        """Set up the plot for regression animation."""
        super().setup_plot(title, xlabel, ylabel, legend_loc, grid, figsize)

        # Check if X_train has more than one feature
        if self.X_train.shape[1] > 1:
            print("Warning: More than one feature detected. Applying PCA.")
            self.X_train, self.X_test = self._apply_pca(self.X_train, self.X_test)

        # Plot static elements (scatter points for training data)
        self.scatter_points = self.ax.scatter(
            self.X_train[:, 0], self.y_train, label="Training Data", color="blue"
        )

        # Create a placeholder for the predicted regression line
        (self.predicted_line,) = self.ax.plot(
            [], [], label="Regression Line", color="red"
        )

        if self.add_legend:
            # Add legend to the plot
            self.ax.legend(loc=legend_loc)

    def _apply_pca(self, X_train, X_test):
        """Apply PCA to reduce dimensionality if X_train has more than one feature."""
        pca = PCA(n_components=1)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        return X_train, X_test

    def update_model(self, frame):
        """Update the regression model for the current frame.

        Args:
            frame: The current frame (e.g., parameter value).
        """
        # Dynamically update the model with the current frame and include static parameters
        self.model_instance = self.model(
            **{self.dynamic_parameter: frame}, **self.static_parameters
        )
        self.model_instance.fit(self.X_train, self.y_train)
        self.predicted_values = self.model_instance.predict(self.X_test)

    def update_plot(self, frame):
        """Update the plot for the current frame.

        Args:
            frame: The current frame (e.g., parameter value).
        """
        if self.keep_previous and self.predicted_line:
            # For all previous predicted lines, set alpha from 0.1 to 0.5 based on the number of lines
            self.previous_predicted_lines.append(self.predicted_line)
            for i, line in enumerate(self.previous_predicted_lines):
                line.set_alpha(0.1 + (0.4 / len(self.previous_predicted_lines)) * i)
                line.set_color("lightcoral")

            # Add a new predicted line
            (self.predicted_line,) = self.ax.plot(
                [], [], label="Regression Line", color="red"
            )

        # Update the regression line with the predicted values
        self.predicted_line.set_data(self.X_test[:, 0], self.predicted_values)

        # Update the title with the current frame and optional metrics
        if self.metric_fn:
            if len(self.metric_fn) == 1:
                # If only one metric function is provided, use it directly
                metric_value = self.metric_fn[0](self.y_test, self.predicted_values)
                metric_value = round(metric_value, 4)
                frame = round(frame, 2)

                self.ax.set_title(
                    f"Regression ({self.dynamic_parameter}={frame}) - {self.metric_fn[0].__name__.capitalize()}: {metric_value:.4f}"
                )
                print(
                    f"{self.dynamic_parameter}: {frame}, {self.metric_fn[0].__name__.capitalize()}: {metric_value:.4f}",
                    end="\r",
                )
            else:
                # If multiple metric functions are provided, calculate and display each one
                metrics = [
                    metric_fn(self.y_test, self.predicted_values)
                    for metric_fn in self.metric_fn
                ]
                frame = round(frame, 2)

                self.ax.set_title(
                    f"Regression ({self.dynamic_parameter}={frame}) - {', '.join([f'{fn.__name__.capitalize()}: {metric:.4f}' for fn, metric in zip(self.metric_fn, metrics, strict=False)])}"
                )
                print(
                    f"{self.dynamic_parameter}: {frame}, {', '.join([f'{fn.__name__.capitalize()}: {metric:.4f}' for fn, metric in zip(self.metric_fn, metrics, strict=False)])}",
                    end="\r",
                )
        else:
            self.ax.set_title(f"Regression ({self.dynamic_parameter}={frame})")
            print(f"{self.dynamic_parameter}: {frame}", end="\r")

        return (self.predicted_line,)


class ClassificationAnimation(AnimationBase):
    """Class for creating animations of classification models."""

    def __init__(
        self,
        model,
        X,
        y,
        test_size=0.3,
        dynamic_parameter=None,
        static_parameters=None,
        keep_previous=False,
        scaler=None,
        **kwargs,
    ):
        """Initialize the classification animation class.

        Args:
            model: The classification model.
            X: Feature matrix (input data).
            y: Target vector (output data).
            test_size: Proportion of the dataset to include in the test split.
            dynamic_parameter: The parameter to update dynamically (e.g., 'alpha', 'beta').
            static_parameters: Additional static parameters for the model.
                Should be a dictionary with parameter names as keys and their values.
            keep_previous: Whether to keep all previous lines with reduced opacity.
            scaler: Optional scaler for preprocessing the data.
            **kwargs: Additional customization options (e.g., colors, line styles).
        """
        if scaler is not None:
            # Apply the scaler to the data
            X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        super().__init__(
            model,
            (X_train, y_train),
            (X_test, y_test),
            dynamic_parameter,
            static_parameters,
            keep_previous,
            **kwargs,
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        # Create mesh grid for decision boundary
        self.x_min, self.x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        self.y_min, self.y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        self.xx, self.yy = None, None

        if self.keep_previous:
            self.previous_decision_lines = []  # Store previous decision boundaries

    def setup_plot(
        self, title, xlabel, ylabel, legend_loc="upper left", grid=True, figsize=(12, 6)
    ):
        """Set up the plot for classification animation."""
        super().setup_plot(title, xlabel, ylabel, legend_loc, grid, figsize)

        # Create mesh grid for decision boundary
        self.xx, self.yy = np.meshgrid(
            np.arange(self.x_min, self.x_max, 0.01),
            np.arange(self.y_min, self.y_max, 0.01),
        )

        # Plot training data points
        for class_value in np.unique(self.y_train):
            self.ax.scatter(
                self.X_train[self.y_train == class_value, 0],
                self.X_train[self.y_train == class_value, 1],
                label=f"Class {class_value}",
                edgecolors="k",
            )

        if self.add_legend:
            self.ax.legend(loc=legend_loc)

    def update_model(self, frame):
        """Update the classification model for the current frame.

        Args:
            frame: The current frame (e.g., parameter value).
        """
        # Dynamically update the model with the current frame and include static parameters
        self.model_instance = self.model(
            **{self.dynamic_parameter: frame}, **self.static_parameters
        )
        self.model_instance.fit(self.X_train, self.y_train)

    def update_plot(self, frame):
        """Update the plot for the current frame.

        Args:
            frame: The current frame (e.g., parameter value).
        """
        # Clear the previous decision boundary if it exists
        if hasattr(self, "decision_boundary") and self.decision_boundary:
            for collection in self.decision_boundary.collections:
                collection.remove()

        # Clear the previous decision boundary lines if they exist
        if hasattr(self, "decision_boundary_lines") and self.decision_boundary_lines:
            if self.keep_previous:
                # For all previous decision boundaries, set alpha from 0.1 to 0.5 based on the number of lines
                self.previous_decision_lines.append(self.decision_boundary_lines)
                for i, collection in enumerate(self.previous_decision_lines):
                    collection.set_alpha(
                        0.1 + (0.4 / len(self.previous_decision_lines)) * i
                    )
                    collection.set_color("black")
            else:
                # Remove previous decision boundary lines
                for collection in self.decision_boundary_lines.collections:
                    collection.remove()

        # Predict on the mesh grid to create decision boundary
        Z = self.model_instance.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
        Z = Z.reshape(self.xx.shape)

        # Plot the current decision boundary contour with filled regions
        self.decision_boundary = self.ax.contourf(
            self.xx,
            self.yy,
            Z,
            alpha=0.25,
            cmap="coolwarm",
        )

        # If only two classes, plot the decision boundary lines
        if len(np.unique(self.y_train)) == 2:
            # Plot decision boundary lines
            self.decision_boundary_lines = self.ax.contour(
                self.xx,
                self.yy,
                Z,
                levels=[0.5],
                linewidths=1,
                colors="black",
            )

        # Update the title with the current frame and optional metrics
        if self.metric_fn:
            if len(self.metric_fn) == 1:
                # If only one metric function is provided, use it directly
                metric_value = self.metric_fn[0](
                    self.y_test, self.model_instance.predict(self.X_test)
                )
                metric_value = round(metric_value, 4)
                frame = round(frame, 2)

                self.ax.set_title(
                    f"Classification ({self.dynamic_parameter}={frame}) - {self.metric_fn[0].__name__.capitalize()}: {metric_value:.4f}"
                )
                print(
                    f"{self.dynamic_parameter}: {frame}, {self.metric_fn[0].__name__.capitalize()}: {metric_value:.4f}",
                    end="\r",
                )
            else:
                # If multiple metric functions are provided, calculate and display each one
                metrics = [
                    metric_fn(self.y_test, self.model_instance.predict(self.X_test))
                    for metric_fn in self.metric_fn
                ]
                frame = round(frame, 2)

                self.ax.set_title(
                    f"Classification ({self.dynamic_parameter}={frame}) - {', '.join([f'{fn.__name__.capitalize()}: {metric:.4f}' for fn, metric in zip(self.metric_fn, metrics, strict=False)])}"
                )
                print(
                    f"{self.dynamic_parameter}: {frame}, {', '.join([f'{fn.__name__.capitalize()}: {metric:.4f}' for fn, metric in zip(self.metric_fn, metrics, strict=False)])}",
                    end="\r",
                )
        else:
            self.ax.set_title(f"Classification ({self.dynamic_parameter}={frame})")
            print(f"{self.dynamic_parameter}: {frame}", end="\r")

        if len(np.unique(self.y_train)) == 2:
            return (
                self.decision_boundary,
                self.decision_boundary_lines,
            )
        else:
            return (self.decision_boundary,)
