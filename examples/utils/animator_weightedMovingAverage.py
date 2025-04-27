import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series.moving_average import WeightedMovingAverage
from sega_learn.utils import Metrics, make_time_series
from sega_learn.utils.animator import ForcastingAnimation

warnings.filterwarnings("ignore", category=UserWarning)

# Generate a synthetic time series
time_series = make_time_series(
    n_samples=1,
    n_timestamps=300,
    n_features=1,
    trend="linear",
    seasonality="sine",
    seasonality_period=30,
    noise=0.1,
    random_state=1,
)

# Flatten the time series to 1D if it's not already
time_series = time_series.flatten()

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]
forecast_steps = len(test_series)

# Create the animation using ForcastingAnimation
animator = ForcastingAnimation(
    model=WeightedMovingAverage,
    train_series=train_series,
    test_series=test_series,
    forecast_steps=forecast_steps,
    keep_previous=True,
    dynamic_parameter="window",
    metric_fn=[
        Metrics.mean_squared_error,
        Metrics.mean_absolute_error,
    ],
)

# Set up the plot
animator.setup_plot(
    title="Weighted Moving Average Forecast",
    xlabel="Time Step",
    ylabel="Value",
    legend_loc="upper left",
    grid=True,
    figsize=(12, 6),
)

# Create and save the animation
window_range = range(1, 51)  # Windows from 1 to 50
animator.animate(frames=window_range, interval=150, blit=True, repeat=False)
animator.save(
    filename="examples/utils/plots/animator_wma_forecast_animation.gif",
    writer="pillow",
    fps=5,
    dpi=100,
)

# To show the animation
# animator.show()
