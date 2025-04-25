import os
import sys
import warnings

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series.moving_average import ExponentialMovingAverage
from sega_learn.utils import Metrics, make_time_series

warnings.filterwarnings("ignore", category=UserWarning)
mean_squared_error = Metrics.mean_squared_error

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

# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the static elements
ax.plot(
    range(len(time_series)), time_series, label="Original Time Series", color="blue"
)
ax.axvline(x=train_size, color="black", linestyle="--", label="Forecast Start")

# Placeholders for the dynamic lines (forecast and fitted)
# Initialize with empty data or data from the first frame
initial_alpha = 0.01
ema_model_init = ExponentialMovingAverage(alpha=initial_alpha)
fitted_values_init = ema_model_init.fit(train_series)
forecasted_values_init = ema_model_init.forecast(steps=forecast_steps)

# Indices for plotting
train_indices = range(len(train_series))
forecast_indices = range(train_size, train_size + forecast_steps)

# Create the line objects that will be updated
(fitted_line,) = ax.plot(
    train_indices, fitted_values_init, label="Fitted Values", color="green"
)
(forecast_line,) = ax.plot(
    forecast_indices,
    forecasted_values_init,
    label="Forecast",
    linestyle="--",
    color="red",
)

# Set plot limits and labels
ax.set_xlim(0, len(time_series))
# Auto-adjust y-limits based on the full time series range, add some padding
min_y = np.min(time_series) - np.std(time_series) * 0.5
max_y = np.max(time_series) + np.std(time_series) * 0.5
ax.set_ylim(min_y, max_y)

ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
title = ax.set_title(f"Exponential Moving Average Forecast (Alpha={initial_alpha})")
ax.legend(loc="upper left")  # Fix legend location
ax.grid(True)
plt.tight_layout()

# Define the alpha range for the animation
alpha_range = np.arange(0.01, 0.5, 0.01)


# Update function for the animation
def update(alpha):
    """Update the plot with the new alpha value."""
    # Clip alpha to two decimal places
    alpha = round(alpha, 2)

    # Create and fit the model for the current window size
    ema_model = ExponentialMovingAverage(alpha=alpha)
    fitted_values = ema_model.fit(train_series)
    forecasted_values = ema_model.forecast(steps=forecast_steps)

    # Update the data of the lines
    # Handle NaNs potentially returned by fit for the first few points
    fitted_line.set_ydata(fitted_values)
    forecast_line.set_ydata(forecasted_values)

    # Update the title
    title.set_text(f"Exponential Moving Average Forecast (Alpha={alpha})")

    # Print MSE for tracking
    ema_des = mean_squared_error(test_series, forecasted_values)
    print(
        f"Alpha: {alpha}, MSE: {ema_des:.4f}", end="\r"
    )  # Use '\r' to overwrite the line

    # Return the updated line objects and title
    return fitted_line, forecast_line, title


# Create the animation
# Set blit=True for potentially faster rendering (if backend supports it)
# interval is the delay between frames in milliseconds
ani = animation.FuncAnimation(
    fig, update, frames=alpha_range, interval=150, blit=True, repeat=False
)

# --- Saving the Animation ---
animation_filename = os.path.join(
    "examples/time_series/plots/mvg_ema_forecast_animation.gif"
)
# Use pillow for GIF
ani.save(animation_filename, writer="pillow", fps=5)
print("\nAnimation saved successfully.")


# Close the plot display if running as a script without interactive showing
plt.close(fig)
