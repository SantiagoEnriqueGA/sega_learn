import os
import sys
import warnings

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series.exponential_smoothing import SimpleExponentialSmoothing
from sega_learn.utils import Metrics, make_time_series

warnings.filterwarnings("ignore", category=UserWarning)
mean_squared_error = Metrics.mean_squared_error

# Generate a synthetic time series
time_series = make_time_series(
    n_samples=1,
    n_timestamps=300,
    n_features=1,
    trend="linear",
    seasonality=None,
    noise=0.1,
    random_state=1,
)

# Flatten the time series to 1D
time_series = time_series.flatten()

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# List of alpha values to evaluate
alpha_values = [0.25, 0.5, 0.75]

plt.figure(figsize=(12, 12))

for i, alpha in enumerate(alpha_values, start=1):
    # Initialize and fit the Simple Exponential Smoothing model
    ses_model = SimpleExponentialSmoothing(alpha=alpha)
    fitted_values = ses_model.fit(train_series)

    # Forecast future values using the Simple Exponential Smoothing model
    forecast_steps = len(test_series)
    forecasted_values_ses = ses_model.forecast(steps=forecast_steps)

    # Evaluate the Simple Exponential Smoothing model
    mse_ses = mean_squared_error(test_series, forecasted_values_ses)
    print(
        f"Alpha {alpha}: Simple Exponential Smoothing Mean Squared Error: {mse_ses:.4f}"
    )

    # Plot the forecasted values
    plt.subplot(len(alpha_values), 1, i)
    plt.plot(range(len(time_series)), time_series, label="Original Time Series")
    plt.plot(
        range(len(train_series), len(train_series) + len(forecasted_values_ses)),
        forecasted_values_ses,
        label=f"Forecast (Alpha={alpha})",
        linestyle="--",
    )
    plt.plot(
        range(len(train_series)),
        fitted_values,
        label="Fitted Values",
        # linestyle=":",
    )
    plt.axvline(
        x=len(train_series), color="black", linestyle="--", label="Forecast Start"
    )
    plt.title(f"Simple Exponential Smoothing Forecast (Alpha={alpha})")
    plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("examples/time_series/plots/ses_forecast_comparison.png", dpi=300)
