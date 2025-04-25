import os
import sys
import warnings

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series.moving_average import WeightedMovingAverage
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

# Flatten the time series to 1D
time_series = time_series.flatten()

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

wma_model = WeightedMovingAverage(window=30)

# Fit the model with the best alpha, beta, and gamma values
fitted_values = wma_model.fit(train_series)

# Forecast future values using the Weighted Moving Average model
forecast_steps = len(test_series)
forecasted_values_des = wma_model.forecast(steps=forecast_steps)

# Evaluate the Weighted Moving Average model
mse_des = mean_squared_error(test_series, forecasted_values_des)
print(f"Weighted Moving Average Mean Squared Error: {mse_des:.4f}")

# Plot the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(range(len(time_series)), time_series, label="Original Time Series")
plt.plot(
    range(len(train_series), len(train_series) + len(forecasted_values_des)),
    forecasted_values_des,
    label="Forecast (Window=30)",
    linestyle="--",
)
plt.plot(
    range(len(train_series)),
    fitted_values,
    label="Fitted Values",
    # linestyle=":",
)
plt.axvline(x=len(train_series), color="black", linestyle="--", label="Forecast Start")
plt.title("Weighted Moving Average Forecast")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("examples/time_series/plots/mvg_wma_forecast.png", dpi=300)
