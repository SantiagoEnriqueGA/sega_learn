import os
import sys
import warnings

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series.exponential_smoothing import DoubleExponentialSmoothing
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

des_model = DoubleExponentialSmoothing(alpha=0.5, beta=0.25)

# Find the best alpha and beta values combination
alpha, beta = des_model.find_best_alpha_beta(train_series, test_series, set_best=True)
print(f"Best Alpha: {alpha}, Best Beta: {beta}")

# Fit the model with the best alpha and beta values
fitted_values = des_model.fit(train_series)

# Forecast future values using the Double Exponential Smoothing model
forecast_steps = len(test_series)
forecasted_values_des = des_model.forecast(steps=forecast_steps)

# Evaluate the Double Exponential Smoothing model
mse_des = mean_squared_error(test_series, forecasted_values_des)
print(f"Double Exponential Smoothing Mean Squared Error: {mse_des:.4f}")

# Plot the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(range(len(time_series)), time_series, label="Original Time Series")
plt.plot(
    range(len(train_series), len(train_series) + len(forecasted_values_des)),
    forecasted_values_des,
    label=f"Forecast (Alpha={alpha}, Beta={beta})",
    linestyle="--",
)
plt.plot(
    range(len(train_series)),
    fitted_values,
    label="Fitted Values",
    # linestyle=":",
)
plt.axvline(x=len(train_series), color="black", linestyle="--", label="Forecast Start")
plt.title("Double Exponential Smoothing Forecast")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("examples/time_series/plots/smth_des_forecast_comparison.png", dpi=300)
