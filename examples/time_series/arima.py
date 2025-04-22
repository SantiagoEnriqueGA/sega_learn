import os
import sys

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series import ARIMA
from sega_learn.utils import Metrics, make_time_series

mean_squared_error = Metrics.mean_squared_error


# Generate a synthetic time series
time_series = make_time_series(
    n_samples=1,
    n_timestamps=100,
    n_features=1,
    trend="linear",
    seasonality="cosine",
    # seasonality_period=10,
    noise=0.1,
    random_state=1,
)

# Flatten the time series to 1D
time_series = time_series.flatten()

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# Find the best ARIMA order using the custom ARIMA class
order = ARIMA.suggest_order(train_series)
print(f"Suggested ARIMA order: {order}")

order = ARIMA.find_best_order(train_series, test_series)
print(f"Best ARIMA order: {order}")


# Initialize and fit the custom ARIMA model
# order = (2, 2, 4)  # Adjusted ARIMA(p, d, q) order
arima_model = ARIMA(order=order)
arima_model.fit(train_series)

# Forecast future values using the custom ARIMA model
forecast_steps = len(test_series)
forecasted_values_custom = arima_model.forecast(steps=forecast_steps)

# Evaluate the custom ARIMA model
mse_custom = mean_squared_error(test_series, forecasted_values_custom)
print(f"Custom ARIMA Mean Squared Error: {mse_custom:.4f}")

# Initialize and fit the Statsmodels ARIMA model
statsmodels_arima_model = StatsmodelsARIMA(train_series, order=order)
statsmodels_arima_model_fit = statsmodels_arima_model.fit()

# Forecast future values using the Statsmodels ARIMA model
forecasted_values_statsmodels = statsmodels_arima_model_fit.forecast(
    steps=forecast_steps
)

# Evaluate the Statsmodels ARIMA model
mse_statsmodels = mean_squared_error(test_series, forecasted_values_statsmodels)
print(f"Statsmodels ARIMA Mean Squared Error: {mse_statsmodels:.4f}")

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(range(len(time_series)), time_series, label="Original Time Series")
plt.plot(
    range(len(train_series), len(train_series) + len(forecasted_values_custom)),
    forecasted_values_custom,
    label="Custom ARIMA Forecast",
    linestyle="--",
)
plt.plot(
    range(len(train_series), len(train_series) + len(forecasted_values_statsmodels)),
    forecasted_values_statsmodels,
    label="Statsmodels ARIMA Forecast",
    linestyle=":",
)
# Add a vertical line to indicate where the forecast starts
plt.axvline(x=len(train_series), color="black", linestyle="--", label="Forecast Start")
plt.title("ARIMA Forecast Comparison")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("examples/time_series/plots/arima_forecast_comparison.png", dpi=300)
