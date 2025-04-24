import os
import sys
import warnings

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series import SARIMA
from sega_learn.utils import Metrics, make_time_series

warnings.filterwarnings("ignore", category=UserWarning)
mean_squared_error = Metrics.mean_squared_error


# Generate a synthetic time series
time_series = make_time_series(
    n_samples=1,
    n_timestamps=300,
    n_features=1,
    trend="linear",
    seasonality="cosine",
    seasonality_period=25,
    noise=0.2,
    random_state=1,
)

# Flatten the time series to 1D
time_series = time_series.flatten()

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# Custom order and seasonal_order
order = (2, 1, 2)
seasonal_order = (1, 1, 1, 25)

# Initialize and fit the custom SARIMA model
sarima_model = SARIMA(order=order, seasonal_order=seasonal_order)
sarima_model.fit(train_series)

# Forecast future values using the custom SARIMA model
forecast_steps = len(test_series)
forecasted_values_custom = sarima_model.forecast(steps=forecast_steps)

# Evaluate the custom SARIMA model
mse_custom = mean_squared_error(test_series, forecasted_values_custom)
print(f"Custom SARIMA Mean Squared Error: {mse_custom:.4f}")

# Initialize and fit the Statsmodels SARIMA model
statsmodels_arima_model = StatsmodelsARIMA(
    train_series, order=order, seasonal_order=seasonal_order
)
statsmodels_arima_model_fit = statsmodels_arima_model.fit()
forecasted_values_statsmodels = statsmodels_arima_model_fit.forecast(
    steps=forecast_steps
)

# Evaluate the Statsmodels SARIMA model
mse_statsmodels = mean_squared_error(test_series, forecasted_values_statsmodels)
print(f"Statsmodels SARIMA Mean Squared Error: {mse_statsmodels:.4f}")

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(range(len(time_series)), time_series, label="Original Time Series")
plt.plot(
    range(len(train_series), len(train_series) + len(forecasted_values_custom)),
    forecasted_values_custom,
    label="Custom SARIMA Forecast",
    linestyle="--",
)
plt.plot(
    range(len(train_series), len(train_series) + len(forecasted_values_statsmodels)),
    forecasted_values_statsmodels,
    label="Statsmodels SARIMA Forecast",
    linestyle=":",
)
# Add a vertical line to indicate where the forecast starts
plt.axvline(x=len(train_series), color="black", linestyle="--", label="Forecast Start")
plt.title("SARIMA Forecast Comparison")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("examples/time_series/plots/arima_sarima_forecast_comparison.png", dpi=300)
