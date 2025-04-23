import os
import sys
import warnings

import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX as StatsmodelsSARIMAX

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series import SARIMAX
from sega_learn.utils import Metrics, make_time_series

warnings.filterwarnings("ignore", category=UserWarning)
mean_squared_error = Metrics.mean_squared_error

# Generate a synthetic time series
time_series = make_time_series(
    n_samples=1,
    n_timestamps=300,
    n_features=2,
    trend="linear",
    seasonality="cosine",
    seasonality_period=25,
    noise=0.2,
    random_state=1,
)
# Split into target y and exogenous X
y = time_series[..., 0].flatten()
X = time_series[..., 1].flatten().reshape(-1, 1)

# Split into training and testing sets
train_size = int(len(y) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]


# Find the best SARIMAX order using the custom ARIMA class
order, seasonal_order = SARIMAX.suggest_order(y_train, X_train)
print(f"Suggested ARIMA order: {order, seasonal_order}")

order, seasonal_orders = SARIMAX.find_best_order(y_train, y_test, X_train, X_test)
print(f"Best ARIMA order: {order, seasonal_orders}")

# Specify non-seasonal & seasonal orders
# order = (2, 1, 2)
# seasonal_order = (1, 1, 1, 25)
forecast_steps = len(y_test)

# Initialize and fit the custom SARIMAX model
model = SARIMAX(order=order, seasonal_order=seasonal_order)
model.fit(y_train, X_train)
y_pred_custom = model.forecast(steps=forecast_steps, exog_future=X_test)

# Evaluate the custom SARIMAX model
mse_custom = mean_squared_error(y_test, y_pred_custom)
print(f"Custom SARIMAX Mean Squared Error: {mse_custom:.4f}")

# Initialize and fit the Statsmodels SARIMAX model
sm_model = StatsmodelsSARIMAX(
    endog=y_train,
    exog=X_train,
    order=order,
    seasonal_order=seasonal_order,
)
sm_fit = sm_model.fit(disp=False)

# Evaluate the Statsmodels SARIMAX model
y_pred_sm = sm_fit.predict(
    start=train_size, end=train_size + forecast_steps - 1, exog=X_test
)
mse_sm = mean_squared_error(y_test, y_pred_sm)
print(f"Statsmodels SARIMAX Mean Squared Error: {mse_sm:.4f}")

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(range(len(y)), y, label="Original y")
plt.plot(
    range(train_size, train_size + forecast_steps),
    y_pred_custom,
    "--",
    label="Custom SARIMAX",
)
plt.plot(
    range(train_size, train_size + forecast_steps),
    y_pred_sm,
    ":",
    label="Statsmodels SARIMAX",
)
plt.axvline(x=train_size, color="k", linestyle="--", label="Forecast start")
plt.title("SARIMAX Forecast Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("examples/time_series/plots/sarimax_forecast_comparison.png", dpi=300)
