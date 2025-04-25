import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
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
exog = time_series[..., 1].flatten().reshape(-1, 1)
time_series = time_series[..., 0].flatten()

# Add random noise to the exogenous variable
exog += np.random.normal(0, 0.25, size=exog.shape)

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
y_train, y_test = time_series[:train_size], time_series[train_size:]
X_train, X_test = exog[:train_size], exog[train_size:]


# Find the best SARIMAX order using the custom ARIMA class
order, seasonal_order = SARIMAX.suggest_order(y_train, X_train)
print(f"Suggested SARIMAX order: {order, seasonal_order}")

order, seasonal_orders = SARIMAX.find_best_order(y_train, y_test, X_train, X_test)
print(f"Best SARIMAX order: {order, seasonal_orders}")

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
plt.plot(range(len(time_series)), time_series, label="Original y")
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
plt.savefig("examples/time_series/plots/arima_sarimax.png", dpi=300)
