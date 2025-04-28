import os
import sys
import warnings

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.pipelines import ForecastingPipeline
from sega_learn.time_series import SARIMA, WeightedMovingAverage
from sega_learn.utils import Metrics, make_time_series

warnings.filterwarnings("ignore", category=UserWarning)

# Generate a synthetic time series
time_series = make_time_series(
    n_samples=1,
    n_timestamps=300,
    n_features=1,
    trend="linear",
    seasonality="cosine",
    seasonality_period=100,
    noise=0.5,
    random_state=1,
)

# Flatten the time series to 1D
time_series = time_series.flatten()

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# Initialize the forecasting pipeline
order = (2, 1, 2)
seasonal_order = (1, 1, 1, 100)
pipeline = ForecastingPipeline(
    preprocessors=[
        WeightedMovingAverage(window=5),
        # Add more preprocessors as needed
        # ExponentialMovingAverage(alpha=0.2),
    ],
    model=[
        SARIMA(order=order, seasonal_order=seasonal_order),
        # Add more models as needed
        # SARIMA(order=order, seasonal_order=seasonal_order),
    ],
    evaluators=[
        Metrics.mean_squared_error,
        Metrics.mean_absolute_error,
        Metrics.mean_absolute_percentage_error,
        # Add more evaluators as needed
        # Metrics.mean_squared_log_error,
    ],
)

# Fit the pipeline to the training data
pipeline.fit(train_series)

# Make predictions on the test data
pipeline_predictions = pipeline.predict(test_series, steps=len(test_series))

# Print the pipeline summary
pipeline.summary()

# Evaluate the pipeline model
results = pipeline.evaluate(pipeline_predictions, test_series)
print("\nPipeline Results:")
for metric, value in results.items():
    print(f"\t{metric:30}: {value:.4f}")

# Initialize and fit the SARIMA, without pipeline
sarima_model = SARIMA(order=order, seasonal_order=seasonal_order)
sarima_model.fit(train_series)
forecasted_base = sarima_model.forecast(steps=len(test_series))
mse_base = Metrics.mean_squared_error(test_series, forecasted_base)
print(f"\nBase SARIMA Mean Squared Error: {mse_base:.4f}")

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(range(len(time_series)), time_series, label="Original Time Series")
plt.plot(
    range(len(train_series), len(train_series) + len(pipeline_predictions)),
    pipeline_predictions,
    label="Forecasted Values (Pipeline)",
    linestyle="--",
)
plt.plot(
    range(len(train_series), len(train_series) + len(forecasted_base)),
    forecasted_base,
    label="Forecasted Values (Base)",
    linestyle=":",
)
# Add a vertical line to indicate where the forecast starts
plt.axvline(x=len(train_series), color="black", linestyle="--", label="Forecast Start")
plt.title("ARIMA Forecast Comparison")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("examples/pipelines/plots/forecastingPipelineine.png")
