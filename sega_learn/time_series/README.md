# Time Series Module

This module provides a comprehensive set of tools for analyzing, modeling, and forecasting time series data. It includes implementations of popular time series models such as ARIMA, Exponential Smoothing, and Moving Averages, as well as decomposition techniques to extract trend, seasonality, and residual components. These tools are designed to help users understand the underlying patterns in their data and make accurate predictions for future time points.

The module is structured to support both classical statistical methods and modern machine learning approaches, making it suitable for a wide range of applications, including finance, telecommunications, and demand forecasting.

### Directory Structure
```
sega_learn/
└─ time_series/
   ├─ __init__.py
   ├─ arima.py
   ├─ decomposition.py
   ├─ forecasting.py
   ├─ exponential_smoothing.py
   └─ moving_average.py
```

## Forcasting Pipeline

The `ForecastingPipeline` class provides a flexible and modular way to build time series forecasting workflows. It allows users to integrate preprocessing steps, forecasting models, and evaluation metrics into a single pipeline.

### Features
- **Preprocessors**: Add or remove preprocessing steps to transform the input data.
- **Model**: Integrate any forecasting model (e.g., ARIMA, SARIMA) that implements `fit` and `predict` methods.
- **Evaluators**: Add or remove evaluation metrics to assess the model's performance.

### Example Usage
```python
from sega_learn.time_series import *
from sega_learn.utils import make_time_series

# Generate time series
time_series = make_time_series(n_samples=1, n_timestamps=300, n_features=1)

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# Initialize the pipeline
pipeline = ForecastingPipeline(
    preprocessors=[WeightedMovingAverage()],  # Add preprocessing steps if needed
    model=ARIMA(order=(1, 1, 1)),             # Replace with your desired model
    evaluators=[MeanAbsoluteError()]          # Add evaluation metrics if needed
)

# Fit the pipeline
pipeline.fit(train_series)

# Forecast future values
forecast_steps = len(test_series)
forecasted_values = pipeline.predict(train_series, steps=forecast_steps)

# Evaluate the model (if evaluators are added)
results = pipeline.evaluate(forecasted_values, test_series)

# Print pipeline summary
pipeline.summary()
```

## ARIMA Models

ARIMA (**Auto-Regressive Integrated Moving Average**) models are widely used for time series forecasting. They combine three components: Auto-Regressive (AR), differencing (I), and Moving Average (MA). These models are suitable for univariate time series data and can handle trends and seasonality with extensions like SARIMA and SARIMAX.

- **Auto-Regression (AR)**: This component uses the relationship between an observation and a number of lagged observations (previous time points) to predict future values. The AR part of the model captures the influence of past values on the current value.
- **Differencing (I)**: This component involves subtracting the previous observation from the current observation to make the time series stationary. By stationary we mean that the statistical properties of the series do not change over time. Differencing helps to remove trends and seasonality from the data, making it easier to model.
- **Moving Average (MA)**: This component uses the relationship between an observation and a residual error from a moving average model applied to lagged observations. The MA part of the model captures the influence of past forecast errors on the current value.


### ARIMA
The ARIMA model is defined by three parameters: (p, d, q), where:
- **p**: The number of lag observations included in the model (AR component).
- **d**: The number of times the data is differenced to make it stationary (I component).
- **q**: The size of the moving average window (MA component).

ARIMA is ideal for time series data that is stationary or can be made stationary through differencing.

### SARIMA
SARIMA (**Seasonal ARIMA**) extends ARIMA by incorporating seasonal components. It is particularly useful for time series data with seasonal patterns, such as monthly sales data or daily temperature readings. SARIMA models account for both non-seasonal and seasonal factors, making them more flexible and powerful for seasonal time series forecasting. It works by first fitting an ARIMA on the seasonally differenced data, and then forcasts the seasonally differenced data and inverts the seasonal differencing to obtain the final forecast.

The SARIMA model is defined by two sets of parameters:
- Non-seasonal: (p, d, q)
- Seasonal: (P, D, Q, m), where:
  - **P**: Seasonal AR order.
  - **D**: Seasonal differencing order.
  - **Q**: Seasonal MA order.
  - **m**: The number of time steps in a seasonal period.


### SARIMAX
SARIMAX (**Seasonal ARIMA with eXogenous variables**) further extends SARIMA by allowing the inclusion of exogenous variables. These variables can provide additional explanatory power for the time series, making SARIMAX useful for scenarios where external factors influence the data. Thus, SARIMAX is a powerful tool for modeling complex time series data with both seasonal and non-seasonal components, as well as external influences. SARIMAX first fits an OLS regression model to the exogenous variables, and then fits an ARIMA model to the residuals of the regression. The final forecast is obtained by combining the forecasts from both models.

The SARIMAX model is defined by three sets of parameters:
- Non-seasonal: (p, d, q)
- Seasonal: (P, D, Q, m)
- Exogenous: (k), where:
  - **k**: The number of exogenous variables.


## Decomposition Models

Decomposition models are used to break down a time series into its underlying components: trend, seasonality, and residual (noise). These models help in understanding the structure of the data and are often used as a preprocessing step for forecasting. Additive and multiplicative decomposition differ in how they treat the relationship between the components. Use additive decomposition when the seasonal variations are roughly constant over time, and multiplicative decomposition when the seasonal variations change proportionally with the trend.

### Additive Decomposition

In additive decomposition, the time series is assumed to be the sum of its components: $Y_t = T_t + S_t + R_t$
- **Trend (T)**: Represents the long-term progression of the series.
- **Seasonality (S)**: Captures repeating patterns or cycles in the data.
- **Residual (R)**: Represents the random noise or irregular fluctuations.

This method is suitable for time series where the seasonal variations are roughly constant over time, regardless of the trend.

### Multiplicative Decomposition

In multiplicative decomposition, the time series is assumed to be the product of its components:$Y_t = T_t \times S_t \times R_t$
- **Trend (T)**: Represents the long-term progression of the series.
- **Seasonality (S)**: Captures repeating patterns or cycles in the data, proportional to the trend.
- **Residual (R)**: Represents the random noise or irregular fluctuations.

This method is suitable for time series where the seasonal variations change proportionally with the trend (e.g., higher values during peaks and lower values during troughs).


## Exponential Smoothing Models

Exponential Smoothing models are a family of forecasting methods that use weighted averages of past observations, where the weights decrease exponentially over time. These models are suitable for time series data with different characteristics, such as no trend, a linear trend, or both trend and seasonality.

### Simple Exponential Smoothing

Simple Exponential Smoothing (SES) is used for time series data without a trend or seasonality. It forecasts future values as a weighted average of past observations, with more recent observations receiving higher weights. The model is defined by a single smoothing parameter, **alpha (α)**, which controls the rate of exponential decay.

- **Use Case**: Suitable for stationary time series without trend or seasonality.
- **Forecast Formula**:
  $l_t = \alpha y_t + (1 - \alpha) l_{t-1}$
  where $l_t$ is the level at time $t$, and $y_t$ is the observed value.

### Double Exponential Smoothing (Holt's Linear Trend Model)

Double Exponential Smoothing (DES), also known as Holt's Linear Trend Model, extends SES to handle time series with a linear trend. It introduces a second smoothing parameter, **beta (β)**, to model the trend component.

- **Use Case**: Suitable for time series with a linear trend but no seasonality.
- **Forecast Formula**:
  $l_t = \alpha y_t + (1 - \alpha) (l_{t-1} + b_{t-1})$
  $b_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1}$
  $y_{t+h} = l_t + h b_t$
  where $l_t$ is the level, $b_t$ is the trend, and $h$ is the forecast horizon.

### Triple Exponential Smoothing (Holt-Winters Seasonal Model)

Triple Exponential Smoothing (TES), also known as the Holt-Winters Seasonal Model, extends DES to handle time series with both trend and seasonality. It introduces a third smoothing parameter, **gamma (γ)**, to model the seasonal component. TES can handle both additive and multiplicative seasonality.

- **Use Case**: Suitable for time series with both trend and seasonality.
- **Forecast Formula (Additive Seasonality)**:
  $l_t = \alpha (y_t - s_{t-m}) + (1 - \alpha) (l_{t-1} + b_{t-1})$
  $b_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1}$
  $s_t = \gamma (y_t - l_t) + (1 - \gamma) s_{t-m}$
  $y_{t+h} = l_t + h b_t + s_{t+h-m}$
  where $l_t$ is the level, $b_t$ is the trend, $s_t$ is the seasonal component, $m$ is the seasonal period, and $h$ is the forecast horizon.

## Moving Average Models

Moving Average models are used to smooth time series data by averaging observations over a specified window. These models help reduce noise and reveal underlying trends or patterns in the data.

### Simple Moving Average (SMA)

The Simple Moving Average (SMA) calculates the unweighted mean of the last $n$ observations in a time series. It is a straightforward method for smoothing data and is often used as a baseline for comparison with more complex models.

- **Use Case**: Suitable for time series data without significant trends or seasonality.
- **Formula**:
  $SMA_t = \frac{1}{n} \sum_{i=0}^{n-1} y_{t-i}$
  where $n$ is the window size, and $y_{t-i}$ are the observations.

### Weighted Moving Average (WMA)

The Weighted Moving Average (WMA) assigns different weights to observations within the window, typically giving more importance to recent observations. This makes WMA more responsive to recent changes in the data compared to SMA.

- **Use Case**: Suitable for time series data where recent observations are more relevant for forecasting.
- **Formula**:
  $WMA_t = \frac{\sum_{i=0}^{n-1} w_i y_{t-i}}{\sum_{i=0}^{n-1} w_i}$
  where $w_i$ are the weights, $n$ is the window size, and $y_{t-i}$ are the observations. The weights are typically normalized to sum to 1.

### Exponential Moving Average (EMA)

The Exponential Moving Average (EMA) is a type of weighted moving average that applies exponentially decreasing weights to past observations. It is more responsive to recent changes in the data compared to SMA and WMA.
- **Use Case**: Suitable for time series data where recent observations are more relevant for forecasting, and it is often used in financial applications.
- **Formula**:
  $EMA_t = \alpha y_t + (1 - \alpha) EMA_{t-1}$
  where $\alpha$ is the smoothing factor (0 < α < 1), and $y_t$ is the observed value. The smoothing factor determines the weight given to the most recent observation compared to the previous EMA.

## Example Usages

### ARIMA
```python
from sega_learn.time_series.arima import ARIMA, SARIMA, SARIMAX
from sega_learn.utils import make_time_series

# Generate time series
time_series = make_time_series(n_samples=1,n_timestamps=300,n_features=1)

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# OPTIONAL: Find the best ARIMA order using the custom ARIMA class
order = ARIMA.suggest_order(train_series)
print(f"Suggested ARIMA order: {order}")

order = ARIMA.find_best_order(train_series, test_series)
print(f"Best ARIMA order: {order}")

# Initialize and fit the custom ARIMA model
# order = (1, 2, 1)
arima_model = ARIMA(order=order)
arima_model.fit(train_series)

# Forecast future values using the custom ARIMA model
forecast_steps = len(test_series)
forecasted_values_custom = arima_model.forecast(steps=forecast_steps)
print(f"Forecasted values (custom ARIMA): {forecasted_values_custom}")
```

### Decomposition
```python
from sega_learn.time_series.decomposition import AdditiveDecomposition, MultiplicativeDecomposition
from sega_learn.utils import make_time_series

# Generate time series
time_series = make_time_series(n_samples=1,n_timestamps=300,n_features=1)

# Perform Decomposition
additive_model = AdditiveDecomposition(period=period)

# Fit the model to the time series
additive_model.fit(time_series)

# Get the decomposed components
components = additive_model.get_components()
trend = components["trend"]
seasonal = components["seasonal"]
residual = components["residual"]

# Plot or visualize the components
...
```

### Exponential Smoothing
```python
from sega_learn.time_series.exponential_smoothing import SimpleExponentialSmoothing, DoubleExponentialSmoothing, TripleExponentialSmoothing
from sega_learn.utils import make_time_series

# Generate time series
time_series = make_time_series(n_samples=1,n_timestamps=300,n_features=1)

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# Initialize and fit the Smoothing model
alpha = 0.2  # Smoothing parameter (0 < alpha < 1)
ses_model = SimpleExponentialSmoothing(alpha=alpha)
fitted_values = ses_model.fit(train_series)

# Forecast future values using the Smoothing model
forecast_steps = len(test_series)
forecasted_values_ses = ses_model.forecast(steps=forecast_steps)

# Plot or visualize the forecasted values
...
```

<!-- ### Forecasting Models
```python
``` -->

### Moving Average
```python
from sega_learn.time_series.moving_average import SimpleMovingAverage, WeightedMovingAverage, ExponentialMovingAverage
from sega_learn.utils import make_time_series

# Generate time series
time_series = make_time_series(n_samples=1,n_timestamps=300,n_features=1)

# Split into training and testing sets
train_size = int(len(time_series) * 0.8)
train_series, test_series = time_series[:train_size], time_series[train_size:]

# Initialize and fit the Moving Average model
window_size = 10  # Window size for moving average
sma_model = SimpleMovingAverage(window_size=window_size)

# Fit the model to the training data
sma_model.fit(train_series)

# Forecast future values using the Moving Average model
forecast_steps = len(test_series)
forecasted_values_sma = sma_model.forecast(steps=forecast_steps)

# Plot or visualize the forecasted values
...
```
