import warnings

import numpy as np


class ARIMA:
    """ARIMA model for time series forecasting.

    ARIMA is a class of models that explains a given time series based on its own past values,
    its own past forecast errors, and a number of lagged forecast errors.
    It is a combination of Auto-Regressive (AR), Moving Average (MA) models, and differencing (I) to make the series stationary.

    The model is defined by three parameters: p, d, and q, which represent the order of the AR,
    the degree of differencing, and the order of the MA components, respectively.

    Attributes:
        order (tuple): The order of the ARIMA model (p, d, q).
        p (int): The order of the Auto-Regressive (AR) component.
        d (int): The degree of differencing.
        q (int): The order of the Moving Average (MA) component.
        model (array-like): The original time series data.
        fitted_model (dict): The fitted ARIMA model containing AR and MA components.
    """

    def __init__(self, order):
        """Initialize the ARIMA model.

        ARIMA(p, d, q) model where:
            - p: Order of the Auto-Regressive (AR) component.
            - d: Degree of differencing (number of times the series is differenced).
            - q: Order of the Moving Average (MA) component.

        Args:
            order (tuple): The order of the ARIMA model (p, d, q).

        Selecting the right values:
            - p: Use the Partial Autocorrelation Function (PACF) plot to determine the lag where the PACF cuts off.
            - d: Use the Augmented Dickey-Fuller (ADF) test to check stationarity. Increase `d` until the series becomes stationary.
            - q: Use the Autocorrelation Function (ACF) plot to determine the lag where the ACF cuts off.
        """
        # Validate input order
        if not isinstance(order, list | tuple) or len(order) != 3:
            raise ValueError("Order must be a list or tuple of length 3 (p, d, q).")
        if not all(isinstance(i, int) and i >= 0 for i in order):
            raise ValueError("p, d, and q must be non-negative integers.")

        self.order = order
        self.p = order[0]
        self.d = order[1]
        self.q = order[2]
        self.model = None
        self.fitted_model = None

    def fit(self, time_series):
        """Fit the ARIMA model to the given time series data.

        Args:
            time_series (array-like): The time series data to fit the model to.
        """
        # Step 1: Perform differencing to make the series stationary
        differenced_series = self._difference_series(time_series, self.d)

        # Step 2: Fit the AR (Auto-Regressive) component
        ar_coefficients = self._fit_ar_model(differenced_series, self.p)

        # Step 3: Compute residuals from the AR model
        residuals = self._compute_residuals(differenced_series, ar_coefficients)

        # Step 4: Fit the MA (Moving Average) component
        ma_coefficients = self._fit_ma_model(residuals, self.q)

        # Step 5: Combine AR and MA components into a single model
        self.fitted_model = self._combine_ar_ma(ar_coefficients, ma_coefficients)

        # Store the original time series for inverse differencing during forecasting
        self.model = time_series

    def forecast(self, steps):
        """Forecast future values using the fitted ARIMA model.

        Args:
            steps (int): The number of steps to forecast.

        Returns:
            array-like: The forecasted values.
        """
        if self.fitted_model is None:
            raise ValueError("The model must be fitted before forecasting.")

        # Step 1: Forecast future values using the fitted ARIMA model
        forecasted_values = self._forecast_arima(self.fitted_model, steps)

        # Step 2: Apply inverse differencing to reconstruct the original scale
        forecasted_values = self._inverse_difference(
            self.model, forecasted_values, self.d
        )

        return forecasted_values

    def _compute_residuals(self, differenced_series, ar_coefficients):
        """Compute residuals from the AR model."""
        return differenced_series[self.p :] - np.dot(
            np.array(
                [
                    differenced_series[i : len(differenced_series) - self.p + i]
                    for i in range(self.p)
                ]
            ).T,
            ar_coefficients,
        )

    def _compute_ar_part(self, ar_coefficients, forecasted_values, p):
        """Compute the AR contribution to the forecast."""
        return sum(ar_coefficients[i] * forecasted_values[-i - 1] for i in range(p))

    def _compute_ma_part(self, ma_coefficients, residuals, q):
        """Compute the MA contribution to the forecast."""
        return sum(ma_coefficients[i] * residuals[-i - 1] for i in range(q))

    def _difference_series(self, time_series, d):
        """Perform differencing on the time series to make it stationary.

        Args:
            time_series (array-like): The original time series data.
            d (int): The degree of differencing.

        Returns:
            array-like: The differenced time series.
        """
        if len(time_series) <= d:
            raise ValueError("Differencing degree exceeds time series length.")

        # For each degree of differencing, compute the difference
        # between consecutive observations
        for _ in range(d):
            time_series = np.diff(time_series)
        return time_series

    def _fit_ar_model(self, time_series, p):
        """Fit the Auto-Regressive (AR) component of the model.

        Args:
            time_series (array-like): The stationary time series data.
            p (int): The order of the AR component.

        Returns:
            array-like: The AR coefficients.
        """
        # If p is 0, return an empty array (no AR component)
        if p == 0:
            return np.array([])

        # Construct the design matrix for AR(p)
        # X is a matrix where each row contains p lagged values of the time series
        # X[i] = [time_series[i], time_series[i-1], ..., time_series[i-p+1]]
        X = np.array([time_series[i : len(time_series) - p + i] for i in range(p)]).T

        # y is the current value of the time series
        # y[i] = time_series[i+p]
        y = time_series[p:]

        # Ensure X and y have matching lengths
        if len(X) != len(y):
            X = X[: len(y)]

        # Compute and return the AR coefficients using least squares
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def _fit_ma_model(self, residuals, q):
        """Fit the Moving Average (MA) component of the model.

        Args:
            residuals (array-like): The residuals from the AR model.
            q (int): The order of the MA component.

        Returns:
            array-like: The MA coefficients.
        """
        # If q is 0, return an empty array (no MA component)
        if q == 0:
            return np.array([])

        # Construct the design matrix for MA(q)
        # X is a matrix where each row contains q lagged residuals
        # X[i] = [residuals[i], residuals[i-1], ..., residuals[i-q+1]]
        X = np.array([residuals[i : len(residuals) - q + i] for i in range(q)]).T

        # y is the current value of the residuals
        # y[i] = residuals[i+q]
        # Note: We need to shift the residuals by q to align with the design matrix
        y = residuals[q:]

        # Ensure X and y have matching lengths
        if len(X) != len(y):
            X = X[: len(y)]

        # Compute and return the MA coefficients using least squares
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def _combine_ar_ma(self, ar_coefficients, ma_coefficients):
        """Combine AR and MA components into a single model.

        Args:
            ar_coefficients (array-like): The AR coefficients.
            ma_coefficients (array-like): The MA coefficients.

        Returns:
            dict: The combined ARIMA model.
        """
        # Store the AR and MA coefficients in a dictionary to represent the model
        return {
            "ar_coefficients": ar_coefficients,
            "ma_coefficients": ma_coefficients,
            "order": self.order,
        }

    def _forecast_arima(self, fitted_model, steps):
        """Forecast future values using the fitted ARIMA model.

        Args:
            fitted_model (dict): The fitted ARIMA model containing AR and MA components.
            steps (int): The number of steps to forecast.

        Returns:
            array-like: The forecasted values.
        """
        # Get AR and MA coefficients from the fitted model
        ar_coefficients = fitted_model["ar_coefficients"]
        ma_coefficients = fitted_model["ma_coefficients"]
        # Get p and q from length of coefficients (can also be obtained from order)
        p = len(ar_coefficients)
        q = len(ma_coefficients)

        # Initialize forecasted values and residuals
        forecasted_values = list(self.model[-p:])  # Start with the last `p` values
        residuals = list(
            self.model[-self.q :]
        )  # Use the last `q` residuals from the fitted model

        for _ in range(steps):
            # Compute AR and MA contributions
            ar_part = self._compute_ar_part(ar_coefficients, forecasted_values, p)
            ma_part = self._compute_ma_part(ma_coefficients, residuals, q)

            # Forecast next value
            next_value = ar_part + ma_part
            forecasted_values.append(next_value)

            # Update residuals (assume zero error for forecasted steps)
            residuals.append(next_value - ar_part)

        # Return the last `steps` forecasted values
        return np.array(forecasted_values[-steps:])

    def _inverse_difference(self, original_series, differenced_series, d):
        """Reconstruct the original series from the differenced series.

        Args:
            original_series (array-like): The original time series data.
            differenced_series (array-like): The differenced time series.
            d (int): The degree of differencing.

        Returns:
            array-like: The reconstructed time series.
        """
        if len(original_series) < d:
            raise ValueError(
                "Original series length is insufficient for inverse differencing."
            )

        # For each degree of differencing, compute the cumulative sum
        # to reconstruct the original series
        for _ in range(d):
            # Add back the last value of the original series to reconstruct the scale
            differenced_series = np.r_[original_series[-1], differenced_series].cumsum()
            original_series = original_series[:-1]
        # Return the reconstructed series
        return differenced_series[d:]

    @staticmethod
    def suggest_order(time_series, max_p=5, max_d=2, max_q=5):
        """Suggest the optimal ARIMA order (p, d, q) for the given time series.

        Args:
            time_series (array-like): The time series data.
            max_p (int): Maximum order for AR component.
            max_d (int): Maximum degree of differencing.
            max_q (int): Maximum order for MA component.

        Returns:
            tuple: The optimal order (p, d, q).
        """
        try:
            from statsmodels.tsa.stattools import acf, adfuller, pacf
        except ImportError as e:
            raise ImportError(
                "Please install the required dependencies for this function: statsmodels."
            ) from e

        # Step 1: Determine d (degree of differencing) using the ADF test
        d = 0
        while True:
            try:
                adf_test = adfuller(time_series)
                if adf_test[1] <= 0.05:  # p-value <= 0.05 indicates stationarity
                    break
                time_series = np.diff(time_series, prepend=time_series[0])
                d += 1
            except Exception as e:
                raise ValueError(f"Error during ADF test: {e}") from e

        # Step 2: Determine p (AR order) using the PACF plot
        try:
            pacf_values = pacf(time_series, nlags=20)
            p = min(
                next(
                    (
                        i
                        for i, val in enumerate(pacf_values)
                        if abs(val) < 1.96 / np.sqrt(len(time_series))
                    ),
                    max_p,
                ),
                max_p,
            )
        except Exception as e:
            p = 0  # Default to 0 if PACF fails
            warnings.warn(
                f"Error determining p using PACF: {e}. Defaulting p to 0.",
                UserWarning,
                stacklevel=2,
            )

        # Step 3: Determine q (MA order) using the ACF plot
        try:
            acf_values = acf(time_series, nlags=20)
            q = min(
                next(
                    (
                        i
                        for i, val in enumerate(acf_values)
                        if abs(val) < 1.96 / np.sqrt(len(time_series))
                    ),
                    max_q,
                ),
                max_q,
            )
        except Exception as e:
            q = 0  # Default to 0 if ACF fails
            warnings.warn(
                f"Error determining q using ACF: {e}. Defaulting q to 0.",
                UserWarning,
                stacklevel=2,
            )

        return (p, min(d, max_d), q)

    @staticmethod
    def find_best_order(
        train_series, test_series, max_p=5, max_d=2, max_q=5, subset_size=1.0
    ):
        """Find the best ARIMA order using grid search.

        Args:
            train_series (array-like): The training time series data.
            test_series (array-like): The testing time series data.
            max_p (int): Maximum order for AR component.
            max_d (int): Maximum degree of differencing.
            max_q (int): Maximum order for MA component.
            subset_size (float): Proportion of the training set to use for fitting.

        Returns:
            tuple: The best order (p, d, q).
        """
        # Validate input data
        if not isinstance(train_series, list | np.ndarray) or not isinstance(
            test_series, list | np.ndarray
        ):
            raise ValueError(
                "train_series and test_series must be list or numpy array."
            )
        if len(train_series) < 1 or len(test_series) < 1:
            raise ValueError("train_series and test_series must not be empty.")
        if not (0 < subset_size <= 1.0):
            raise ValueError("subset_size must be between 0 and 1.")

        best_order = None
        best_mse = float("inf")

        if subset_size < 1.0:
            # Randomly sample a subset of the training series
            subset_size = int(len(train_series) * subset_size)
            train_series = np.random.choice(
                train_series, size=subset_size, replace=False
            )

        # Loop through all combinations of (p, d, q) within the specified limits
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    # For each combination, create an ARIMA model and fit it
                    try:
                        arima_model = ARIMA(order=(p, d, q))
                        arima_model.fit(train_series)
                        forecasted_values = arima_model.forecast(steps=len(test_series))
                        mse = np.mean((test_series - forecasted_values) ** 2)

                        # If the MSE is lower than the best found so far, update best order
                        if mse < best_mse:
                            best_mse = mse
                            best_order = (p, d, q)

                    # Handle any exceptions that may arise during fitting or forecasting
                    except Exception as _e:
                        continue
        return best_order


class SARIMA:
    """SARIMA model for time series forecasting."""

    # Implementation of SARIMA model
    pass


class SARIMAX:
    """SARIMAX model for time series forecasting."""

    # Implementation of SARIMAX model
    pass
