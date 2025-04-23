import warnings

import numpy as np


# Keep the existing ARIMA, SARIMA, SARIMAX classes here...
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
        model (array-like): The original time series data used for fitting.
        fitted_model (dict): The fitted ARIMA model containing AR and MA coefficients.
        _differenced_series (array-like): The differenced series used for fitting ARMA.
        _residuals (array-like): The residuals after fitting the AR component.
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
        self._differenced_series = None  # Store differenced series for forecasting
        self._residuals = None  # Store residuals for forecasting

    def fit(self, time_series):
        """Fit the ARIMA model to the given time series data.

        Args:
            time_series (array-like): The time series data to fit the model to.
        """
        self.model = np.asarray(time_series, dtype=float)
        if len(self.model) <= self.d:
            raise ValueError(
                "Time series length must be greater than the differencing order d."
            )

        # Step 1: Perform differencing to make the series stationary
        self._differenced_series = self._difference_series(self.model, self.d)

        if len(self._differenced_series) <= max(self.p, self.q):
            raise ValueError(
                "Differenced series is too short for the specified p and q orders."
            )

        # Step 2: Fit the AR (Auto-Regressive) component
        ar_coefficients = self._fit_ar_model(self._differenced_series, self.p)

        # Step 3: Compute residuals from the AR model
        # Note: Residuals are computed based on the part of the differenced series where AR applies
        if self.p > 0:
            ar_preds = self._predict_ar(self._differenced_series, ar_coefficients)
            self._residuals = self._differenced_series[self.p :] - ar_preds
        else:
            self._residuals = (
                self._differenced_series
            )  # No AR component, residuals are the series itself

        # Step 4: Fit the MA (Moving Average) component using the residuals
        # We fit MA on the residuals of the AR part.
        if len(self._residuals) <= self.q:
            raise ValueError("Residual series is too short for the specified q order.")
        ma_coefficients = self._fit_ma_model(self._residuals, self.q, ar_coefficients)

        # Step 5: Combine AR and MA components into a single model
        self.fitted_model = self._combine_ar_ma(ar_coefficients, ma_coefficients)

    def forecast(self, steps):
        """Forecast future values using the fitted ARIMA model.

        Args:
            steps (int): The number of steps to forecast.

        Returns:
            array-like: The forecasted values.
        """
        if self.fitted_model is None:
            raise ValueError("The model must be fitted before forecasting.")
        if steps <= 0:
            return np.array([])

        # Get components from the fitted model
        ar_coefficients = self.fitted_model["ar_coefficients"]
        ma_coefficients = self.fitted_model["ma_coefficients"]

        # Get the history needed for forecasting
        diff_history = list(self._differenced_series[-(self.p) :]) if self.p > 0 else []
        resid_history = list(self._residuals[-(self.q) :]) if self.q > 0 else []

        forecasted_diff_values = []

        for _ in range(steps):
            # AR part: Sum of lag_value * coeff
            ar_term = sum(
                ar_coefficients[i] * diff_history[-1 - i] for i in range(self.p)
            )

            # MA part: Sum of lag_residual * coeff
            # Use 0 for future residuals as they are unknown (mean of white noise)
            ma_term = sum(
                ma_coefficients[i] * resid_history[-1 - i]
                for i in range(min(self.q, len(resid_history)))
            )

            # Combine AR and MA terms for the forecast of the differenced series
            next_diff_forecast = ar_term + ma_term

            # Append the forecast to the list
            forecasted_diff_values.append(next_diff_forecast)

            # Update history: append the forecast and a zero residual
            if self.p > 0:
                diff_history.append(next_diff_forecast)
                if len(diff_history) > self.p:
                    diff_history.pop(0)
            if self.q > 0:
                # The residual for a forecast step is assumed to be 0 (the expected value of the error term)
                resid_history.append(
                    0
                )  # MA term uses past *known* residuals or 0 for future ones
                if len(resid_history) > self.q:
                    resid_history.pop(0)

        # Inverse differencing to get the forecast in the original scale
        forecasted_values = self._inverse_difference(
            self.model, np.array(forecasted_diff_values), self.d
        )

        return forecasted_values

    def _predict_ar(self, series, ar_coefficients):
        """Predict values using the AR coefficients."""
        p = len(ar_coefficients)
        if p == 0:
            return np.zeros(len(series))  # Predict zero if no AR component

        predictions = []
        for t in range(p, len(series)):
            lagged_vals = series[t - p : t]
            pred = np.dot(
                lagged_vals[::-1], ar_coefficients
            )  # AR(p) uses lags t-1 to t-p
            predictions.append(pred)
        return np.array(predictions)

    # Keep other private methods like _compute_residuals, _difference_series, etc.
    # Slight adjustment to _fit_ar_model and _fit_ma_model for robustness

    def _difference_series(self, time_series, d):
        """Perform differencing on the time series to make it stationary."""
        if d == 0:
            return time_series
        if len(time_series) <= d:
            raise ValueError("Differencing degree exceeds time series length.")

        differenced = np.asarray(time_series).astype(float)
        for _ in range(d):
            differenced = np.diff(differenced)
        return differenced

    def _fit_ar_model(self, time_series, p):
        """Fit the Auto-Regressive (AR) component using OLS."""
        if p == 0:
            return np.array([])
        if len(time_series) <= p:
            raise ValueError(
                f"Time series length ({len(time_series)}) must be greater than AR order p ({p})."
            )

        # Create lagged variables matrix (X) and target variable (y)
        y = time_series[p:]
        X = np.array([time_series[i : len(time_series) - p + i] for i in range(p)]).T
        # Reverse columns so X[t] = [y_{t-1}, y_{t-2}, ..., y_{t-p}]
        X = X[:, ::-1]

        # Fit using Ordinary Least Squares (OLS)
        # Add intercept? Standard ARIMA often assumes zero mean for differenced series, so no intercept.
        try:
            ar_coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular matrix encountered in AR fitting. Coefficients might be unstable.",
                UserWarning,
                stacklevel=2,
            )
            # Fallback or return zeros? Let's return zeros for now
            return np.zeros(p)

        return ar_coefficients

    def _fit_ma_model(self, residuals, q, ar_coefficients):
        """Fit the Moving Average (MA) component.

        Note: Fitting MA model properly often requires non-linear optimization
        (like maximizing likelihood). Using OLS on residuals predicting the original series
        (or differenced series) is an approximation (Hannan-Rissanen algorithm idea).
        A simpler OLS approach (approximating MA as AR on residuals) is sometimes used,
        but it's less accurate. Let's implement a basic OLS approximation where
        y_t (or diff_series_t) is regressed on past residuals e_{t-1}, ..., e_{t-q}.
        This is a simplification. For accurate MA, use statsmodels.
        """
        if q == 0:
            return np.array([])
        if len(residuals) <= q:
            raise ValueError(
                f"Residuals length ({len(residuals)}) must be greater than MA order q ({q})."
            )

        # We need the original (differenced) series corresponding to the residuals
        # Residuals start from index p of the differenced series.
        # y corresponds to diff_series[p+q:]
        # X corresponds to residuals lags starting from q-1 up to 0, aligned with y
        y = self._differenced_series[self.p + q :]

        # Create lagged residuals matrix (X)
        X = np.array([residuals[i : len(residuals) - q + i] for i in range(q)]).T
        # Reverse columns: X[t] = [e_{t-1}, e_{t-2}, ..., e_{t-q}]
        X = X[:, ::-1]

        # Ensure X and y have the same number of rows
        if len(X) != len(y):
            # This can happen if p > 0. Align X with y.
            # Residuals 'start' at time p. So residual[0] corresponds to time p.
            # residual[k] corresponds to time p+k.
            # We need residuals from time p+q-1 down to p.
            # Target y starts at time p+q.
            X = np.array([residuals[i : i + len(y)] for i in range(q)]).T
            X = X[:, ::-1]

        if len(X) == 0 or len(y) == 0:
            warnings.warn(
                "Not enough data points for MA fitting after differencing and AR.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(q)

        # Fit using OLS (approximation)
        try:
            # Regress the *differenced series* (minus AR part, which is handled by residuals) on lagged residuals
            target = self._differenced_series[self.p + q :]  # Part not explained by AR
            if self.p > 0:
                ar_preds_aligned = self._predict_ar(
                    self._differenced_series, ar_coefficients
                )[q:]  # Predict AR part for the target period
                target = target - ar_preds_aligned

            ma_coefficients, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular matrix encountered in MA fitting. Coefficients might be unstable.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(q)
        except ValueError as e:
            warnings.warn(
                f"ValueError during MA fitting: {e}. Returning zero coefficients.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(q)

        return ma_coefficients

    def _combine_ar_ma(self, ar_coefficients, ma_coefficients):
        """Combine AR and MA components into a single model dict."""
        return {
            "ar_coefficients": ar_coefficients,
            "ma_coefficients": ma_coefficients,
            "order": self.order,
        }

    def _inverse_difference(self, original_series, forecast_diff, d):
        """Reconstruct the original scale from the differenced forecast."""
        if d == 0:
            return forecast_diff

        # We need the last 'd' values of each differencing level
        history = [original_series]
        for _i in range(d):
            history.append(np.diff(history[-1]))

        # Start inversion from the forecast (highest difference level)
        inverted = list(forecast_diff)
        for i in range(d, 0, -1):
            # Get the last value of the series at level d-1
            last_val_at_prev_level = history[i - 1][-1]
            # Invert: value[t] = diff[t] + value[t-1]
            # Start cumsum with the last known value
            inverted_level = [last_val_at_prev_level + x for x in np.cumsum(inverted)]
            inverted = inverted_level  # Prepare for next level

        return np.array(inverted)

    # Static methods suggest_order and find_best_order remain the same

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
        train_series,
        test_series,
        max_p=5,
        max_d=2,
        max_q=5,
        subset_size=1.0,
        verbose=False,
    ):
        """Find the best ARIMA order using grid search based on test set MSE.

        Args:
            train_series (array-like): The training time series data.
            test_series (array-like): The testing time series data.
            max_p (int): Maximum order for AR component.
            max_d (int): Maximum degree of differencing.
            max_q (int): Maximum order for MA component.
            subset_size (float): Proportion of the training set to use for fitting.
            verbose (bool): If True, print detailed output.

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

        if subset_size < 1.0:
            subset_idx = int(len(train_series) * subset_size)
            train_series = train_series[-subset_idx:]  # Use the most recent subset

        best_order = None
        best_mse = float("inf")
        n_test = len(test_series)

        if verbose:
            print("Finding best ARIMA order using grid search (Test Set MSE):")
        # Loop through all combinations of (p, d, q) within the specified limits
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    current_order = (p, d, q)
                    if verbose:
                        print(f"Trying order {current_order}...")
                    # For each combination, create an ARIMA model and fit it
                    try:
                        # Use the implemented ARIMA class
                        arima_model = ARIMA(order=current_order)
                        arima_model.fit(train_series)
                        forecasted_values = arima_model.forecast(steps=n_test)

                        if len(forecasted_values) != n_test:
                            if verbose:
                                print(
                                    f"  Warning: Forecast length mismatch ({len(forecasted_values)} vs {n_test}). Skipping."
                                )
                            continue

                        mse = np.mean((test_series - forecasted_values) ** 2)
                        if verbose:
                            print(f"  MSE: {mse}")

                        # If the MSE is lower than the best found so far, update best order
                        if mse < best_mse:
                            best_mse = mse
                            best_order = current_order
                            if verbose:
                                print(
                                    f"  New best order found: {best_order} with MSE {best_mse}"
                                )

                    # Handle any exceptions that may arise during fitting or forecasting
                    except Exception as e:
                        if verbose:
                            print(f"  Failed for order {current_order}: {e}")
                        continue

        if verbose:
            print(f"\nBest order found: {best_order}")
        return best_order


# Keep SARIMA and SARIMAX classes as they were provided.
class SARIMA(ARIMA):
    """SARIMA model for time series forecasting.

    SARIMA extends ARIMA by including seasonal components.

    Attributes:
        order (tuple): The non-seasonal order of the ARIMA model (p, d, q).
        seasonal_order (tuple): The seasonal order of the SARIMA model (P, D, Q, m).
        p (int): The order of the Auto-Regressive (AR) component.
        d (int): The degree of differencing.
        q (int): The order of the Moving Average (MA) component.
        P (int): The order of the seasonal Auto-Regressive (SAR) component.
        D (int): The degree of seasonal differencing.
        Q (int): The order of the seasonal Moving Average (SMA) component.
        m (int): The number of time steps in a seasonal period.
    """

    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1)):
        """Initialize the SARIMA model.

        Args:
            order (tuple): Non-seasonal ARIMA order (p, d, q).
            seasonal_order (tuple): Seasonal order (P, D, Q, m).
        """
        # Validate seasonal_order
        if not isinstance(seasonal_order, list | tuple) or len(seasonal_order) != 4:
            raise ValueError(
                "Seasonal order must be a tuple/list of length 4: (P, D, Q, m)."
            )

        P, D, Q, m = seasonal_order
        if any(x < 0 for x in (P, D, Q)) or m <= 0:
            raise ValueError("P, D, Q must be ≥0 and m must be a positive integer.")
        # Allow m=1 for non-seasonal cases handled by ARIMA part
        if m == 1 and (P > 0 or D > 0 or Q > 0):
            warnings.warn(
                "Seasonal components (P, D, Q) provided with m=1. These will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            seasonal_order = (0, 0, 0, 1)  # Force non-seasonal if m=1

        super().__init__(order)
        # Store the seasonal components
        self.seasonal_order = seasonal_order
        self.P, self.D, self.Q, self.m = seasonal_order
        self.original_series = (
            None  # Store the original series for inverse seasonal diff
        )

    def fit(self, time_series):
        """Fit the SARIMA model to the given time series data.

        Note: This is a simplified implementation. It applies seasonal differencing first,
        then fits a standard ARIMA(p,d,q) model to the result. It does NOT explicitly
        fit the SAR and SMA components. A full SARIMA implementation is much more complex.
        Use `statsmodels.tsa.statespace.SARIMAX` for a proper implementation.

        Args:
            time_series (array-like): The time series data to fit the model to.
        """
        self.original_series = np.asarray(time_series, dtype=float)

        # Apply seasonal differencing if needed
        if self.D > 0 and self.m > 1:
            if len(self.original_series) <= self.D * self.m:
                raise ValueError(
                    "Time series is too short for the specified seasonal differencing (D, m)."
                )
            ts_sd = self._seasonal_difference(self.original_series, self.D, self.m)
        else:
            ts_sd = self.original_series.copy()

        # Fit the ARIMA(p,d,q) on the (potentially) seasonally-differenced series
        # This simplified approach ignores P and Q parameters during fitting.
        if self.P > 0 or self.Q > 0:
            warnings.warn(
                "This simplified SARIMA fit ignores P and Q parameters. It only applies differencing (d, D) and fits ARIMA(p,d,q). Use statsmodels for full SARIMA.",
                UserWarning,
                stacklevel=2,
            )

        super().fit(ts_sd)  # Fit ARIMA(p,d,q)

    def forecast(self, steps):
        """Forecast future values using the fitted SARIMA model.

        Applies the ARIMA forecast on the seasonally differenced scale, then inverts
        the seasonal differencing.

        Args:
            steps (int): The number of steps to forecast.

        Returns:
            array-like: The forecasted values.
        """
        if self.fitted_model is None or self.original_series is None:
            raise ValueError("Fit the model before forecasting.")
        if steps <= 0:
            return np.array([])

        # Get forecasts on the seasonally differenced scale using the parent ARIMA forecast
        fc_sd = super().forecast(steps)

        # Invert seasonal differencing
        if self.D > 0 and self.m > 1:
            forecasted_values = self._inverse_seasonal_difference(fc_sd)
        else:
            # If no seasonal differencing was applied, the forecast is already in the original scale
            # (after potential regular differencing inversion by the parent forecast method)
            # However, the parent forecast method inverts regular differencing based on self.model,
            # which might be the seasonally differenced one. We need to ensure the final inversion uses the *original* series history.
            # Let's re-do the inverse differencing here using the *original* series history if d > 0.
            if self.d > 0:
                # The forecast `fc_sd` is already inverted for regular diff `d` but based on the *seasonally differenced* history.
                # To correctly invert `d` based on the *original* series, we need the raw forecast from the ARMA part.
                # This highlights the complexity and limitations of this simplified approach.
                # A full implementation integrates all components.
                # WORKAROUND: Assume parent `forecast` correctly inverted `d`, now just invert `D`.
                forecasted_values = self._inverse_seasonal_difference(
                    fc_sd
                )  # Apply only seasonal inversion logic
            else:
                forecasted_values = fc_sd  # No differencing (d=0, D=0)

        return forecasted_values

    def _seasonal_difference(self, series, D, m):
        """Apply D rounds of lag-m differencing."""
        arr = series.copy()
        for _ in range(D):
            if len(arr) <= m:
                raise ValueError(
                    "Cannot perform seasonal difference: series length is less than or equal to m."
                )
            arr = arr[m:] - arr[:-m]
        return arr

    def _inverse_seasonal_difference(self, diff_forecast):
        """Reconstruct original scale from seasonally differenced forecasts."""
        if self.D == 0 or self.m <= 1:
            return diff_forecast  # No seasonal differencing to invert

        history = list(self.original_series)  # Use the original full history
        reconstructed = []

        for fc_val in diff_forecast:
            val = fc_val
            # Add back the historical value from m periods ago, D times
            for _d_level in range(self.D, 0, -1):
                # Need the value from history corresponding to the correct lag
                # For the first forecast step (t=N+1), we need history[N+1-m].
                # For the second (t=N+2), we need history[N+2-m], etc.
                # The `history` list grows with each forecast step.
                lag_index = len(history) - self.m  # Index of the value m steps back
                if lag_index < 0:
                    # This case should ideally not happen if initial series was long enough
                    # Fallback: use the earliest available value? Or assume 0? Let's use 0 and warn.
                    warnings.warn(
                        f"Insufficient history for inverse seasonal differencing lag {self.m}. Using 0.",
                        UserWarning,
                        stacklevel=2,
                    )
                    lag_val = 0
                else:
                    # We need the value from the *d_level-1* differenced series history.
                    # This is complex. The simplified approach just uses the original series history.
                    lag_val = history[lag_index]
                val += lag_val

            reconstructed.append(val)
            history.append(
                val
            )  # Add the reconstructed value to history for the next step

        return np.array(reconstructed)

    # Static methods suggest_order and find_best_order need significant overhaul for SARIMA
    # The provided implementations are placeholders and likely won't give good results
    # due to the simplified fit/forecast logic. Use statsmodels versions for reliability.

    @staticmethod
    def suggest_order(
        time_series, max_p=3, max_d=1, max_q=3, max_P=2, max_D=1, max_Q=2, max_m=12
    ):
        """Suggest SARIMA order using statsmodels (reliable method).

        Note: This replaces the previous basic heuristic with a call to statsmodels grid search,
        as implementing a reliable SARIMA order suggestion from scratch is complex.
        """
        warnings.warn(
            "Using statsmodels.tsa.statespace.SARIMAX for reliable order suggestion.",
            UserWarning,
            stacklevel=2,
        )
        try:
            from itertools import product

            import statsmodels.api as sm
        except ImportError as e:
            raise ImportError(
                "Please install statsmodels: pip install statsmodels"
            ) from e

        time_series = np.asarray(time_series)
        best_aic = float("inf")
        best_order = ((0, 0, 0), (0, 0, 0, 1))  # Default

        # Define parameter ranges
        p_range = range(max_p + 1)
        d_range = range(max_d + 1)
        q_range = range(max_q + 1)
        P_range = range(max_P + 1)
        D_range = range(max_D + 1)
        Q_range = range(max_Q + 1)

        # Determine likely seasonal periods (e.g., based on ACF peaks or common values)
        # Simple approach: check common periods if data is long enough
        potential_m = [1]
        if max_m > 1 and len(time_series) > 2 * max_m:  # Check max_m only if reasonable
            potential_m.append(max_m)
        if max_m >= 12 and len(time_series) > 24:
            potential_m.append(12)
        if max_m >= 7 and len(time_series) > 14:
            potential_m.append(7)
        if max_m >= 4 and len(time_series) > 8:
            potential_m.append(4)
        potential_m = sorted(set(potential_m))  # Unique sorted list

        print(f"Searching SARIMA orders for m in {potential_m}...")

        # Grid search over parameters
        non_seasonal_orders = list(product(p_range, d_range, q_range))
        seasonal_orders = list(product(P_range, D_range, Q_range))

        total_combinations = (
            len(non_seasonal_orders) * len(seasonal_orders) * len(potential_m)
        )
        count = 0

        for m in potential_m:
            if m <= 1:  # Handle non-seasonal ARIMA case
                seasonal_params = [(0, 0, 0)]
                m_val = 1
            else:
                seasonal_params = seasonal_orders
                m_val = m

            for order in non_seasonal_orders:
                for seasonal_order_part in seasonal_params:
                    count += 1
                    current_seasonal_order = seasonal_order_part + (m_val,)
                    print(
                        f"[{count}/{total_combinations}] Trying order={order}, seasonal_order={current_seasonal_order}...",
                        end="\r",
                    )

                    # Skip trivial models
                    if order == (0, 0, 0) and current_seasonal_order == (
                        0,
                        0,
                        0,
                        m_val,
                    ):
                        continue
                    # Skip if only differencing exists without AR/MA/SAR/SMA terms
                    if (
                        order[0] == 0
                        and order[2] == 0
                        and seasonal_order_part[0] == 0
                        and seasonal_order_part[2] == 0
                        and (order[1] > 0 or seasonal_order_part[1] > 0)
                    ):
                        continue

                    try:
                        model = sm.tsa.statespace.SARIMAX(
                            time_series,
                            order=order,
                            seasonal_order=current_seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            simple_differencing=False,
                        )  # Use exact MLE
                        results = model.fit(disp=False)
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (order, current_seasonal_order)
                            print(
                                f"\nNew best: order={best_order[0]}, seasonal={best_order[1]}, AIC={best_aic:.2f}"
                            )

                    except Exception:
                        # print(f"\nFailed for order={order}, seasonal={current_seasonal_order}: {e}")
                        continue  # Ignore models that fail to fit

        print(f"\nSuggested Best Order: {best_order} with AIC: {best_aic:.2f}")
        return best_order

    @staticmethod
    def find_best_order(  # noqa: D417
        train_series,
        test_series,
        max_p=2,
        max_d=1,
        max_q=2,
        max_P=1,
        max_D=1,
        max_Q=1,
        max_m=12,
    ):
        """Find the best SARIMA order using grid search based on test set MSE (using statsmodels).

        Note: Uses statsmodels for fitting and forecasting for reliability.
        """
        warnings.warn(
            "Using statsmodels.tsa.statespace.SARIMAX for reliable order search (MSE based).",
            UserWarning,
            stacklevel=2,
        )
        try:
            from itertools import product

            import statsmodels.api as sm
        except ImportError as e:
            raise ImportError(
                "Please install statsmodels: pip install statsmodels"
            ) from e

        train_series = np.asarray(train_series)
        test_series = np.asarray(test_series)
        best_mse = float("inf")
        best_order = ((0, 0, 0), (0, 0, 0, 1))  # Default
        n_test = len(test_series)

        # Define parameter ranges
        p_range = range(max_p + 1)
        d_range = range(max_d + 1)
        q_range = range(max_q + 1)
        P_range = range(max_P + 1)
        D_range = range(max_D + 1)
        Q_range = range(max_Q + 1)

        # Determine likely seasonal periods
        potential_m = [1]
        if max_m > 1 and len(train_series) > 2 * max_m:
            potential_m.append(max_m)
        if max_m >= 12 and len(train_series) > 24:
            potential_m.append(12)
        if max_m >= 7 and len(train_series) > 14:
            potential_m.append(7)
        if max_m >= 4 and len(train_series) > 8:
            potential_m.append(4)
        potential_m = sorted(set(potential_m))

        print(f"Searching SARIMA orders (MSE) for m in {potential_m}...")

        # Grid search
        non_seasonal_orders = list(product(p_range, d_range, q_range))
        seasonal_orders = list(product(P_range, D_range, Q_range))

        total_combinations = (
            len(non_seasonal_orders) * len(seasonal_orders) * len(potential_m)
        )
        count = 0

        for m in potential_m:
            if m <= 1:
                seasonal_params = [(0, 0, 0)]
                m_val = 1
            else:
                seasonal_params = seasonal_orders
                m_val = m

            for order in non_seasonal_orders:
                for seasonal_order_part in seasonal_params:
                    count += 1
                    current_seasonal_order = seasonal_order_part + (m_val,)
                    print(
                        f"[{count}/{total_combinations}] Trying order={order}, seasonal_order={current_seasonal_order}...",
                        end="\r",
                    )

                    if order == (0, 0, 0) and current_seasonal_order == (
                        0,
                        0,
                        0,
                        m_val,
                    ):
                        continue
                    if (
                        order[0] == 0
                        and order[2] == 0
                        and seasonal_order_part[0] == 0
                        and seasonal_order_part[2] == 0
                        and (order[1] > 0 or seasonal_order_part[1] > 0)
                    ):
                        continue

                    try:
                        # Use statsmodels for fitting and forecasting
                        model = sm.tsa.statespace.SARIMAX(
                            train_series,
                            order=order,
                            seasonal_order=current_seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            simple_differencing=False,
                        )
                        results = model.fit(disp=False)
                        forecast = results.get_forecast(steps=n_test).predicted_mean

                        mse = np.mean((test_series - forecast) ** 2)

                        if mse < best_mse:
                            best_mse = mse
                            best_order = (order, current_seasonal_order)
                            print(
                                f"\nNew best: order={best_order[0]}, seasonal={best_order[1]}, MSE={best_mse:.4f}"
                            )

                    except Exception:
                        # print(f"\nFailed for order={order}, seasonal={current_seasonal_order}: {e}")
                        continue

        print(f"\nBest Order (MSE): {best_order} with MSE: {best_mse:.4f}")
        return best_order


class SARIMAX(SARIMA):
    """SARIMAX model with exogenous regressors.

    Inherits SARIMA structure but adds handling for exogenous variables.

    Simplified approach (as with SARIMA):
      1. OLS regression of y on exog to get beta + residuals
      2. Simplified SARIMA fit (seasonal diff + ARIMA(p,d,q)) on the residuals
    Forecast = SARIMA_forecast(resid) + exog_future @ beta (seasonal inversion applied to resid forecast)

    Attributes:
        beta (np.ndarray): The beta coefficients for exogenous regressors.
        k_exog (int): The number of exogenous variables.
        exog_fit (np.ndarray): Exogenous variables used during fitting.
        resid_fit (np.ndarray): Residuals after removing exogenous effect, used for SARIMA fit.
    """

    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1)):
        """Initialize the SARIMAX model.

        Args:
            order (tuple): Non-seasonal ARIMA order (p, d, q).
            seasonal_order (tuple): Seasonal order (P, D, Q, m).
        """
        super().__init__(order=order, seasonal_order=seasonal_order)
        self.beta = None
        self.k_exog = None
        self.exog_fit = None
        self.resid_fit = None  # Store residuals from OLS

    def fit(self, time_series, exog):
        """Fit the SARIMAX model.

        Args:
            time_series (array-like): The time series data (endogenous variable).
            exog (array-like): The exogenous regressors.

        Returns:
            self: The fitted SARIMAX model.
        """
        y = np.asarray(time_series, dtype=float)
        X = np.asarray(exog, dtype=float)

        # Validate dimensions
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(y) != X.shape[0]:
            raise ValueError(
                "endog (time_series) and exog must have the same number of observations."
            )
        self.exog_fit = X
        self.k_exog = X.shape[1]

        # Step 1: OLS Regression y ~ X
        try:
            # Add intercept to exogenous regressors? Usually yes.
            X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
            beta_with_intercept, _, _, _ = np.linalg.lstsq(
                X_with_intercept, y, rcond=None
            )
            self.beta = beta_with_intercept  # Includes intercept as beta[0]
            self.resid_fit = y - X_with_intercept.dot(self.beta)
        except np.linalg.LinAlgError:
            raise RuntimeError(  # noqa: B904
                "Failed to fit initial OLS regression on exogenous variables."
            )

        # Step 2: Fit SARIMA model on the residuals
        # The parent SARIMA fit method will handle seasonal differencing of these residuals
        # and then fit ARIMA(p,d,q) on them.
        # We need to store the original residuals *before* seasonal differencing for inversion later.
        self.original_series = (
            self.resid_fit.copy()
        )  # Store OLS residuals as the series for SARIMA part

        # Call the parent SARIMA fit method on the OLS residuals
        super().fit(self.resid_fit)

        # Important: The SARIMA fit overwrites self.original_series with potentially seasonally differenced residuals.
        # Let's restore the correct original series (OLS residuals) needed for inverse differencing.
        self.original_series = self.resid_fit.copy()

        return self

    def forecast(self, steps, exog_future):
        """Forecast future values using the fitted SARIMAX model.

        Args:
            steps (int): The number of steps to forecast.
            exog_future (array-like): Future values of exogenous regressors (without intercept).

        Returns:
            array-like: The forecasted values.
        """
        if self.beta is None or self.fitted_model is None:
            raise ValueError("Fit the model before forecasting.")
        if steps <= 0:
            return np.array([])

        Xf = np.asarray(exog_future, dtype=float)
        if Xf.ndim == 1:
            Xf = Xf.reshape(-1, 1)
        if Xf.shape[0] != steps:
            raise ValueError(
                f"Length of exog_future ({Xf.shape[0]}) must match steps ({steps})."
            )
        if Xf.shape[1] != self.k_exog:
            raise ValueError(
                f"exog_future must have {self.k_exog} columns (excluding intercept)."
            )

        # Step 1: Forecast the residual component using the fitted SARIMA model
        # The parent forecast method handles ARIMA forecast + seasonal inversion
        # It uses self.original_series (which should be the OLS residuals) for inversion.
        resid_forecast = super().forecast(steps)

        # Step 2: Calculate the exogenous component of the forecast
        # Add intercept column to future exogenous data
        Xf_with_intercept = np.hstack([np.ones((steps, 1)), Xf])
        exog_component = Xf_with_intercept.dot(self.beta)

        # Step 3: Combine residual forecast and exogenous component
        final_forecast = resid_forecast + exog_component

        return final_forecast

    # Static methods suggest_order and find_best_order should also use statsmodels for reliability

    @staticmethod
    def suggest_order(  # noqa: D417
        endog,
        exog,
        max_p=3,
        max_d=1,
        max_q=3,
        max_P=2,
        max_D=1,
        max_Q=2,
        max_m=12,
    ):
        """Suggest SARIMAX order using statsmodels AIC after OLS detrending.

        Note: This is a heuristic. Full SARIMAX fitting is simultaneous.
        """
        warnings.warn(
            "Suggesting SARIMAX order based on residuals after OLS. Using statsmodels for underlying SARIMA suggestion.",
            UserWarning,
            stacklevel=2,
        )
        y = np.asarray(endog, dtype=float)
        X = np.asarray(exog, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(y) != X.shape[0]:
            raise ValueError("endog and exog length mismatch.")

        # OLS detrending
        try:
            X_int = np.hstack([np.ones((X.shape[0], 1)), X])
            beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
            resid = y - X_int.dot(beta)
        except np.linalg.LinAlgError:
            raise RuntimeError("Failed OLS for detrending in suggest_order.")  # noqa: B904

        # Suggest SARIMA order on residuals
        return SARIMA.suggest_order(
            resid, max_p, max_d, max_q, max_P, max_D, max_Q, max_m
        )

    @staticmethod
    def find_best_order(  # noqa: D417
        train_endog,
        test_endog,
        train_exog,
        test_exog,
        max_p=2,
        max_d=1,
        max_q=2,
        max_P=1,
        max_D=1,
        max_Q=1,
        max_m=12,
    ):
        """Find best SARIMAX order using grid search (MSE) via statsmodels."""
        warnings.warn(
            "Using statsmodels.tsa.statespace.SARIMAX for reliable order search (MSE based).",
            UserWarning,
            stacklevel=2,
        )
        try:
            from itertools import product

            import statsmodels.api as sm
        except ImportError as e:
            raise ImportError(
                "Please install statsmodels: pip install statsmodels"
            ) from e

        y_train = np.asarray(train_endog)
        y_test = np.asarray(test_endog)
        X_train = np.asarray(train_exog)
        X_test = np.asarray(test_exog)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        best_mse = float("inf")
        best_order = ((0, 0, 0), (0, 0, 0, 1))
        n_test = len(y_test)

        # Parameter ranges and potential m values (same logic as SARIMA.find_best_order)
        p_range = range(max_p + 1)
        d_range = range(max_d + 1)
        q_range = range(max_q + 1)
        P_range = range(max_P + 1)
        D_range = range(max_D + 1)
        Q_range = range(max_Q + 1)
        potential_m = [1]
        if max_m > 1 and len(y_train) > 2 * max_m:
            potential_m.append(max_m)
        if max_m >= 12 and len(y_train) > 24:
            potential_m.append(12)
        if max_m >= 7 and len(y_train) > 14:
            potential_m.append(7)
        if max_m >= 4 and len(y_train) > 8:
            potential_m.append(4)
        potential_m = sorted(set(potential_m))

        print(f"Searching SARIMAX orders (MSE) for m in {potential_m}...")
        non_seasonal_orders = list(product(p_range, d_range, q_range))
        seasonal_orders = list(product(P_range, D_range, Q_range))
        total_combinations = (
            len(non_seasonal_orders) * len(seasonal_orders) * len(potential_m)
        )
        count = 0

        for m in potential_m:
            if m <= 1:
                seasonal_params = [(0, 0, 0)]
                m_val = 1
            else:
                seasonal_params = seasonal_orders
                m_val = m

            for order in non_seasonal_orders:
                for seasonal_order_part in seasonal_params:
                    count += 1
                    current_seasonal_order = seasonal_order_part + (m_val,)
                    print(
                        f"[{count}/{total_combinations}] Trying order={order}, seasonal_order={current_seasonal_order}...",
                        end="\r",
                    )

                    if order == (0, 0, 0) and current_seasonal_order == (
                        0,
                        0,
                        0,
                        m_val,
                    ):
                        continue
                    if (
                        order[0] == 0
                        and order[2] == 0
                        and seasonal_order_part[0] == 0
                        and seasonal_order_part[2] == 0
                        and (order[1] > 0 or seasonal_order_part[1] > 0)
                    ):
                        continue

                    try:
                        # Use statsmodels SARIMAX
                        model = sm.tsa.statespace.SARIMAX(
                            y_train,
                            exog=X_train,
                            order=order,
                            seasonal_order=current_seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            simple_differencing=False,
                        )
                        results = model.fit(disp=False)
                        # Forecast using future exogenous variables
                        forecast = results.get_forecast(
                            steps=n_test, exog=X_test
                        ).predicted_mean

                        mse = np.mean((y_test - forecast) ** 2)

                        if mse < best_mse:
                            best_mse = mse
                            best_order = (order, current_seasonal_order)
                            print(
                                f"\nNew best: order={best_order[0]}, seasonal={best_order[1]}, MSE={best_mse:.4f}"
                            )

                    except Exception:
                        # print(f"\nFailed for order={order}, seasonal={current_seasonal_order}: {e}")
                        continue

        print(f"\nBest Order (MSE): {best_order} with MSE: {best_mse:.4f}")
        return best_order
