import warnings

import numpy as np


def _centered_moving_average(series, window):
    """Calculates centered moving average, handling even/odd windows."""
    if window % 2 == 1:  # Odd window
        # Use convolution for efficiency
        weights = np.repeat(1.0, window) / window
        cma = np.convolve(series, weights, "valid")
        # Pad NaNs to match original series length (centered)
        pad_size = (len(series) - len(cma)) // 2
        return np.pad(
            cma, (pad_size, len(series) - len(cma) - pad_size), constant_values=np.nan
        )
    else:  # Even window -> 2xMA
        # First MA of window `window`
        ma1 = np.convolve(series, np.repeat(1.0, window) / window, "valid")
        # Second MA of order 2 on the result, centered correctly
        weights2 = np.repeat(1.0, 2) / 2
        cma = np.convolve(ma1, weights2, "valid")
        # Pad NaNs (calculation is slightly more complex for centering 2xMA)
        # Total length reduction is (window - 1) + (2 - 1) = window
        pad_size = window // 2
        return np.pad(
            cma, (pad_size, len(series) - len(cma) - pad_size), constant_values=np.nan
        )


class AdditiveDecomposition:
    """Performs classical additive decomposition of a time series.

    Decomposes the series Y into Trend (T), Seasonal (S), and Residual (R) components
    such that Y = T + S + R. Assumes seasonality is constant over time.

    Attributes:
        period (int): The seasonal period.
        time_series (np.ndarray): The original time series data.
        trend (np.ndarray): The estimated trend component.
        seasonal (np.ndarray): The estimated seasonal component.
        residual (np.ndarray): The estimated residual component.
    """

    def __init__(self, period):
        """Initialize the AdditiveDecomposition model.

        Args:
            period (int): The seasonal period (e.g., 12 for monthly, 7 for daily). Must be > 1.
        """
        if not isinstance(period, int) or period <= 1:
            raise ValueError("Period must be an integer greater than 1.")
        self.period = period
        self.time_series = None
        self.trend = None
        self.seasonal = None
        self.residual = None

    def fit(self, time_series):
        """Perform additive decomposition on the time series.

        Args:
            time_series (array-like): The time series data. Must be 1-dimensional
                                      and have length >= 2 * period.
        """
        self.time_series = np.asarray(time_series, dtype=float).flatten()
        n = len(self.time_series)

        if n < 2 * self.period:
            raise ValueError(
                f"Time series length ({n}) must be at least twice the period ({self.period})."
            )
        if np.isnan(self.time_series).any():
            warnings.warn(
                "Time series contains NaNs. Decomposition results may be affected.",
                UserWarning,
                stacklevel=2,
            )

        # 1. Estimate Trend (T) using centered moving average
        self.trend = _centered_moving_average(self.time_series, self.period)

        # 2. Detrend the series: Y_detrended = Y - T
        detrended = self.time_series - self.trend  # Contains NaNs where trend is NaN

        # 3. Estimate Seasonal component (S)
        # Average the detrended values for each season
        seasonal_factors = np.full(self.period, np.nan)
        for i in range(self.period):
            # Get all detrended values for season i (0, p, 2p, ...)
            season_values = detrended[i :: self.period]
            # Calculate mean ignoring NaNs
            if not np.all(np.isnan(season_values)):
                seasonal_factors[i] = np.nanmean(season_values)

        # Adjust seasonal factors to sum to zero
        if not np.isnan(seasonal_factors).any():
            seasonal_factors -= np.mean(seasonal_factors)
        else:
            warnings.warn(
                "Could not estimate all seasonal factors due to NaNs. Seasonal component might be incomplete.",
                UserWarning,
                stacklevel=2,
            )

        # Tile the seasonal factors to match the length of the series
        self.seasonal = np.tile(seasonal_factors, n // self.period + 1)[:n]

        # 4. Estimate Residual component (R): R = Y - T - S
        self.residual = self.time_series - self.trend - self.seasonal

        # Optionally, fill NaNs at ends of components if desired (e.g., with zeros or mean)
        # For simplicity, we leave them as NaNs here.

    def get_components(self):
        """Return the calculated components."""
        if self.trend is None:
            raise ValueError("Model has not been fitted yet.")
        return {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "residual": self.residual,
        }

    def reconstruct(self):
        """Reconstruct the series from components (Y = T + S + R)."""
        components = self.get_components()
        # Use nan_to_num to handle potential NaNs if adding them directly
        return np.nansum(
            [components["trend"], components["seasonal"], components["residual"]],
            axis=0,
        )


class MultiplicativeDecomposition:
    """Performs classical multiplicative decomposition of a time series.

    Decomposes the series Y into Trend (T), Seasonal (S), and Residual (R) components
    such that Y = T * S * R. Assumes seasonality changes proportionally to the trend.
    Requires the time series to be strictly positive.

    Attributes:
        period (int): The seasonal period.
        time_series (np.ndarray): The original time series data.
        trend (np.ndarray): The estimated trend component.
        seasonal (np.ndarray): The estimated seasonal component.
        residual (np.ndarray): The estimated residual component.
    """

    def __init__(self, period):
        """Initialize the MultiplicativeDecomposition model.

        Args:
            period (int): The seasonal period (e.g., 12 for monthly, 7 for daily). Must be > 1.
        """
        if not isinstance(period, int) or period <= 1:
            raise ValueError("Period must be an integer greater than 1.")
        self.period = period
        self.time_series = None
        self.trend = None
        self.seasonal = None
        self.residual = None

    def fit(self, time_series):
        """Perform multiplicative decomposition on the time series.

        Args:
            time_series (array-like): The time series data. Must be 1-dimensional,
                                      strictly positive, and have length >= 2 * period.
        """
        self.time_series = np.asarray(time_series, dtype=float).flatten()
        n = len(self.time_series)

        if n < 2 * self.period:
            raise ValueError(
                f"Time series length ({n}) must be at least twice the period ({self.period})."
            )
        if np.any(self.time_series <= 0):
            raise ValueError(
                "Multiplicative decomposition requires strictly positive time series values."
            )
        if np.isnan(self.time_series).any():
            warnings.warn(
                "Time series contains NaNs. Decomposition results may be affected.",
                UserWarning,
                stacklevel=2,
            )

        # 1. Estimate Trend (T) using centered moving average
        self.trend = _centered_moving_average(self.time_series, self.period)

        # 2. Detrend the series: Y_detrended = Y / T
        # Use np.errstate to avoid warnings for division by zero/NaN if trend has them
        with np.errstate(divide="ignore", invalid="ignore"):
            detrended = (
                self.time_series / self.trend
            )  # Contains NaNs/Infs where trend is NaN/zero

        # 3. Estimate Seasonal component (S)
        # Average the detrended values for each season
        seasonal_factors = np.full(self.period, np.nan)
        for i in range(self.period):
            # Get all detrended values for season i, ignoring non-finite values
            season_values = detrended[i :: self.period]
            finite_season_values = season_values[np.isfinite(season_values)]
            if len(finite_season_values) > 0:
                seasonal_factors[i] = np.mean(
                    finite_season_values
                )  # Use mean of finite values

        # Adjust seasonal factors to average to 1
        if not np.isnan(seasonal_factors).any():
            seasonal_factors /= np.mean(seasonal_factors)
        else:
            warnings.warn(
                "Could not estimate all seasonal factors due to NaNs/Infs. Seasonal component might be incomplete.",
                UserWarning,
                stacklevel=2,
            )

        # Tile the seasonal factors to match the length of the series
        self.seasonal = np.tile(seasonal_factors, n // self.period + 1)[:n]

        # 4. Estimate Residual component (R): R = Y / (T * S)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = self.trend * self.seasonal
            self.residual = self.time_series / denominator
            # Replace Infs potentially created by division by zero with NaNs
            self.residual[~np.isfinite(self.residual)] = np.nan

    def get_components(self):
        """Return the calculated components."""
        if self.trend is None:
            raise ValueError("Model has not been fitted yet.")
        return {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "residual": self.residual,
        }

    def reconstruct(self):
        """Reconstruct the series from components (Y = T * S * R)."""
        components = self.get_components()
        # Use np.nan_to_num before multiplying? Or just multiply allowing NaNs?
        # Multiplying preserves NaNs appropriately.
        return components["trend"] * components["seasonal"] * components["residual"]
