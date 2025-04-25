import os
import sys
import warnings

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.time_series import MultiplicativeDecomposition
from sega_learn.utils import Metrics, make_time_series

warnings.filterwarnings("ignore", category=UserWarning)
mean_squared_error = Metrics.mean_squared_error

# Generate a synthetic time series
period = 25
time_series = make_time_series(
    n_samples=1,
    n_timestamps=300,
    n_features=1,
    trend="linear",
    seasonality="cosine",
    seasonality_period=period,
    noise=0.5,
    random_state=1,
)

# Flatten the time series to 1D
time_series = time_series.flatten()

# Perform Multiplicative Decomposition
multiplicative_model = MultiplicativeDecomposition(period=period)

# Fit the model to the time series
multiplicative_model.fit(time_series)

# Get the decomposed components
components = multiplicative_model.get_components()
trend = components["trend"]
seasonal = components["seasonal"]
residual = components["residual"]

# Plot the decomposed components
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Plot 1: Original Time Series
axes[0].plot(time_series, label="Original Time Series", color="blue")
axes[0].set_ylabel("Original")
axes[0].legend(loc="upper left")
axes[0].grid(True)

# Plot 2: Trend Component
axes[1].plot(trend, label="Trend Component", color="orange")
axes[1].set_ylabel("Trend")
axes[1].legend(loc="upper left")
axes[1].grid(True)

# Plot 3: Seasonal Component
axes[2].plot(seasonal, label="Seasonal Component", color="green")
axes[2].set_ylabel("Seasonal")
axes[2].legend(loc="upper left")
axes[2].grid(True)

# Plot 4: Residual Component
axes[3].plot(residual, label="Residual Component", color="red", linestyle=":")
# Add a horizontal line at 0 for reference
axes[3].axhline(0, color="black", linestyle="--", linewidth=0.8, label="Zero Line")
axes[3].set_ylabel("Residual")
axes[3].set_xlabel("Time Step")
axes[3].legend(loc="upper left")
axes[3].grid(True)

# Add overall title and adjust layout
fig.suptitle("Multiplicative Decomposition Breakdown", fontsize=14)
plt.tight_layout()
# plt.show()
plt.savefig(
    "examples/time_series/plots/decom_multiplicative_decomposition.png", dpi=300
)
