import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import Ridge
from sega_learn.utils import Metrics, make_regression
from sega_learn.utils.animator import RegressionAnimation

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=1, noise=0.5, random_state=42)


# Create the animation using RegressionAnimation
animator = RegressionAnimation(
    model=Ridge,
    X=X,
    y=y,
    test_size=0.25,
    dynamic_parameter="max_iter",
    static_parameters={"alpha": 1.0, "fit_intercept": True},
    keep_previous=True,
    metric_fn=[
        Metrics.mean_squared_error,
        Metrics.mean_absolute_error,
        Metrics.r_squared,
    ],
)

# Set up the plot
animator.setup_plot(
    title="Ridge Regression Animation",
    xlabel="Feature Coefficient",
    ylabel="Target Value",
    legend_loc="upper left",
    grid=True,
    figsize=(12, 6),
)

# Create and save the animation
max_iter_range = range(
    1, 10_000, 100
)  # Windows from 10 to 10,000 (total 100 iterations)
animator.animate(frames=max_iter_range, interval=150, blit=True, repeat=False)
animator.save(
    filename="examples/utils/plots/animator_ridge.gif",
    writer="pillow",
    fps=10,
    dpi=100,
)
