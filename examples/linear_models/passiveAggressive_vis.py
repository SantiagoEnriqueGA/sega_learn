import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import numpy as np
from sega_learn.linear_models import PassiveAggressiveRegressor

np.random.seed(42)  # Random seed for reproducibility

# Example data, contains many outliers
X = np.array(
    [
        -0.848,
        -0.800,
        -0.704,
        -0.632,
        -0.488,
        -0.472,
        -0.368,
        -0.336,
        -0.280,
        -0.200,
        -0.00800,
        -0.0840,
        0.0240,
        0.100,
        0.124,
        0.148,
        0.232,
        0.236,
        0.324,
        0.356,
        0.368,
        0.440,
        0.512,
        0.548,
        0.660,
        0.640,
        0.712,
        0.752,
        0.776,
        0.880,
        0.920,
        0.944,
        -0.108,
        -0.168,
        -0.720,
        -0.784,
        -0.224,
        -0.604,
        -0.740,
        -0.0440,
        0.388,
        -0.0200,
        0.752,
        0.416,
        -0.0800,
        -0.348,
        0.988,
        0.776,
        0.680,
        0.880,
        -0.816,
        -0.424,
        -0.932,
        0.272,
        -0.556,
        -0.568,
        -0.600,
        -0.716,
        -0.796,
        -0.880,
        -0.972,
        -0.916,
        0.816,
        0.892,
        0.956,
        0.980,
        0.988,
        0.992,
        0.00400,
    ]
).reshape(-1, 1)
y = np.array(
    [
        -0.917,
        -0.833,
        -0.801,
        -0.665,
        -0.605,
        -0.545,
        -0.509,
        -0.433,
        -0.397,
        -0.281,
        -0.205,
        -0.169,
        -0.0531,
        -0.0651,
        0.0349,
        0.0829,
        0.0589,
        0.175,
        0.179,
        0.191,
        0.259,
        0.287,
        0.359,
        0.395,
        0.483,
        0.539,
        0.543,
        0.603,
        0.667,
        0.679,
        0.751,
        0.803,
        -0.265,
        -0.341,
        0.111,
        -0.113,
        0.547,
        0.791,
        0.551,
        0.347,
        0.975,
        0.943,
        -0.249,
        -0.769,
        -0.625,
        -0.861,
        -0.749,
        -0.945,
        -0.493,
        0.163,
        -0.469,
        0.0669,
        0.891,
        0.623,
        -0.609,
        -0.677,
        -0.721,
        -0.745,
        -0.885,
        -0.897,
        -0.969,
        -0.949,
        0.707,
        0.783,
        0.859,
        0.979,
        0.811,
        0.891,
        -0.137,
    ]
).reshape(-1)


reg = PassiveAggressiveRegressor(C=0.01, max_iter=1000, tol=1e-4)
reg.fit(X, y, save_steps=True)

# Plot the Passive Aggressive Regressor
plt.figure(figsize=(8, 8))

# For each iteration, plot the regression line
colors = plt.cm.Greens(np.linspace(0.5, 1, len(reg.steps_) - 1))
for i, (coef, intercept) in enumerate(reg.steps_):
    y_pred = X * coef + intercept
    plt.plot(
        X, y_pred, label=f"Iteration {i}", color=colors[i] if i < len(colors) else "red"
    )
plt.scatter(X, y, color="lightblue", label="Data Points", alpha=0.75)
plt.xlabel("Feature 0")
plt.ylabel("Target")
plt.title("Passive Aggressive Regressor")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
# plt.show()
plt.savefig("examples/linear_models/plots/passiveAggressive.png", dpi=300)
