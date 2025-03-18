import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.svm import *
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

# Random seed for reproducibility
np.random.seed(42)

# Generate data
pnts = 300
X = np.linspace(0, 5, pnts)
y = 10 * np.sin(X) + np.random.normal(0, 1, pnts)
X_reshaped = X.reshape(-1, 1)

# Define kernel parameters
kernels = ['poly', 'rbf', 'sigmoid']
colors = ['red', 'purple', 'orange']
kernel_params = {
    'poly': {'degree': 3, 'gamma': 0.1, 'coef0': 1.0},
    'rbf': {'gamma': 0.5},
    'sigmoid': {'gamma': 0.1, 'coef0': 1.0}
}

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig)

# Create a combined plot (top row spanning all columns)
ax_combined = fig.add_subplot(gs[0, :])
ax_combined.scatter(X, y, color='blue', label='Data Points', alpha=0.5)
ax_combined.plot(X, 10 * np.sin(X), color='green', label='True Function', linewidth=2)

# Create individual plots for each kernel
axs = [fig.add_subplot(gs[1, i]) for i in range(3)]

# Train models and plot results
r2_scores = []
for i, kernel in enumerate(kernels):
    # Get kernel specific parameters
    params = kernel_params[kernel]
    
    # Create and fit SVR model with specific kernel
    svr = GeneralizedSVR(
        C=1.0, 
        tol=1e-4, 
        max_iter=1000, 
        learning_rate=0.01, 
        epsilon=0.1,
        kernel=kernel, 
        **params
    )
    
    # Fit and get predictions
    svr.fit(X_reshaped, y)
    y_pred = svr.predict(X_reshaped)
    score = r2_score(y, y_pred)
    r2_scores.append(score)
    
    # Plot on combined plot
    ax_combined.plot(X, y_pred, color=colors[i], label=f'{kernel} Kernel', linewidth=2, alpha=0.7)
    
    # Plot on individual subplot
    axs[i].scatter(X, y, color='blue', label='Data Points', alpha=0.3, s=10)
    axs[i].plot(X, y_pred, color=colors[i], label=f'Prediction (R² = {score:.3f})', linewidth=2)
    axs[i].plot(X, 10 * np.sin(X), color='green', label='True Function', linewidth=2)
    
    # Set subplot title and labels
    param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
    axs[i].set_title(f'{kernel} Kernel\nParameters: {param_str}')
    # axs[i].set_xlabel('Feature 0')
    # axs[i].set_ylabel('Target')
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)

# Set combined plot title and labels
ax_combined.set_title('SVR Kernel Comparison on Sine Curve with Noise', fontsize=16)
# ax_combined.set_xlabel('Feature 0', fontsize=12)
# ax_combined.set_ylabel('Target', fontsize=12)
ax_combined.legend()
ax_combined.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
# plt.show()
plt.savefig('examples/svm/plots/generalizedSVR_kernels.png', dpi=300)