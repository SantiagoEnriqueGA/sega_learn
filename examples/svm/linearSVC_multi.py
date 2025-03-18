import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.svm import *
from sega_learn.utils.makeData import make_blobs


# Set random seed for reproducibility
np.random.seed(42)

# Generate a multi-class classification dataset
X, y, _ = make_blobs(n_samples=1000, n_features=2, centers=3, random_state=42, cluster_std=3.0, center_box=(-15.0, 15.0))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit our LinearSVC model
svc = LinearSVC(C=1.0, tol=1e-4, max_iter=2000, learning_rate=0.01)
svc.fit(X_scaled, y)

# Evaluate the model
accuracy = svc.score(X_scaled, y)
# print(f"LinearSVC - Accuracy: {accuracy:.4f}")

# Plotting the decision boundary
def plot_decision_boundary(X, y, model, ax):
    cmap = plt.cm.cividis
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', marker='o')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    colors = cmap(np.linspace(0, 1, len(model.classes_)))
    for i, m in enumerate(model.models_):
        # Plot the SVM decision boundary (the line where w·x + b = 0)
        w_original = m.w
        b_original = m.b
        
        # Transform w back to the original feature space
        w = scaler.scale_ * w_original
        b = b_original - np.dot(scaler.mean_, w_original)
        
        # Plot the decision boundary line
        slope = -w[0] / w[1]        
        intercept = -b / w[1]
        x_support = np.linspace(x_min, x_max, 2)
        y_support = slope * x_support + intercept
        
        # dashed line for each class
        ax.plot(x_support, y_support, 'k--', linewidth=2, color=colors[i], label=f'Decision boundary {i}')
        
        # Plot the margins
        margin_width = 1 / np.linalg.norm(w)
        margin_top = y_support + margin_width
        margin_bottom = y_support - margin_width
        ax.plot(x_support, margin_top, ':', linewidth=2, color=colors[i], alpha=0.5)
        ax.plot(x_support, margin_bottom, ':', linewidth=2, color=colors[i], alpha=0.5)


# Create a plot
fig, ax = plt.subplots(figsize=(10, 8))
plot_decision_boundary(X_scaled, y, svc, ax)
plt.title(f"LinearSVC Decision Boundary - Accuracy: {accuracy:.4f}")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1)
plt.ylim(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1)
plt.tight_layout()
plt.legend()
# plt.show()
plt.savefig('examples/svm/plots/linear_svc_multi.png', dpi=300)

