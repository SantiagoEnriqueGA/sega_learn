import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Test example one_hot_encode
# -----------------------------------------------------------------------------
# Make random data with categorical and numerical columns
num_rows =  100
data = np.empty((num_rows, 5), dtype=object)  # Use object dtype for mixed types
data[:, 0] = np.random.choice(['a', 'b', 'c'], size=num_rows)
data[:, 1] = np.random.choice(['d', 'e', 'f'], size=num_rows)
data[:, 2] = np.random.choice(['g', 'h', 'i'], size=num_rows)
# Numerical columns integers
data[:, 3:] = np.random.randint(0, 10, size=(num_rows, 2))
ohe_data = one_hot_encode(data)

print(f"Data before one-hot encoding:\n{data[:5]}")
print(f"\nData after one-hot encoding:\n{ohe_data[:5]}")


# -----------------------------------------------------------------------------
# Test examples normalize
# -----------------------------------------------------------------------------
# Make random data
num_rows = 1000
data = np.random.randn(num_rows, 5)  # Random choice from a normal distribution

# Normalize the data
l1_norm_data = normalize(data, norm='l1')
l2_norm_data = normalize(data, norm='l2')
max_norm_data = normalize(data, norm='max')
min_max_norm_data = normalize(data, norm='minmax')

# Plot all data distribution for each data
# Create subplots for each normalization method
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Define unique colors for each normalization method
colors = {
    'Original': 'blue',
    'L1 Norm': 'green',
    'L2 Norm': 'orange',
    'Max Norm': 'red',
    'Min-Max Norm': 'purple'
}

# L1 Norm
sns.kdeplot(data.flatten(), label='Original Data', linewidth=2, color=colors['Original'], ax=axes[0])
sns.kdeplot(l1_norm_data.flatten(), label='L1 Norm', linewidth=2, color=colors['L1 Norm'], ax=axes[0])
axes[0].set_title('L1 Norm vs Original Data', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True)

# L2 Norm
sns.kdeplot(data.flatten(), label='Original Data', linewidth=2, color=colors['Original'], ax=axes[1])
sns.kdeplot(l2_norm_data.flatten(), label='L2 Norm', linewidth=2, color=colors['L2 Norm'], ax=axes[1])
axes[1].set_title('L2 Norm vs Original Data', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True)

# Max Norm
sns.kdeplot(data.flatten(), label='Original Data', linewidth=2, color=colors['Original'], ax=axes[2])
sns.kdeplot(max_norm_data.flatten(), label='Max Norm', linewidth=2, color=colors['Max Norm'], ax=axes[2])
axes[2].set_title('Max Norm vs Original Data', fontsize=14)
axes[2].legend(fontsize=10)
axes[2].grid(True)

# Min-Max Norm
sns.kdeplot(data.flatten(), label='Original Data', linewidth=2, color=colors['Original'], ax=axes[3])
sns.kdeplot(min_max_norm_data.flatten(), label='Min-Max Norm', linewidth=2, color=colors['Min-Max Norm'], ax=axes[3], warn_singular=False)
axes[3].set_title('Min-Max Norm vs Original Data', fontsize=14)
axes[3].legend(fontsize=10)
axes[3].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
# plt.show()
plt.savefig('examples/utils/plots/normalized.png')

# -----------------------------------------------------------------------------
# Test example Scaler
# -----------------------------------------------------------------------------
# Make random data
num_rows = 100
data = np.random.randn(num_rows, 5)

stand_scaler = Scaler(method='standard')
stand_data = stand_scaler.fit_transform(data)

minmax_scaler = Scaler(method='minmax')
minmax_data = minmax_scaler.fit_transform(data)

norm_scaler = Scaler(method='normalize')
norm_data = norm_scaler.fit_transform(data)

print(f"\nScaler Object Stats:")
print("-"*80)
print(f"Scaler method=\'standard\': \n\tmean={stand_scaler.mean}, \n\tstd={stand_scaler.std}")
print(f"Scaler method=\'minmax\': \n\tmin={minmax_scaler.min}, \n\tmax={minmax_scaler.max}")
print(f"Scaler method=\'normalize\': \n\tnorm={norm_scaler.norm[:5]}...")
