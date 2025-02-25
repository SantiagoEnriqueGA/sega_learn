import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV files
cpu_results = pd.read_csv("tests_performance/nn/nn_timing_results.csv")
cupy_results = pd.read_csv("tests_performance/nn/nn_timing_results_cupy.csv")

# Add a column to distinguish CPU and CuPy results
cpu_results["Implementation"] = "CPU"
cupy_results["Implementation"] = "CuPy"

# Combine the datasets
results = pd.concat([cpu_results, cupy_results])

# Create a pivot table for easier comparison
pivot_results = results.pivot(index='Function', columns='Implementation', values='Average Time (s)').reset_index()
pivot_results['Speedup (%)'] = ((pivot_results['CPU'] - pivot_results['CuPy']) / pivot_results['CPU'] * 100)

# Sort functions by CPU execution time for better readability
pivot_results = pivot_results.sort_values('CPU', ascending=False)

# Set Seaborn style with improved readability
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['font.family'] = 'sans-serif'

# First plot: Timing comparison
plt.figure(figsize=(14, 8))
melted_data = pd.melt(pivot_results, id_vars=['Function'], value_vars=['CPU', 'CuPy'])
melted_data.rename(columns={'variable': 'Implementation'}, inplace=True)

sns.barplot(x='Function', y='value', hue='Implementation', 
            data=melted_data,
            palette=['#3A6EA5', '#FF6B6B'])

plt.title('Neural Network Function Timing: CPU vs CuPy', fontsize=18, fontweight='bold')
plt.ylabel('Average Time (seconds)', fontsize=14)
plt.xlabel('')
plt.xticks(rotation=45, fontsize=12)
plt.legend(title='Implementation', fontsize=12, title_fontsize=14)

# Add value labels on bars
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.3f', fontsize=10)

plt.tight_layout()
plt.show()

# Second plot: Speedup percentage
plt.figure(figsize=(14, 8))
speedup_bars = sns.barplot(x='Function', y='Speedup (%)', data=pivot_results, 
                           palette=['#50C878'])
plt.title('CuPy Speedup Percentage Compared to CPU', fontsize=18, fontweight='bold')
plt.ylabel('Speedup (%)', fontsize=14)
plt.xlabel('Function', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

# Add value labels on speedup bars
for i, v in enumerate(pivot_results['Speedup (%)']):
    plt.text(i, max(v + 5, 5) if v > 0 else v - 15, 
             f"{v:.1f}%", ha='center', fontsize=10, 
             fontweight='bold', color='black')

# Highlight negative speedups (where CPU is faster)
for i, v in enumerate(pivot_results['Speedup (%)']):
    if v < 0:
        speedup_bars.patches[i].set_facecolor('#FF7F7F')  # Light red for negative speedup

plt.tight_layout()
plt.show()

# Create a summary DataFrame for the console
summary = pivot_results[['Function', 'CPU', 'CuPy', 'Speedup (%)']].copy()
summary = summary.sort_values('Speedup (%)', ascending=False)
summary = summary.rename(columns={'CPU': 'CPU Time (s)', 'CuPy': 'CuPy Time (s)'})
summary['CPU Time (s)'] = summary['CPU Time (s)'].round(4)
summary['CuPy Time (s)'] = summary['CuPy Time (s)'].round(4)
summary['Speedup (%)'] = summary['Speedup (%)'].round(1)

print("Neural Network Performance Summary (Sorted by Speedup):")
print(summary)
