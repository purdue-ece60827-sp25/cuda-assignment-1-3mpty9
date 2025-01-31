import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Extracted data from log files
api_data = {
    "Vector Size": [5, 10, 15, 20],
    "CUDA memcpy HtoD (%)": [34.51, 38.94, 66.16, 79.10],
    "CUDA memcpy DtoH (%)": [32.76, 30.77, 27.91, 20.15],
    "saxpy_gpu (%)": [32.74, 30.29, 5.93, 0.75]
}

# Convert to DataFrame
df_api = pd.DataFrame(api_data)

# Plot stacked bar chart with time percentages displayed
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.6
bottoms = np.zeros(len(df_api["Vector Size"]))

api_labels = ["CUDA memcpy HtoD ", "CUDA memcpy DtoH", "saxpy_gpu ", "cudaMemcpy"]
api_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, col in enumerate(df_api.columns[1:]):
    bars = ax.bar(df_api["Vector Size"], df_api[col], bar_width, label=api_labels[i], bottom=bottoms, color=api_colors[i])
    for bar in bars:
        height = bar.get_height()
        if height > 1:  # Only show labels for significant values
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{height:.1f}%', ha='center', va='center', fontsize=10, color='black')
    bottoms += df_api[col]

# Labels and title
ax.set_xlabel("Vector Size (2^)")
ax.set_ylabel("Execution Time (%)")
ax.set_title("Breakdown of Execution Time by Kernels for SAXPY")
ax.legend()

# Show and save plot
plt.xticks(df_api["Vector Size"])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("saxpy_kernel_execution_breakdown.png", dpi=300, bbox_inches="tight")
