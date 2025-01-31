import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data extracted from logs


data = {
    "1e4": {"generatePoints": 98.61, "reduceCounts": 1.08, "CUDA memcpy DtoH": 0.31},
    "1e5": {"generatePoints": 99.83, "reduceCounts": 0.13, "CUDA memcpy DtoH": 0.04},
    "1e6": {"generatePoints": 99.98, "reduceCounts": 0.01, "CUDA memcpy DtoH": 0.00},
    "1e7": {"generatePoints": 100.00, "reduceCounts": 0.00, "CUDA memcpy DtoH": 0.00},
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
df.T.plot(kind="bar", stacked=True, ax=ax)

# Labels and title
ax.set_xlabel("Sample Size")
ax.set_ylabel("GPU Activities' Time Percentage (%)")
ax.set_title("GPU Activities' Time Percentage per Sample Size")
ax.legend(title="GPU Activities")

# Save the plot
plot_filename = "MCPi_kernel_execution_time_breakdown.png"
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)  # Save as high-resolution image

# Display a confirmation message
print(f"Plot saved as '{plot_filename}'")
