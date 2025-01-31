import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data extracted from logs


data = {
    "8": {"generatePoints": 99.5, "reduceCounts": 0.04, "CUDA memcpy DtoH": 0.00},
    "16": {"generatePoints": 99.97, "reduceCounts": 0.02, "CUDA memcpy DtoH": 0.00},
    "32": {"generatePoints": 99.98, "reduceCounts": 0.01, "CUDA memcpy DtoH": 0.00},
    "64": {"generatePoints": 99.99, "reduceCounts": 0.01, "CUDA memcpy DtoH": 0.00},
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
df.T.plot(kind="bar", stacked=True, ax=ax)

# Labels and title
ax.set_xlabel("Reduce Size")
ax.set_ylabel("GPU Activities' Time Percentage (%)")
ax.set_title("GPU Activities' Time Percentage per Sample Size")
ax.legend(title="GPU Activities")

# Save the plot
plot_filename = "MCPi_kernel_reduce_size_execution_time_breakdown.png"
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)  # Save as high-resolution image

# Display a confirmation message
print(f"Plot saved as '{plot_filename}'")
