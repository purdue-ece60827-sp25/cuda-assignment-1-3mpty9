import re
import pandas as pd
import matplotlib.pyplot as plt

# List of log files
log_files = {
    "8": "gpu_MCPi_rs8.log",
    "16": "gpu_MCPi_rs16.log",
    "32": "gpu_MCPi_rs32.log",
    "64": "gpu_MCPi_rs64.log"
}

# API call names to extract
api_calls = ["cudaMalloc", "cudaLaunchKernel", "cudaMemcpy", "cuDeviceGetAttribute", "cudaFree", "cuDeviceGetName", "cuDeviceGetPCIBusId"]

# Initialize data dictionary
data = {api: [] for api in api_calls}
data["Reduce Size"] = []

# Function to parse log files
def parse_log(file_path):
    percentages = {api: 0.0 for api in api_calls}  # Default to 0% if not found
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"(\d+\.\d+)%\s+\d+\.\d+ms.*\b(cuda\w+)\b", line)
            if match:
                time_percent = float(match.group(1))
                api_name = match.group(2)
                if api_name in api_calls:
                    percentages[api_name] = time_percent
    return percentages

# Process each log file
for size, log_file in log_files.items():
    parsed_data = parse_log(log_file)
    data["Reduce Size"].append(size)
    for api in api_calls:
        data[api].append(parsed_data[api])

# Convert data to DataFrame
df = pd.DataFrame(data)
df.set_index("Reduce Size", inplace=True)

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")

# Labels and title
ax.set_ylabel("Execution Time (%)")
ax.set_xlabel("Reduce Size")
ax.set_title("Breakdown of Execution Time by API Calls (Percentage)")
ax.legend(title="API Calls", bbox_to_anchor=(1.05, 1), loc="upper left")

# Save the plot
plot_filename = "MCPi_api_reduce_size_execution_time_breakdown.png"
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)  # Save as high-resolution image

# Display a confirmation message
print(f"Plot saved as '{plot_filename}'")
