import re
import pandas as pd
import matplotlib.pyplot as plt

# List of log files
log_files = {
    "1e4": "gpu_MCPi_1e4.log",
    "1e5": "gpu_MCPi_1e5.log",
    "1e6": "gpu_MCPi_1e6.log",
    "1e7": "gpu_MCPi_1e7.log"
}

# API call names to extract
api_calls = ["cudaMalloc", "cudaLaunchKernel", "cudaMemcpy", "cuDeviceGetAttribute", "cudaFree", "cuDeviceGetName", "cuDeviceGetPCIBusId"]

# Initialize data dictionary
data = {api: [] for api in api_calls}
data["Sample Size"] = []

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
    data["Sample Size"].append(size)
    for api in api_calls:
        data[api].append(parsed_data[api])

# Convert data to DataFrame
df = pd.DataFrame(data)
df.set_index("Sample Size", inplace=True)

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
df.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")

# Labels and title
ax.set_ylabel("Execution Time (%)")
ax.set_xlabel("Sample Size")
ax.set_title("Breakdown of Execution Time by API Calls (Percentage)")
ax.legend(title="API Calls", bbox_to_anchor=(1.05, 1), loc="upper left")

# Save the plot
plot_filename = "MCPi_api_execution_time_breakdown.png"
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plot_filename, dpi=300)  # Save as high-resolution image

# Display a confirmation message
print(f"Plot saved as '{plot_filename}'")
