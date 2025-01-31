import csv
import re

# Read log file
def parse_nvprof_log(log_filename, csv_filename):
    with open(log_filename, 'r') as file:
        data = file.read()
    
    # Extract API call lines using regex
    api_calls_section = re.findall(r'\s*([\d\.]+)%\s+([\d\.a-zA-Z]+)\s+\d+\s+[\d\.a-zA-Z]+\s+[\d\.a-zA-Z]+\s+[\d\.a-zA-Z]+\s+([a-zA-Z0-9_\[\]<>]+)', data)
    
    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time%", "API Name"])
        for entry in api_calls_section:
            writer.writerow([entry[0], entry[2]])
    
    print(f"CSV file '{csv_filename}' created successfully.")

# Example usage
log_filename = "gpu_MCPi_s1e7.log"  # Replace with the actual log file path
csv_filename = "gpu_MCPi_s1e7.csv"
parse_nvprof_log(log_filename, csv_filename)