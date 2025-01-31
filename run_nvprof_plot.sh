#!/bin/bash

# Set default values
NVPROF_LOG="nvprof_output.log"
OUTPUT_PREFIX="nvprof_plot"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --log) NVPROF_LOG="$2"; shift ;;
        --prefix) OUTPUT_PREFIX="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the provided arguments
python3 -c "
import sys
from reportProfile import plot_profiler_output
plot_profiler_output('$NVPROF_LOG', '$OUTPUT_PREFIX')
"

echo "Plots generated successfully with prefix: $OUTPUT_PREFIX"
