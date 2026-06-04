#!/bin/bash

# Batch run all Gemini tests using single configuration file
# Usage: ./batch_run_gemini_all.sh

echo "=========================================="
echo "Running All Gemini Tests"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run using the batch configuration file
python "$SCRIPT_DIR/main.py" --config "$SCRIPT_DIR/config/batch_gemini_all.yaml"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
