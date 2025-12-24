#!/bin/bash

echo "========================================"
echo "Musubi Tuner Web UI"
echo "========================================"
echo

cd "$(dirname "$0")"

# Check if .venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    echo "Please ensure .venv exists and is properly configured"
    exit 1
fi

# Force Accelerate to use the repo config (single-GPU by default)
export ACCELERATE_CONFIG_FILE="$(pwd)/accelerate_config.yaml"

# Check if the package is installed by trying to import it
python -c "import musubi_tuner" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing/updating dependencies..."
    pip install -e ".[gui]"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Verify torch is available (GUI/training tabs typically need it)
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo
    echo "ERROR: PyTorch is not installed in this venv."
    echo "Install with ONE of these (pick your CUDA):"
    echo "  pip install -e \".[gui,cu128]\""
    echo "  pip install -e \".[gui,cu124]\""
    echo
    exit 1
fi

echo "Starting Musubi Tuner Web UI..."
echo

python musubi_gui.py --inbrowser "$@"


