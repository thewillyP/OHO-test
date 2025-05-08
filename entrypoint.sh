#!/bin/bash
set -e

# Set the user variable, using the $USER environment variable, or default to 'wlp9800' if not set
USER_NAME="${USER:-wlp9800}"

# Set WANDB environment variables
export WANDB_DIR=/wandb_data
export WANDB_CACHE_DIR=/wandb_data/.cache/wandb
export WANDB_CONFIG_DIR=/wandb_data/.config/wandb
export WANDB_DATA_DIR=/wandb_data/.cache/wandb-data/
export WANDB_ARTIFACT_DIR=/wandb_data/.artifacts

# Function to append to .bashrc if the line doesn't exist
append_if_not_exists() {
    local line="$1"
    local file="$2"
    
    grep -q "$line" "$file" || echo "$line" >> "$file"
}

# Define the path to the .bashrc
USER_BASHRC="/home/${USER_NAME}/.bashrc"

touch "$USER_BASHRC"

# Ensure the necessary environment variables and configurations are set in .bashrc
append_if_not_exists "source /.singularity.d/env/10-docker2singularity.sh" "$USER_BASHRC"
append_if_not_exists 'export LD_LIBRARY_PATH="/.singularity.d/libs"' "$USER_BASHRC"

# Apply the changes
source "$USER_BASHRC"

# Check inputs
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <sweep_id> <controller_url>"
    exit 1
fi

SWEEP_ID="$1"
CONTROLLER_URL="$2"

git config --global --add safe.directory /OHO-test

echo "Starting sweep-agent with:"
echo "  Sweep ID: $SWEEP_ID"
echo "  Controller URL: $CONTROLLER_URL"

sweep-agent "$SWEEP_ID" "$CONTROLLER_URL"
