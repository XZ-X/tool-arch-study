#!/bin/bash

# Save the original directory
ORIG_DIR=$(pwd)

# Change to the simple-agent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Number of parallel workers (default: 4)
NUM_WORKERS=${1:-4}

# Configuration
AGENT_NAME=Sonnet-4.5-python
SPLIT=vmini
RUN_ID=simple-agent-tool-Sonnet-4.5-py-interface-med-temp
AGENT_CONFIG=agent_configs/Sonnet-4.5-python-med-temp.yaml
WORKING_DIR=evaluation/$RUN_ID

echo "Running $AGENT_NAME on $SPLIT split with $NUM_WORKERS workers"
echo "Working directory: $WORKING_DIR"

# Create working directory if it doesn't exist
mkdir -p "$WORKING_DIR"

# Run the evaluation
cd evaluation
sg docker -c "python3 swebench_live.py \
    --split $SPLIT \
    --agent-config ../$AGENT_CONFIG \
    --provider $AGENT_NAME \
    --working-dir ../$WORKING_DIR \
    --swebench-live-harness-path ./SWE-bench-Live \
    --nproc $NUM_WORKERS --num_samples 10 --run-id $RUN_ID"

# Return to original directory
cd "$ORIG_DIR"

echo "Done! Results saved to: $SCRIPT_DIR/$WORKING_DIR"
