#!/bin/bash

ORIG_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_WORKERS=${1:-4}

AGENT_NAME=Sonnet-4.5-python
DATASET=${DATASET:-vtiny}
RUN_ID=simple-agent-verified-Sonnet-4.5-py-interface-med-temp
WORKING_DIR=evaluation/$RUN_ID
AGENT_CONFIG=agent_configs/Sonnet-4.5-python-med-temp.yaml

echo "Running $AGENT_NAME on Verified $DATASET with $NUM_WORKERS workers"
echo "Working directory: $WORKING_DIR"

mkdir -p "$WORKING_DIR"

cd evaluation
sg docker -c "python3 swebench.py \
    --dataset $DATASET \
    --agent-config ../$AGENT_CONFIG \
    --provider $AGENT_NAME \
    --working-dir ../$WORKING_DIR \
    --swebench-harness-path ./SWE-bench \
    --nproc $NUM_WORKERS --num_samples 5 --run-id $RUN_ID"

cd "$ORIG_DIR"

echo "Done! Results saved to: $SCRIPT_DIR/$WORKING_DIR"
