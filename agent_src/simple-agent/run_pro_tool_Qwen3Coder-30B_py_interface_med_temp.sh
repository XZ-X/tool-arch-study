#!/bin/bash

ORIG_DIR=$(pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_WORKERS=${1:-${NUM_WORKERS:-4}}

AGENT_NAME=${AGENT_NAME:-Qwen3Coder-30B-python}
DATASET=${DATASET:-ptiny}
NUM_SAMPLES=${NUM_SAMPLES:-1}
MAX_STEPS=${MAX_STEPS:-200}
RUN_ID=${RUN_ID:-simple-agent-pro-Qwen3Coder-30B-py-interface-med-temp}
WORKING_DIR=${WORKING_DIR:-evaluation/$RUN_ID}
AGENT_CONFIG=${AGENT_CONFIG:-agent_configs/Qwen3Coder-30B-python-med-temp.yaml}
SWEBENCH_PRO_HARNESS_PATH=${SWEBENCH_PRO_HARNESS_PATH:-../../trae-agent/evaluation/SWE-bench_Pro-os}
SWEBENCH_PRO_PYTHON=${SWEBENCH_PRO_PYTHON:-../../trae-agent/evaluation/swebench-pro-venv/bin/python}
DOCKERHUB_USERNAME=${DOCKERHUB_USERNAME:-jefzda}
DOCKER_PLATFORM=${DOCKER_PLATFORM:-}
INSTANCE_IDS_FILE=${INSTANCE_IDS_FILE:-}
MODE=${MODE:-e2e}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "Running $AGENT_NAME on SWE-bench Pro $DATASET with $NUM_WORKERS workers"
echo "Working directory: $WORKING_DIR"

mkdir -p "$WORKING_DIR"

cd evaluation
CMD=(
    python3 swebench_pro.py
    --dataset "$DATASET"
    --agent-config "../$AGENT_CONFIG"
    --provider "$AGENT_NAME"
    --working-dir "../$WORKING_DIR"
    --swebench-pro-harness-path "$SWEBENCH_PRO_HARNESS_PATH"
    --eval-python "$SWEBENCH_PRO_PYTHON"
    --dockerhub-username "$DOCKERHUB_USERNAME"
    --nproc "$NUM_WORKERS"
    --num_samples "$NUM_SAMPLES"
    --max-steps "$MAX_STEPS"
    --mode "$MODE"
)

if [ -n "$DOCKER_PLATFORM" ]; then
    CMD+=(--docker-platform "$DOCKER_PLATFORM")
fi

if [ -n "$INSTANCE_IDS_FILE" ]; then
    CMD+=(--instance-ids-file "$INSTANCE_IDS_FILE")
fi

# shellcheck disable=SC2206
EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
CMD+=("${EXTRA_ARGS_ARRAY[@]}")

sg docker -c "$(printf '%q ' "${CMD[@]}")"

cd "$ORIG_DIR"

echo "Done! Results saved to: $SCRIPT_DIR/$WORKING_DIR"
