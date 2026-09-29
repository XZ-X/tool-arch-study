#!/bin/bash

export AGENT_NAME=${AGENT_NAME:-Sonnet-4.5-python}
export RUN_ID=${RUN_ID:-simple-agent-pro-Sonnet-4.5-py-interface-med-temp}
export WORKING_DIR=${WORKING_DIR:-evaluation/$RUN_ID}
export AGENT_CONFIG=${AGENT_CONFIG:-agent_configs/Sonnet-4.5-python-med-temp.yaml}

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pro_tool_Qwen3Coder-30B_py_interface_med_temp.sh" "$@"
