#!/bin/bash

export AGENT_NAME=${AGENT_NAME:-Kimi-K2.5-python}
export RUN_ID=${RUN_ID:-simple-agent-pro-Kimi-K2.5-py-interface-med-temp}
export WORKING_DIR=${WORKING_DIR:-evaluation/$RUN_ID}
export AGENT_CONFIG=${AGENT_CONFIG:-agent_configs/Kimi-K2.5-python-med-temp.yaml}

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pro_tool_Qwen3Coder-30B_py_interface_med_temp.sh" "$@"
