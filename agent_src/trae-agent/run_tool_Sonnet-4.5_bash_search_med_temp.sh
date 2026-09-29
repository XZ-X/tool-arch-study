#!/bin/bash

# get an optional argument for number of workers, default to 4
NUM_WORKERS=${1:-4}

source .venv/bin/activate

OUT_DIR=evaluation/trae-Sonnet-4.5-live-tool-bash-search-med-temp
WORKDING_DIR=trae-Sonnet-4.5-live-tool-bash-search-med-temp
CONFIG_FILE=../trae_config_tool_Sonnet-4.5_bash_search_med_temp.json
RUN_ID=Sonnet-4.5-live-tool-bash-search-med-temp


mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/trae-agent

# prepare the files for the agent

# the api keys file
cp example-api-keys.yaml $OUT_DIR/api_keys.yaml
cp mcp_url.yaml $OUT_DIR/mcp_url.yaml

# overwrite old files
rm -r $OUT_DIR/trae-agent/trae_agent
cp -r ../trae-agent/trae_agent $OUT_DIR/trae-agent
cp  ../trae-agent/pyproject.toml $OUT_DIR/trae-agent
cp  ../trae-agent/LICENSE $OUT_DIR/trae-agent

# setup the git repo to prevent some agent modifying its own code
cd $OUT_DIR/trae-agent && (
    rm -rf .git
    git init
    git add trae_agent
    git add pyproject.toml
    git add LICENSE
    git commit -m "Initial commit"
)

# now we are at evaluation/
cd ../../

python swebench_live.py --nproc $NUM_WORKERS --split vmini  --config-file $CONFIG_FILE   --run-id $RUN_ID --swebench-live-harness-path ./SWE-bench-Live --working-dir $WORKDING_DIR  --num_samples 10