# Evaluation setup

Run the commands below from `agent_src/` in the release repository.

The experiment runners generate agent patches; the benchmark harnesses run the tests and produce evaluation reports. Set up the agent environment and the harness for each benchmark you want to run.

## 1. Host and agent environments

Use Linux with Docker, Git, and `uv`. Install Docker following the [official installation guide](https://docs.docker.com/engine/install/), and enable access for your user following the [Linux post-installation guide](https://docs.docker.com/engine/install/linux-postinstall/). The launchers use the `docker` group. Confirm access with `docker info` and allow disk space for task images and build artifacts.

Run these commands from this folder:

```bash
cd trae-agent
uv sync --python 3.12 --extra evaluation
cd ../simple-agent
uv sync --python 3.12 --extra evaluation
cd ..
```

The agents use Python 3.12. Keep each benchmark harness in its own environment: SWE-bench and SWE-bench Live both install a package named `swebench`.

## 2. SWE-bench / SWE-bench Verified

The existing setup script clones [SWE-bench](https://github.com/SWE-bench/SWE-bench), checks out commit `2bf15e1be3c995a0758529bd29848a8987546090`, and installs it in `SWE-bench/swebench_venv`.

From this folder:

```bash
cd trae-agent
source .venv/bin/activate
cd evaluation
bash swebench_setup.sh
cd ../..
```

Run the setup script once in a fresh evaluation directory. Check the harness installation without invoking an agent:

```bash
cd trae-agent/evaluation/SWE-bench
swebench_venv/bin/python -m swebench.harness.run_evaluation --help
cd ../../..
```

To exercise Docker and the test harness with one reference patch:

```bash
cd trae-agent/evaluation/SWE-bench
swebench_venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --instance_ids sympy__sympy-20590 \
  --predictions_path gold --max_workers 1 --run_id setup-verified-gold
cd ../../..
```

The study runners use this checkout through `--swebench-harness-path ./SWE-bench`, with the working directory set to the agent's `evaluation/` directory.

## 3. SWE-bench Live

Use the upstream [Python-only branch](https://github.com/microsoft/SWE-bench-Live/tree/python-only), which provides the `swebench.harness.run_evaluation` interface used by the runners. The upstream [setup instructions](https://github.com/microsoft/SWE-bench-Live/blob/python-only/README.md#-set-up) describe this harness.

From this folder:

```bash
git clone --branch python-only https://github.com/microsoft/SWE-bench-Live.git trae-agent/evaluation/SWE-bench-Live
uv venv --python 3.12 trae-agent/evaluation/SWE-bench-Live/swe_live_venv
uv pip install --python trae-agent/evaluation/SWE-bench-Live/swe_live_venv/bin/python -e trae-agent/evaluation/SWE-bench-Live
```

The environment name `swe_live_venv` is required by the runners. Check the installation:

```bash
cd trae-agent/evaluation/SWE-bench-Live
swe_live_venv/bin/python3 -m swebench.harness.run_evaluation --help
cd ../../..
```

To exercise Docker and the test harness with the upstream reference-patch example:

```bash
cd trae-agent/evaluation/SWE-bench-Live
swe_live_venv/bin/python3 -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench-Live/SWE-bench-Live --split lite \
  --instance_ids amoffat__sh-744 --namespace starryzhang \
  --predictions_path gold --max_workers 1 --run_id setup-live-gold
cd ../../..
```

The main-study launchers use `--split vmini`; the runner loads task IDs from the bundled `swebench_live_verified_mini.json` and evaluates them against the benchmark's `verified` split. Keep the bundled task files in place. Datasets and task Docker images are downloaded by the runners as needed.

## 4. Share the harness checkouts with the Python agent

The Python-agent launchers also expect `SWE-bench` and `SWE-bench-Live` inside their own `evaluation/` directory. On a fresh setup, link the checkouts prepared above:

```bash
cd simple-agent/evaluation
ln -s ../../trae-agent/evaluation/SWE-bench SWE-bench
ln -s ../../trae-agent/evaluation/SWE-bench-Live SWE-bench-Live
cd ../..
```

The resulting paths are:

| Benchmark | Harness path from either agent's `evaluation/` | Harness interpreter |
|---|---|---|
| SWE-bench Verified | `./SWE-bench` | `swebench_venv/bin/python` inside the checkout |
| SWE-bench Live | `./SWE-bench-Live` | `swe_live_venv/bin/python3` inside the checkout |

## 5. Configure the agents and launch an experiment

Set the model serving IDs and API endpoints in `trae-agent/trae_config_tool_*.json` and `simple-agent/models.yaml`. Their shipped endpoint and model values are configuration placeholders.

For the Trae launchers, set the `openai-compatible` credential in the local `trae-agent/example-api-keys.yaml`; the launchers copy this file into the experiment workspace as `api_keys.yaml`. For the Python agent, copy `simple-agent/api_keys.example.yaml` to `simple-agent/api_keys.yaml` and fill in the same credential reference. Keep actual credentials out of the shared code. If using `${OPENAI_API_KEY}`, the variable must be available in the process that runs the agent; Trae agents run inside task containers.

Trae also requires `trae-agent/mcp_url.yaml`, based on its template, with `url` pointing to an available context-history MCP service. That service is a separate prerequisite and is not bundled here. For NLSearch, set `ctx_tool` to the search service and start the search server in another terminal:

```bash
cd trae-agent
source .venv/bin/activate
python analyze_tools/context_subagent/run_mcp.py --model Qwen3Coder-30B --port 14389
```

The search server uses `trae-agent/models.yaml` and `api_keys.yaml`; copy `example-api-keys.yaml` to `api_keys.yaml` locally for this server. Configure its serving model and endpoint as well. MCP URLs must be reachable from the task containers; replace the template's loopback URLs with reachable service addresses.

From this folder, run a main-study example with four workers:

```bash
cd trae-agent
source .venv/bin/activate
bash run_tool_Qwen3Coder-30B_bashonly_med_temp.sh 4
cd ..
```

Or run the Python interface:

```bash
cd simple-agent
source .venv/bin/activate
bash run_tool_Qwen3Coder-30B_py_interface_med_temp.sh 4
cd ..
```

Use the corresponding `run_verified_tool_*.sh` launcher for SWE-bench Verified. Generated patches, trajectories, and predictions go under the launcher's `evaluation/` working directory; harness test logs and reports go under the harness checkout.

## 6. SWE-bench Pro (additional experiments)

The Pro runners use the legacy `swe_bench_pro_eval.py` and `run_scripts/` interface in the [upstream repository](https://github.com/scaleapi/SWE-bench_Pro-os). Prepare it from this folder:

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git trae-agent/evaluation/SWE-bench_Pro-os
uv venv --python 3.12 trae-agent/evaluation/swebench-pro-venv
uv pip install --python trae-agent/evaluation/swebench-pro-venv/bin/python -r trae-agent/evaluation/SWE-bench_Pro-os/requirements.txt
```

The Python-agent Pro launchers already point to this shared checkout and environment. Their default `ptiny` tasks come from the bundled task metadata. They use local Docker and the upstream `jefzda/sweap-images` images; Modal setup is not required for this path. The Trae Pro runner evaluates with its active Python interpreter, so install the harness requirements into that agent environment if invoking it:

```bash
uv pip install --python trae-agent/.venv/bin/python -r trae-agent/evaluation/SWE-bench_Pro-os/requirements.txt
```

Use the [upstream v1 evaluation instructions](https://github.com/scaleapi/SWE-bench_Pro-os#installation) for the legacy harness. The runners target its v1 task format.
