# Tool-interface implementations

The study varies how information and actions are exposed to a coding agent. Five setups use the Trae-derived agent loop; the Python setup uses an adapted loop that accepts executable code. The implementations retain their upstream Trae Agent attribution.

## Interface overview

| Setup | What the actor uses | Main implementation |
|---|---|---|
| BashOnly | Shell commands and a completion tool | [Bash tool](agent_src/trae-agent/trae_agent/tools/bash_tool.py) |
| Atomic | Shell commands plus structured search, view, edit, insert, and create operations | [Atomic tool registration](agent_src/trae-agent/trae_agent/tools/__init__.py) |
| NLSearch | Shell commands plus a natural-language code-context query | [Search tool](agent_src/trae-agent/trae_agent/tools/subagent_search_tool.py) |
| Python | Executable Python blocks, followed by execution observations | [Python agent loop](agent_src/simple-agent/simple_agent/agent.py) |
| HypoTrack | Shell commands plus structured hypothesis tracking | [Hypothesis tool](agent_src/trae-agent/trae_agent/tools/hypotheses_tracking_tool.py) |
| ScratchPad | Shell commands plus an intermediate-notes tool | [Scratchpad tool](agent_src/trae-agent/trae_agent/tools/scratchpad_tool.py) |

## BashOnly: shell operations

The actor issues a `bash` tool call containing a command. The shell tool runs it in the task environment and returns output or an error. The actor can search, inspect files, edit code, and run tests through shell commands; `task_done` signals completion.

**Code:** [shell execution](agent_src/trae-agent/trae_agent/tools/bash_tool.py), [completion tool](agent_src/trae-agent/trae_agent/tools/task_done_tool.py), and [shared agent loop](agent_src/trae-agent/trae_agent/agent/base.py).

## Atomic: structured file operations

Atomic adds five tools to the BashOnly interface. Each operation has explicit arguments, such as a path, a search pattern, a line range, or replacement text. These tools perform file operations directly and return an observation to the actor. The shell remains available for other operations, including test execution.

| Operation | API tool name | Code |
|---|---|---|
| Search file contents | `search_tool` | [Search](agent_src/trae-agent/trae_agent/tools/atomic_search_tool.py) |
| View a file range | `view_tool` | [View](agent_src/trae-agent/trae_agent/tools/atomic_view_tool.py) |
| Replace text | `str_replace_based_edit_tool` | [Edit](agent_src/trae-agent/trae_agent/tools/atomic_edit_tool.py) |
| Insert text | `string_insert_tool` | [Insert](agent_src/trae-agent/trae_agent/tools/atomic_insert_tool.py) |
| Create a file | `file_create_tool` | [Create](agent_src/trae-agent/trae_agent/tools/atomic_create_tool.py) |

The [tool base classes](agent_src/trae-agent/trae_agent/tools/base.py) define the argument schemas and tool-result format. The [agent setup](agent_src/trae-agent/trae_agent/agent/trae_agent.py) enables these tools through `use_atomic_tool_common`.

## NLSearch: a natural-language search interface

The actor calls `find_code_context` with a natural-language query. The tool sends the query and task-container identifier to a search subagent over MCP. The subagent searches the repository with shell operations, collects relevant code spans, and returns context to the actor. This implementation uses repository exploration rather than a precomputed embedding index. The actor's shell interface remains available.

**Code:**

- [Actor-facing search tool](agent_src/trae-agent/trae_agent/tools/subagent_search_tool.py): query submission and returned context.
- [MCP search server](agent_src/trae-agent/analyze_tools/context_subagent/run_mcp.py): exposes `search_context`.
- [Search subagent](agent_src/trae-agent/analyze_tools/context_subagent/main_context_subagent.py): model configuration and search execution.
- [Search and context operations](agent_src/trae-agent/analyze_tools/context_subagent/context_subagent_tools.py): shell access and code-span collection.
- [Search instructions](agent_src/trae-agent/analyze_tools/context_subagent/prompts/context_subagent_sys_prompt.md): subagent prompt.

The actor enables this interface with `tool_find_code_context`. See [evaluation setup](agent_src/evaluation-setup.md#5-configure-the-agents-and-launch-an-experiment) for service configuration.

## Python: executable actions

The actor generates Python code inside `<python>...</python>` blocks. The agent extracts the blocks and executes them inside the task container, then returns standard output, standard error, and the exit status as an observation. A block can combine file inspection, searching, editing, and subprocess calls before returning its output. The task filesystem persists across actions; each execution starts a new Python process. `TASK_DONE` signals completion.

**Code:** [action parsing and the Python interaction loop](agent_src/simple-agent/simple_agent/agent.py), [container executor](agent_src/simple-agent/simple_agent/tools.py), and [Python system prompt](agent_src/simple-agent/simple_agent/prompts/system_prompt_python.txt).

The [per-model configurations](agent_src/simple-agent/agent_configs/) set `tool_interface: python`. Serving models and endpoints are configured in [models.yaml](agent_src/simple-agent/models.yaml).

## HypoTrack and ScratchPad: cognitive scaffolding

HypoTrack adds a structured `hypotheses_tracking` tool for recording and revising hypotheses, confidence, supporting or opposing evidence, and planned tests. The tool maintains the hypothesis records within the task. Its [implementation](agent_src/trae-agent/trae_agent/tools/hypotheses_tracking_tool.py) and [prompt addition](agent_src/trae-agent/trae_agent/agent/sys_prompt_hypo_tracking.txt) define the interface.

ScratchPad adds a `scratchpad` tool for intermediate notes and reminders. It stores note history within the task and returns an update acknowledgement. See the [tool](agent_src/trae-agent/trae_agent/tools/scratchpad_tool.py) and [prompt addition](agent_src/trae-agent/trae_agent/agent/sys_prompt_scratchpad.txt).

Both retain BashOnly's shell operations. They are enabled with `cog_use_hypotheses_tracking` and `cog_use_scratchpad`, respectively.

## Shared components and configuration

- [Trae agent setup](agent_src/trae-agent/trae_agent/agent/trae_agent.py) assembles each tool set and its prompts.
- [Tool registry](agent_src/trae-agent/trae_agent/tools/__init__.py) maps configured tools to implementations.
- [Configuration loader](agent_src/trae-agent/trae_agent/utils/config.py) reads the setup flags and model parameters.
- [Model client](agent_src/trae-agent/trae_agent/utils/openai_compatible_client.py) sends model requests and converts returned tool calls into agent actions.
- Trajectory recorders in the [Trae agent](agent_src/trae-agent/trae_agent/utils/trajectory_recorder.py) and [Python agent](agent_src/simple-agent/simple_agent/trajectory_recorder.py) capture interactions, tool results, and token usage.
- Benchmark runners under [trae-agent/evaluation/](agent_src/trae-agent/evaluation/) and [simple-agent/evaluation/](agent_src/simple-agent/evaluation/) prepare task containers, collect patches, and invoke evaluation harnesses.

Trae configuration filenames use `bashonly`, `atomic_common`, `bash_search`, `bash_hypo`, and `bash_scratchpad` for the five setups. The Python configurations are under `simple-agent/agent_configs/`. See [evaluation setup](agent_src/evaluation-setup.md) to run new experiments and [Reproduce the paper results](reproduce-paper-results.md) to regenerate the released tables and figures from processed data.
