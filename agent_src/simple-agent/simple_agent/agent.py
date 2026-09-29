from __future__ import annotations

import json
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .tools import DockerBashTool, DockerPythonExecutor, TaskDoneTool, Tool
from .trajectory_recorder import ToolCallRecord, ToolResultRecord, TrajectoryRecorder

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a system prompt from the prompts directory."""
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


_TASK_PROFILE_PROMPTS = {
    "default": "",
    "swebench_pro": """
## SWE-bench Pro Task Profile

This task comes from SWE-bench Pro. Treat it as a professional software
engineering change request, not necessarily as a narrow bug report. The task may
ask for a bug fix, feature addition, API/interface addition, behavior change,
optimization, security improvement, UI/UX change, or backend change.

When this profile narrows or clarifies earlier generic issue-fixing wording,
follow this SWE-bench Pro profile for the current task.

Use the problem statement, Requirements section, and Interface section together
as the acceptance criteria. Requirements and interface details are authoritative
when they clarify or constrain the requested behavior, but they should not be
treated as prescribing one exact implementation unless they explicitly say so.

Your goal is to produce a minimal, coherent patch to the target repository that
implements the requested change and preserves existing behavior. Verify with
focused tests or scripts when feasible.
""".strip(),
}


_PYTHON_INTERFACE_RUNTIME_PROMPT = """
## Python Execution Semantics

Each <python> block is executed as a fresh Python process. Python interpreter
state does not persist across turns: variables, imports, function definitions,
open files, and in-memory objects from earlier <python> blocks are gone in the
next block.

Filesystem changes do persist. Files you create or edit remain in the
repository. If you need a module, variable, helper function, or subprocess call
in a later step, import or define it again in that later <python> block.

Before declaring completion, run a fresh verification command that imports or
compiles the changed code from disk. Do not call <TASK_DONE> if syntax errors,
import errors, failed tests, or failed verification remain.
""".strip()


TEST_PATCH_PATTERNS = ["/test/", "/tests/", "/testing/", "test_", "tox.ini"]


@dataclass
class AgentResult:
    diff: str
    trajectory: dict[str, Any]
    status: str  # "success" | "fail" | "timeout"


class CodingAgent:
    """A coding agent that runs tools inside a Docker container.

    Args:
        container: A Docker container object (from the docker SDK).
        model: OpenAI-compatible model name.
        base_url: OpenAI-compatible API base URL.
        provider: Provider label for trajectory logging.
        max_steps: Maximum agent loop iterations.
        command_timeout_sec: Timeout per docker exec command.
        extra_tools: Additional Tool instances to register alongside bash.
        tool_interface: "tool_call" for standard tool calling, or "python"
            for free-form Python code blocks.
        temperature: Sampling temperature for the LLM. None uses the model default.
        max_tokens: Maximum number of tokens the LLM may generate per call.
    """

    def __init__(
        self,
        container,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        provider: str = "openai",
        max_steps: int = 200,
        command_timeout_sec: int = 120,
        extra_tools: list[Tool] | None = None,
        tool_interface: str = "tool_call",
        temperature: float | None = None,
        max_tokens: int = 4096,
        repo_path: str = "/testbed",
        task_profile: str = "default",
    ):
        if tool_interface not in ("tool_call", "python"):
            raise ValueError(
                f"tool_interface must be 'tool_call' or 'python', got '{tool_interface}'"
            )
        if task_profile not in _TASK_PROFILE_PROMPTS:
            raise ValueError(
                f"task_profile must be one of {sorted(_TASK_PROFILE_PROMPTS)}, got {task_profile!r}"
            )
        self.container = container
        self.container_name: str = container.name
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.max_steps = max_steps
        self.command_timeout_sec = command_timeout_sec
        self.tool_interface = tool_interface
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repo_path = repo_path
        self.task_profile = task_profile

        # Load the appropriate system prompt
        if self.tool_interface == "python":
            self.system_prompt = _load_prompt("system_prompt_python.txt")
            self.system_prompt = (
                f"{self.system_prompt}\n\n{_PYTHON_INTERFACE_RUNTIME_PROMPT}"
            )
        else:
            self.system_prompt = _load_prompt("system_prompt_tool_call.txt")
        profile_prompt = _TASK_PROFILE_PROMPTS[self.task_profile]
        if profile_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{profile_prompt}"

        # Build tool registry: name -> tool (used in tool_call mode)
        bash_tool = DockerBashTool(
            container_name=self.container_name,
            timeout_sec=self.command_timeout_sec,
        )
        task_done_tool = TaskDoneTool()
        self.tools: dict[str, Tool] = {
            bash_tool.name: bash_tool,
            task_done_tool.name: task_done_tool,
        }
        if extra_tools:
            for t in extra_tools:
                self.tools[t.name] = t

        # Python executor (used in python mode)
        self.python_executor = DockerPythonExecutor(
            container_name=self.container_name,
            timeout_sec=self.command_timeout_sec,
            repo_path=self.repo_path,
        )

    # ------------------------------------------------------------------
    # Parsing helpers for python mode
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_python_block(block: str) -> str:
        """Normalize code extracted from tolerant Python-interface markup."""
        code = block.strip()
        code = re.sub(r"^\s*<python(?:\s[^>]*)?>\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*</python>\s*$", "", code, flags=re.IGNORECASE)
        code = re.sub(r"^\s*```\s*(?:python|py)?\s*\n?", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\n?\s*```\s*$", "", code)
        return code.strip()

    @classmethod
    def _parse_python_blocks(cls, text: str) -> list[str]:
        """Extract Python code blocks from XML tags or common Markdown fences.

        The Python-interface prompt asks for <python>...</python>, but some
        models drift into ```python fences or mixed delimiters. Treat those as
        executable blocks instead of spending an extra turn on a format repair.
        """
        patterns = [
            re.compile(
                r"<python(?:\s[^>]*)?>(.*?)(?:</python>|```|\Z)",
                re.DOTALL | re.IGNORECASE,
            ),
            re.compile(
                r"```\s*(?:python|py)\s*\n(.*?)(?:```|</python>|\Z)",
                re.DOTALL | re.IGNORECASE,
            ),
        ]
        matches: list[tuple[int, int, str]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), match.group(1)))

        blocks: list[str] = []
        seen_blocks: set[str] = set()
        used_spans: list[tuple[int, int]] = []
        for start, end, raw_block in sorted(matches, key=lambda item: item[0]):
            # Avoid executing the same mixed-delimiter block twice when both
            # patterns match overlapping text.
            if any(start < used_end and end > used_start for used_start, used_end in used_spans):
                continue
            code = cls._clean_python_block(raw_block)
            if code and code not in seen_blocks:
                blocks.append(code)
                seen_blocks.add(code)
                used_spans.append((start, end))
        return blocks

    @staticmethod
    def _check_task_done(text: str) -> bool:
        """Check if the text contains the <TASK_DONE> signal."""
        return "<TASK_DONE>" in text

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, task: str, trajectory_path: str | None = None) -> AgentResult:
        """Run the agent on a task and return the result.

        Args:
            task: The problem statement / task prompt.
            trajectory_path: Full file path to save the trajectory JSON.
                             If None, trajectory is not saved to disk.

        Returns:
            AgentResult with diff, trajectory dict, and status.
        """
        if "[Project root path]:" not in task:
            task = f"{task}\n\n[Project root path]: {self.repo_path}"

        recorder = TrajectoryRecorder(trajectory_path)
        recorder.start_recording(
            task=task,
            provider=self.provider,
            model=self.model,
            max_steps=self.max_steps,
        )

        client_kwargs: dict[str, Any] = {"base_url": self.base_url}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        client = OpenAI(**client_kwargs)

        if self.tool_interface == "python":
            return self._run_python_mode(client, task, recorder)
        else:
            return self._run_tool_call_mode(client, task, recorder)

    # ------------------------------------------------------------------
    # tool_call mode (existing logic)
    # ------------------------------------------------------------------

    def _run_tool_call_mode(
        self, client: OpenAI, task: str, recorder: TrajectoryRecorder,
    ) -> AgentResult:
        tool_schemas = [t.schema for t in self.tools.values()]
        tool_names = list(self.tools.keys())

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        step_number = 0
        final_output: str | None = None
        status = "success"

        try:
            for _ in range(self.max_steps):
                step_number += 1

                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tool_schemas,
                    "tool_choice": "auto",
                    "max_tokens": self.max_tokens,
                }
                if self.temperature is not None:
                    create_kwargs["temperature"] = self.temperature
                response = client.chat.completions.create(**create_kwargs)
                choice = response.choices[0]
                assistant = choice.message

                # Only take the first tool call if multiple are returned.
                first_tc = assistant.tool_calls[0] if assistant.tool_calls else None

                serialized_tool_call: ToolCallRecord | None = None
                assistant_tool_call_for_msg: dict[str, Any] | None = None
                if first_tc:
                    serialized_tool_call = ToolCallRecord(
                        call_id=first_tc.id,
                        name=first_tc.function.name,
                        arguments=first_tc.function.arguments,
                        id=first_tc.id,
                    )
                    assistant_tool_call_for_msg = {
                        "id": first_tc.id,
                        "type": "function",
                        "function": {
                            "name": first_tc.function.name,
                            "arguments": first_tc.function.arguments,
                        },
                    }

                # Parse arguments from JSON string to dict for downstream tools.
                parsed_args: dict | str | None = None
                if serialized_tool_call:
                    try:
                        parsed_args = json.loads(serialized_tool_call.arguments)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = serialized_tool_call.arguments

                llm_response = {
                    "content": assistant.content,
                    "model": response.model,
                    "finish_reason": choice.finish_reason,
                    "usage": {
                        "input_tokens": response.usage.prompt_tokens if response.usage else None,
                        "output_tokens": response.usage.completion_tokens if response.usage else None,
                    } if response.usage else None,
                    "tool_calls": [
                        {"call_id": serialized_tool_call.call_id, "name": serialized_tool_call.name,
                         "arguments": parsed_args, "id": serialized_tool_call.id}
                    ] if serialized_tool_call else None,
                }

                recorder.record_llm_interaction(
                    messages=list(messages),
                    context=list(messages),
                    response_content=assistant.content or "",
                    provider=self.provider,
                    model=response.model,
                    tool_calls=[serialized_tool_call] if serialized_tool_call else None,
                    usage={
                        "input_tokens": response.usage.prompt_tokens if response.usage else None,
                        "output_tokens": response.usage.completion_tokens if response.usage else None,
                    },
                    finish_reason=choice.finish_reason,
                    tools_available=tool_names,
                )

                # Build assistant message for conversation history.
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant.content or "",
                }
                if assistant_tool_call_for_msg:
                    assistant_message["tool_calls"] = [assistant_tool_call_for_msg]
                messages.append(assistant_message)

                # No tool call => agent is done.
                if not first_tc:
                    recorder.record_agent_step(
                        step_number=step_number,
                        state="llm_response",
                        llm_messages=list(messages),
                        llm_response=llm_response,
                        tool_calls=None,
                    )
                    final_output = assistant.content or ""
                    break

                # Execute the single tool call.
                tool_content = self._execute_tool_call(first_tc)

                is_error = tool_content.startswith("Error:")
                tool_result_record = ToolResultRecord(
                    call_id=first_tc.id,
                    success=not is_error,
                    result=tool_content if not is_error else None,
                    error=tool_content if is_error else None,
                    id=first_tc.id,
                )

                # Record a single agent step per LLM call: response + tool execution together.
                recorder.record_agent_step(
                    step_number=step_number,
                    state="llm_response",
                    llm_messages=list(messages),
                    llm_response=llm_response,
                    tool_calls=[serialized_tool_call],
                    tool_results=[tool_result_record],
                    error=tool_content if is_error else None,
                )

                # task_done => agent finished successfully.
                if first_tc.function.name == "task_done":
                    final_output = assistant.content or ""
                    break

                messages.append({
                    "role": "tool",
                    "tool_call_id": first_tc.id,
                    "content": tool_content,
                })

            if final_output is None:
                final_output = "Task execution exceeded maximum steps without completion."
                status = "timeout"

            recorder.finalize_recording(success=True, final_result=final_output)

        except Exception as exc:
            err = str(exc)
            status = "fail"
            final_output = err
            synthetic_call_id = f"call_{uuid.uuid4().hex[:8]}"
            recorder.record_agent_step(
                step_number=step_number + 1,
                state="failed",
                llm_messages=[{"role": "user", "content": task}],
                tool_calls=[ToolCallRecord(call_id=synthetic_call_id, name="error", arguments="")],
                tool_results=[ToolResultRecord(
                    call_id=synthetic_call_id,
                    success=False,
                    result=None,
                    error=err,
                )],
                error=err,
            )
            recorder.finalize_recording(success=False, final_result=err)

        diff = self._get_diff()

        return AgentResult(
            diff=diff,
            trajectory=recorder.trajectory_data,
            status=status,
        )

    # ------------------------------------------------------------------
    # python mode (new)
    # ------------------------------------------------------------------

    # Few-shot example messages to steer models toward using <python> tags.
    # Some models (e.g. kimi) ignore system prompt instructions and fall back
    # to their native tool-call format; a concrete example reliably fixes this.
    _PYTHON_FEW_SHOT = [
        {
            "role": "user",
            "content": "Fix the bug.\n[Project root path]: .",
        },
        {
            "role": "assistant",
            "content": (
                "Let me explore the codebase first.\n\n"
                "<python>\nimport os\nprint(os.listdir('.'))\n</python>"
            ),
        },
        {
            "role": "user",
            "content": (
                "[Python execution result]\n"
                "stdout:\nsetup.py\nsrc\ntests\n\nexit_code: 0"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Let me look at the source files.\n\n"
                "<python>\nimport os\nfor f in os.listdir('src'):\n    print(f)\n</python>"
            ),
        },
        {
            "role": "user",
            "content": (
                "[Python execution result]\n"
                "stdout:\nmain.py\nutils.py\n\nexit_code: 0"
            ),
        },
    ]

    def _run_python_mode(
        self, client: OpenAI, task: str, recorder: TrajectoryRecorder,
    ) -> AgentResult:
        """Run agent loop using <python>...</python> code blocks instead of tool calls."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *self._PYTHON_FEW_SHOT,
            {"role": "user", "content": task},
        ]

        step_number = 0
        final_output: str | None = None
        status = "success"

        try:
            for _ in range(self.max_steps):
                step_number += 1

                # No tools/tool_choice — model generates free-form text.
                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                }
                if self.temperature is not None:
                    create_kwargs["temperature"] = self.temperature
                response = client.chat.completions.create(**create_kwargs)
                choice = response.choices[0]
                assistant_content = choice.message.content or ""

                llm_response = {
                    "content": assistant_content,
                    "model": response.model,
                    "finish_reason": choice.finish_reason,
                    "usage": {
                        "input_tokens": response.usage.prompt_tokens if response.usage else None,
                        "output_tokens": response.usage.completion_tokens if response.usage else None,
                    } if response.usage else None,
                    "tool_calls": None,
                }

                recorder.record_llm_interaction(
                    messages=list(messages),
                    context=list(messages),
                    response_content=assistant_content,
                    provider=self.provider,
                    model=response.model,
                    tool_calls=None,
                    usage={
                        "input_tokens": response.usage.prompt_tokens if response.usage else None,
                        "output_tokens": response.usage.completion_tokens if response.usage else None,
                    },
                    finish_reason=choice.finish_reason,
                    tools_available=["python"],
                )

                # Append assistant message (content only, no tool_calls).
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # Parse <python>...</python> blocks first.
                python_blocks = self._parse_python_blocks(assistant_content)
                has_task_done = self._check_task_done(assistant_content)

                # If no python blocks and no TASK_DONE — model is just reasoning.
                if not python_blocks and not has_task_done:
                    recorder.record_agent_step(
                        step_number=step_number,
                        state="llm_response",
                        llm_messages=list(messages),
                        llm_response=llm_response,
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "Please provide your next action as Python code wrapped in "
                            "<python>...</python> tags, or output <TASK_DONE> if you are finished."
                        ),
                    })
                    continue

                # If only TASK_DONE without any python blocks, finish immediately.
                if not python_blocks and has_task_done:
                    recorder.record_agent_step(
                        step_number=step_number,
                        state="task_done",
                        llm_messages=list(messages),
                        llm_response=llm_response,
                    )
                    final_output = assistant_content
                    break

                # Execute each python block and collect results.
                all_tool_calls: list[ToolCallRecord] = []
                all_tool_results: list[ToolResultRecord] = []
                execution_outputs: list[str] = []

                for i, code_block in enumerate(python_blocks):
                    call_id = f"python_{step_number}_{i}"

                    tc_record = ToolCallRecord(
                        call_id=call_id,
                        name="python",
                        arguments=json.dumps({"code": code_block}),
                        id=call_id,
                    )
                    all_tool_calls.append(tc_record)

                    success, output = self.python_executor.execute_code(code_block)

                    tr_record = ToolResultRecord(
                        call_id=call_id,
                        success=success,
                        result=output if success else None,
                        error=output if not success else None,
                        id=call_id,
                    )
                    all_tool_results.append(tr_record)
                    execution_outputs.append(output)

                # Record agent step with all python executions.
                recorder.record_agent_step(
                    step_number=step_number,
                    state="python_execution",
                    llm_messages=list(messages),
                    llm_response=llm_response,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                )

                # After executing python blocks, check if task is done.
                if has_task_done:
                    # Python blocks were executed, now we're done.
                    final_output = assistant_content
                    break

                # Build execution result message for conversation history.
                if len(execution_outputs) == 1:
                    result_content = f"[Python execution result]\n{execution_outputs[0]}"
                else:
                    parts = []
                    for i, output in enumerate(execution_outputs):
                        parts.append(f"[Python block {i + 1} result]\n{output}")
                    result_content = "\n\n".join(parts)

                messages.append({
                    "role": "user",
                    "content": result_content,
                })

            if final_output is None:
                final_output = "Task execution exceeded maximum steps without completion."
                status = "timeout"

            recorder.finalize_recording(success=True, final_result=final_output)

        except Exception as exc:
            err = str(exc)
            status = "fail"
            final_output = err
            synthetic_call_id = f"call_{uuid.uuid4().hex[:8]}"
            recorder.record_agent_step(
                step_number=step_number + 1,
                state="failed",
                llm_messages=[{"role": "user", "content": task}],
                tool_calls=[ToolCallRecord(call_id=synthetic_call_id, name="error", arguments="")],
                tool_results=[ToolResultRecord(
                    call_id=synthetic_call_id,
                    success=False,
                    result=None,
                    error=err,
                )],
                error=err,
            )
            recorder.finalize_recording(success=False, final_result=err)

        diff = self._get_diff()

        return AgentResult(
            diff=diff,
            trajectory=recorder.trajectory_data,
            status=status,
        )

    def _execute_tool_call(self, tc) -> str:
        """Execute a single tool call, returning the content string for the tool message.

        Handles tool-not-found, JSON parse errors, and tool execution errors
        by returning an error message that gets sent back to the model.
        """
        tool_name = tc.function.name
        raw_args = tc.function.arguments or ""

        # Unknown tool
        if tool_name not in self.tools:
            return (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(self.tools.keys())}"
            )

        # Parse arguments
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in tool arguments: {e}"

        if not isinstance(arguments, dict):
            return f"Error: Tool arguments must be a JSON object, got {type(arguments).__name__}"

        # Execute
        try:
            success, result_text, error_text = self.tools[tool_name].execute(arguments)
        except Exception as e:
            return f"Error: Tool execution failed: {e}"

        if not success:
            return f"Error: {error_text}"

        return result_text

    def _get_diff(self) -> str:
        """Get git diff from the container, filtering out test-file changes."""
        command = f"cd {shlex.quote(self.repo_path)} && git --no-pager diff"
        completed = subprocess.run(
            ["docker", "exec", self.container_name, "/bin/sh", "-lc", command],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if completed.returncode != 0:
            return ""
        diff_text = completed.stdout
        if not diff_text:
            return ""
        return self._remove_test_patches(diff_text)

    @staticmethod
    def _extract_stdout(tool_output: str) -> str:
        """Extract the stdout content from DockerBashTool output."""
        lines = tool_output.split("\n")
        result_lines: list[str] = []
        in_stdout = False
        for line in lines:
            if line == "stdout:":
                in_stdout = True
                continue
            if line.startswith("stderr:") or line.startswith("exit_code:"):
                in_stdout = False
                continue
            if in_stdout:
                result_lines.append(line)
        return "\n".join(result_lines)

    @staticmethod
    def _remove_test_patches(patch: str) -> str:
        """Remove hunks that touch test directories/files from a unified diff.

        Matches trae-agent's remove_patches_to_tests logic.
        """
        lines = patch.splitlines(keepends=True)
        filtered: list[str] = []
        is_test = False

        for line in lines:
            if line.startswith("diff --git a/"):
                target_path = line.split()[-1]
                is_test = target_path.startswith("b/") and any(
                    p in target_path for p in TEST_PATCH_PATTERNS
                )
            if not is_test:
                filtered.append(line)

        return "".join(filtered)
