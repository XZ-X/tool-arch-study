from __future__ import annotations

import subprocess
import shlex
from typing import Any, Protocol


_MAX_OUTPUT_CHARS = 16_384  # 16 KB per stream (stdout / stderr)


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate *text* to *limit* characters, appending a notice if trimmed."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [truncated {len(text) - limit} characters]"


BASH_TOOL_DESCRIPTION = (
    "Run commands in a bash shell\n"
    '* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.\n'
    "* You have access to a mirror of common linux and python packages via apt and pip.\n"
    "* State is persistent across command calls and discussions with the user.\n"
    "* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n"
    "* Please avoid commands that may produce a very large amount of output.\n"
    "* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.\n"
)


class Tool(Protocol):
    """Protocol that all tools must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> dict: ...

    def execute(self, arguments: dict[str, Any]) -> tuple[bool, str, str | None]:
        """Execute the tool. Returns (success, result_text, error_text)."""
        ...


class DockerBashTool:
    """Executes shell commands inside a Docker container."""

    def __init__(self, container_name: str, timeout_sec: int = 120):
        self.container_name = container_name
        self.timeout_sec = timeout_sec

    @property
    def name(self) -> str:
        return "bash"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "bash",
                "description": BASH_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to run.",
                        },
                        "restart": {
                            "type": "boolean",
                            "description": "Set to true to restart the bash session.",
                        },
                    },
                    "required": ["command", "restart"],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> tuple[bool, str, str | None]:
        command = arguments.get("command")
        if not command:
            return False, "", "Missing required argument: 'command'"
        result_text = self._run(str(command))
        return True, result_text, None

    def _run(self, command: str) -> str:
        cmd = ["docker", "exec", self.container_name, "/bin/sh", "-lc", command]
        try:
            completed = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_sec,
            )
            output = []
            if completed.stdout:
                output.append(f"stdout:\n{_truncate(completed.stdout)}")
            if completed.stderr:
                output.append(f"stderr:\n{_truncate(completed.stderr)}")
            output.append(f"exit_code: {completed.returncode}")
            return "\n".join(output)
        except subprocess.TimeoutExpired:
            return f"stderr:\ncommand timed out after {self.timeout_sec}s\nexit_code: 124"


class TaskDoneTool:
    """Tool the model calls to signal it has finished the task."""

    @property
    def name(self) -> str:
        return "task_done"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "task_done",
                "description": (
                    "Report the completion of the task. "
                    "You cannot call this tool before any verification is done. "
                    "Write a reproduce / test script to verify your solution first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, arguments: dict[str, Any]) -> tuple[bool, str, str | None]:
        return True, "Task done.", None


class DockerPythonExecutor:
    """Executes Python code inside a Docker container.

    Writes code to a temp file via stdin piping, executes it, and cleans up.
    This is a utility class for the "python" tool interface mode — it is NOT
    a Tool and is not registered with the tool-calling API.
    """

    TEMP_SCRIPT_PATH = "/tmp/_agent_code.py"

    def __init__(
        self,
        container_name: str,
        timeout_sec: int = 120,
        repo_path: str = "/testbed",
    ):
        self.container_name = container_name
        self.timeout_sec = timeout_sec
        self.repo_path = repo_path

    def execute_code(self, code: str) -> tuple[bool, str]:
        """Execute Python code in the container.

        Returns (success, output_text) where output_text contains
        stdout, stderr, and exit code in the same format as DockerBashTool.
        """
        shell_cmd = (
            f"cat > {self.TEMP_SCRIPT_PATH} && "
            f"cd {shlex.quote(self.repo_path)} && python3 {self.TEMP_SCRIPT_PATH}; "
            f"_exit=$?; rm -f {self.TEMP_SCRIPT_PATH}; exit $_exit"
        )
        cmd = [
            "docker", "exec", "-i", self.container_name,
            "bash", "-c", shell_cmd,
        ]
        try:
            completed = subprocess.run(
                cmd,
                input=code,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_sec,
            )
            output = []
            if completed.stdout:
                output.append(f"stdout:\n{_truncate(completed.stdout)}")
            if completed.stderr:
                output.append(f"stderr:\n{_truncate(completed.stderr)}")
            output.append(f"exit_code: {completed.returncode}")
            return (completed.returncode == 0), "\n".join(output)
        except subprocess.TimeoutExpired:
            # Clean up temp file on timeout
            cleanup_cmd = [
                "docker", "exec", self.container_name,
                "rm", "-f", self.TEMP_SCRIPT_PATH,
            ]
            try:
                subprocess.run(cleanup_cmd, capture_output=True, timeout=5)
            except Exception:
                pass
            return False, f"stderr:\nPython execution timed out after {self.timeout_sec}s\nexit_code: 124"
