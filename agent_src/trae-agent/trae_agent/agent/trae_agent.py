# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""TraeAgent for software engineering tasks."""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import override

from ..tools import tools_registry
from ..tools.base import Tool, ToolExecutor, ToolResult
from ..utils.config import Config
from ..utils.llm_basics import LLMMessage, LLMResponse
from .agent_basics import AgentError, AgentExecution
from .base import Agent

TraeAgentToolNames = ["task_done", "bash"]

TEST_PATCH_DIR_NAMES = ("test", "tests", "testing")
TEST_PATCH_FILE_PREFIXES = ("test_",)
TEST_PATCH_FILE_SUFFIXES = ("_test.py", "_tests.py")
TEST_PATCH_CONFIG_FILES = ("tox.ini",)

class TraeAgent(Agent):
    """Trae Agent specialized for software engineering tasks."""

    def __init__(self, config: Config):
        self.agent_ablation_config = config.agent_ablation_config
        self.sweb_live = config.sweb_live
        self.project_path: str = ""
        self.base_commit: str | None = None
        self.must_patch: str = "false"
        self.patch_path: str | None = None
        super().__init__(config)

    def setup_trajectory_recording(self, trajectory_path: str | None = None) -> str:
        """Set up trajectory recording for this agent.

        Args:
            trajectory_path: Path to save trajectory file. If None, generates default path.

        Returns:
            The path where trajectory will be saved.
        """
        from ..utils.trajectory_recorder import TrajectoryRecorder

        recorder = TrajectoryRecorder(trajectory_path)
        self.set_trajectory_recorder(recorder)

        # Start recording with task info
        if hasattr(self, "task") and self.task:
            recorder.start_recording(
                task=self.task,
                provider=self.llm_client.provider.value,
                model=self.model_parameters.model,
                max_steps=self.max_steps,
            )

        return recorder.get_trajectory_path()

    @override
    def new_task(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        """Create a new task."""
        self.task: str = task
        if not extra_args:
            raise AgentError("Project path and issue information are required.")

        if tool_names is None:
            tool_names = TraeAgentToolNames

        # Get the model provider from the LLM client
        provider = self.llm_client.provider.value
        self.llm_client.instantiate_context_manager(
            task_id=extra_args.get("problem_instance", "default_task")
        )

        self.tools: list[Tool] = [
            tools_registry[tool_name](model_provider=provider)
            for tool_name in tool_names
            if "bash" not in tool_name
        ]

        self.tools.append(
            tools_registry["bash"](model_provider=provider, is_live=self.sweb_live)
        )
        if self.agent_ablation_config.tool_bash_only:
            self.tools: list[Tool] = [
                tools_registry["bash"](model_provider=provider, is_live=self.sweb_live),
                tools_registry["task_done"](model_provider=provider),
            ]

        if self.agent_ablation_config.tool_find_code_context:
            self.tools.append(
                tools_registry["find_code_context"](
                    model_provider=provider,
                    is_live=self.sweb_live,
                )
            )
        
        if self.agent_ablation_config.use_atomic_tool_common:
            tool_names = [
                'atomic_edit',
                'atomic_search',
                'atomic_view',
                'atomic_insert',
                'atomic_create'
            ]
            self.tools.extend([
                tools_registry[tool_name](model_provider=provider, is_live=self.sweb_live)
                for tool_name in tool_names
            ])
        if self.agent_ablation_config.cog_use_hypotheses_tracking:
            tool_names += ["hypotheses_tracking"]
            self.tools.append(
                tools_registry["hypotheses_tracking"](
                    model_provider=provider,
                    is_live=self.sweb_live,
                )
            )
        if self.agent_ablation_config.cog_use_scratchpad:
            tool_names += ["scratchpad"]
            self.tools.append(
                tools_registry["scratchpad"](
                    model_provider=provider,
                    is_live=self.sweb_live,
                )
            )

        self.tool_caller: ToolExecutor = ToolExecutor(self.tools)

        self.initial_messages: list[LLMMessage] = []
        sys_prompt = self.get_system_prompt_wo_other_tools()
        if self.agent_ablation_config.cog_use_hypotheses_tracking:
            sys_prompt += "\n" + self.get_system_prompt_hypotheses_tracking()
        if self.agent_ablation_config.cog_use_scratchpad:
            sys_prompt += "\n" + self.get_system_prompt_scratchpad()
        self.initial_messages.append(LLMMessage(role="system", content=sys_prompt))
        user_message = ""
        if "project_path" not in extra_args:
            raise AgentError("Project path is required")

        self.project_path = extra_args.get("project_path", "")
        user_message += f"[Project root path]:\n{self.project_path}\n\n"

        if "issue" in extra_args:
            user_message += f"[Problem statement]: We're currently solving the following issue within our repository. Here's the issue text:\n{extra_args['issue']}\n"
        optional_attrs_to_set = ["base_commit", "must_patch", "patch_path"]
        for attr in optional_attrs_to_set:
            if attr in extra_args:
                setattr(self, attr, extra_args[attr])

        self.initial_messages.append(LLMMessage(role="user", content=user_message))

        # If trajectory recorder is set, start recording
        if self.trajectory_recorder:
            self.trajectory_recorder.start_recording(
                task=task,
                provider=self.llm_client.provider.value,
                model=self.model_parameters.model,
                max_steps=self.max_steps,
            )

    @override
    async def execute_task(self) -> AgentExecution:
        """Execute the task and finalize trajectory recording."""
        console_task = (
            asyncio.create_task(self.cli_console.start()) if self.cli_console else None
        )
        execution = await super().execute_task()
        if self.cli_console and console_task and not console_task.done():
            await console_task

        # Finalize trajectory recording if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.finalize_recording(
                success=execution.success, final_result=execution.final_result
            )

        if self.patch_path is not None:
            with open(self.patch_path, "w") as patch_f:
                patch_f.write(self.get_git_diff())

        return execution

    def get_system_prompt_wo_other_tools(self) -> str:
        return self._read_agent_prompt("sys_prompt_wo_other_tools.txt")

    def get_system_prompt_hypotheses_tracking(self) -> str:
        return self._read_agent_prompt("sys_prompt_hypo_tracking.txt")

    def get_system_prompt_scratchpad(self) -> str:
        return self._read_agent_prompt("sys_prompt_scratchpad.txt")

    def _read_agent_prompt(self, filename: str) -> str:
        return (Path(__file__).resolve().parent / filename).read_text()

    @override
    def reflect_on_result(self, tool_results: list[ToolResult]) -> str | None:
        return None

    def get_git_diff(self) -> str:
        """Get the git diff of the project."""
        pwd = os.getcwd()
        if not os.path.isdir(self.project_path):
            return ""
        os.chdir(self.project_path)
        try:
            if not self.base_commit:
                stdout = subprocess.check_output(["git", "--no-pager", "diff"]).decode()
            else:
                stdout = subprocess.check_output(
                    ["git", "--no-pager", "diff", self.base_commit, "HEAD"]
                ).decode()
        except (subprocess.CalledProcessError, FileNotFoundError):
            stdout = ""
        finally:
            os.chdir(pwd)
        return stdout

    # Copyright (c) 2024 paul-gauthier
    # SPDX-License-Identifier: Apache-2.0
    # Original remove_patches_to_tests function was released under Apache-2.0 License, with the full license text
    # available at https://github.com/Aider-AI/aider-swe-bench/blob/6e98cd6c3b2cbcba12976d6ae1b07f847480cb74/LICENSE.txt
    # Original function is at https://github.com/Aider-AI/aider-swe-bench/blob/6e98cd6c3b2cbcba12976d6ae1b07f847480cb74/tests.py#L45

    def remove_patches_to_tests(
        self, model_patch: str, include_config_tests: bool = True
    ) -> str:
        """
        Remove any changes to the tests directory from the provided patch.
        This is to ensure that the model_patch does not disturb the repo's
        tests when doing acceptance testing with the `test_patch`.
        """
        lines = model_patch.splitlines(keepends=True)
        filtered_lines: list[str] = []
        is_tests = False

        for line in lines:
            if line.startswith("diff --git a/"):
                target_path = self._diff_target_path(line)
                is_tests = self._is_test_patch_path(
                    target_path, include_config_tests=include_config_tests
                )

            if not is_tests:
                filtered_lines.append(line)

        return "".join(filtered_lines)

    def _diff_target_path(self, diff_header: str) -> str:
        parts = diff_header.split()
        if not parts:
            return ""
        target_path = parts[-1]
        if target_path.startswith("b/"):
            return target_path[2:]
        return target_path

    def _is_test_patch_path(
        self, path: str, include_config_tests: bool = False
    ) -> bool:
        normalized = path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        basename = normalized.rsplit("/", 1)[-1]
        return (
            any(part in TEST_PATCH_DIR_NAMES for part in parts[:-1])
            or basename.startswith(TEST_PATCH_FILE_PREFIXES)
            or basename.endswith(TEST_PATCH_FILE_SUFFIXES)
            or (include_config_tests and basename in TEST_PATCH_CONFIG_FILES)
        )

    @override
    def llm_indicates_task_completed(self, llm_response: LLMResponse) -> bool:
        """Check if the LLM indicates that the task is completed."""
        if llm_response.tool_calls is None:
            return False
        return any(
            tool_call.name == "task_done" for tool_call in llm_response.tool_calls
        )

    @override
    def is_task_completed(self, llm_response: LLMResponse) -> bool:
        """Enhanced task completion detection."""
        if self.must_patch == "true":
            model_patch = self.get_git_diff()

            patch = self.remove_patches_to_tests(model_patch)
            if not patch.strip():
                return False

        return True

    @override
    def task_incomplete_message(self) -> str:
        """Return a message indicating that the task is incomplete."""
        return (
            "ERROR! Your Patch is empty. Please provide a patch that fixes the problem."
        )
