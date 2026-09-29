from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ToolCallRecord:
    call_id: str
    name: str
    arguments: str
    id: str | None = None


@dataclass
class ToolResultRecord:
    call_id: str
    success: bool
    result: str | None
    error: str | None = None
    id: str | None = None


class TrajectoryRecorder:
    """Records trajectory data with the same JSON schema as trae-agent."""

    def __init__(self, trajectory_path: str | None = None):
        if trajectory_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_path = f"trajectory_{timestamp}.json"

        self.trajectory_path = Path(trajectory_path)
        self.trajectory_data: dict[str, Any] = {
            "task": "",
            "start_time": "",
            "end_time": "",
            "provider": "",
            "model": "",
            "max_steps": 0,
            "llm_interactions": [],
            "agent_steps": [],
            "success": False,
            "final_result": None,
            "execution_time": 0.0,
        }
        self._start_time: datetime | None = None

    def start_recording(self, task: str, provider: str, model: str, max_steps: int) -> None:
        self._start_time = datetime.now()
        self.trajectory_data.update({
            "task": task,
            "start_time": self._start_time.isoformat(),
            "provider": provider,
            "model": model,
            "max_steps": max_steps,
            "llm_interactions": [],
            "agent_steps": [],
        })
        self.save_trajectory()

    def record_llm_interaction(
        self,
        messages: list[dict[str, Any]],
        context: list[dict[str, Any]],
        response_content: str,
        provider: str,
        model: str,
        tool_calls: list[ToolCallRecord] | None = None,
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        tools_available: list[str] | None = None,
    ) -> None:
        serialized_messages = [self._serialize_message(m) for m in messages]
        serialized_context = [self._serialize_message(m) for m in context]
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "input_messages": serialized_messages,
            "context": serialized_context,
            "context_eval_string": str(context),
            "response": {
                "content": response_content,
                "model": model,
                "finish_reason": finish_reason,
                "usage": {
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens") if usage else None,
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens") if usage else None,
                    "reasoning_tokens": usage.get("reasoning_tokens") if usage else None,
                },
                "tool_calls": [self._serialize_tool_call(tc) for tc in tool_calls]
                if tool_calls
                else None,
            },
            "tools_available": tools_available,
        }
        self.trajectory_data["llm_interactions"].append(interaction)
        self.save_trajectory()

    def record_agent_step(
        self,
        step_number: int,
        state: str,
        llm_messages: list[dict[str, Any]] | None = None,
        llm_response: dict[str, Any] | None = None,
        tool_calls: list[ToolCallRecord] | None = None,
        tool_results: list[ToolResultRecord] | None = None,
        reflection: str | None = None,
        error: str | None = None,
    ) -> None:
        step_data = {
            "step_number": step_number,
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "llm_messages": [self._serialize_message(m) for m in llm_messages]
            if llm_messages
            else None,
            "llm_response": llm_response,
            "tool_calls": [self._serialize_tool_call(tc) for tc in tool_calls]
            if tool_calls
            else None,
            "tool_results": [self._serialize_tool_result(tr) for tr in tool_results]
            if tool_results
            else None,
            "reflection": reflection,
            "error": error,
        }
        self.trajectory_data["agent_steps"].append(step_data)
        self.save_trajectory()

    def finalize_recording(self, success: bool, final_result: str | None = None) -> None:
        end_time = datetime.now()
        self.trajectory_data.update({
            "end_time": end_time.isoformat(),
            "success": success,
            "final_result": final_result,
            "execution_time": (end_time - self._start_time).total_seconds()
            if self._start_time
            else 0.0,
        })
        self.save_trajectory()

    def save_trajectory(self) -> None:
        try:
            self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trajectory_path, "w", encoding="utf-8") as f:
                json.dump(self.trajectory_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save trajectory to {self.trajectory_path}: {e}")

    @staticmethod
    def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
        message = copy.deepcopy(message)
        if "tool_calls" in message:
            message["tool_calls"] = [
                {
                    "call_id": tc.get("id", tc.get("call_id", "")),
                    "name": tc.get("function", {}).get("name", tc.get("name", "")),
                    "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", "")),
                    "id": tc.get("id", tc.get("call_id", "")),
                }
                for tc in message["tool_calls"]
            ]
        return message

    @staticmethod
    def _serialize_tool_call(tool_call: ToolCallRecord) -> dict[str, Any]:
        # Parse arguments from JSON string to dict if needed, so that
        # downstream analysis tools (e.g. study_traj.py) can access fields directly.
        args = tool_call.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": args,
            "id": tool_call.id,
        }

    @staticmethod
    def _serialize_tool_result(tool_result: ToolResultRecord) -> dict[str, Any]:
        return {
            "call_id": tool_result.call_id,
            "success": tool_result.success,
            "result": tool_result.result,
            "error": tool_result.error,
            "id": tool_result.id,
        }
