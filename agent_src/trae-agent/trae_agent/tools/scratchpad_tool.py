# Copyright (c) 2023 Anthropic
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 13 June 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/anthropics/anthropic-quickstarts/blob/main/LICENSE
#
# This modified file is released under the same license.

import json
from dataclasses import dataclass
from typing import override

from .base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


@dataclass
class ScratchpadData:
    scratchpad: str


class ScratchpadTool(Tool):
    """A scratchpad tool for jotting intermediate thoughts and results."""

    @override
    def get_name(self) -> str:
        return "scratchpad"

    @override
    def get_description(self) -> str:
        return """A scratchpad for intermediate notes.

Use this tool to write intermediate thoughts, partial results, or reminders.
Keep it focused. Do not put the final answer here."""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="scratchpad",
                type="string",
                description="Intermediate notes / scratchpad text.",
                required=True,
            ),
        ]

    def __init__(self, model_provider: str | None = None, is_live: bool = False) -> None:
        super().__init__(model_provider)
        self.is_live = is_live
        self.history: list[ScratchpadData] = []
        self.latest: str = ""

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    def _validate(self, arguments: ToolCallArguments) -> ScratchpadData:
        if "scratchpad" not in arguments or not isinstance(arguments["scratchpad"], str):
            raise ValueError("Invalid scratchpad: must be a string")
        text = str(arguments["scratchpad"])
        if not text.strip():
            raise ValueError("Invalid scratchpad: must be a non-empty string")
        return ScratchpadData(scratchpad=text)

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            data = self._validate(arguments)
            self.latest = data.scratchpad
            self.history.append(data)

            response_data = {
                "history_length": len(self.history),
                "latest_length": len(self.latest),
            }

            return ToolExecResult(
                output=(
                    "Scratchpad updated.\n\nStatus:\n"
                    f"{json.dumps(response_data, indent=2)}"
                )
            )

        except Exception as e:
            error_data = {"error": str(e), "status": "failed"}
            return ToolExecResult(
                error=(
                    f"Scratchpad failed: {str(e)}\n\nDetails:\n"
                    f"{json.dumps(error_data, indent=2)}"
                ),
                error_code=-1,
            )
