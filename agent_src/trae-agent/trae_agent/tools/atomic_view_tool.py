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

import asyncio
import os
from typing import override

from .base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter
from .bash_tool import _BashSession

class AtomicViewTool(Tool):


    def __init__(self, model_provider: str | None = None, is_live: bool = False) -> None:
        super().__init__(model_provider)
        self._session: _BashSession | None = None
        self._is_live = is_live
        
    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "view_tool"

    @override
    def get_description(self) -> str:
        return """View the content of a file.
The `file` parameter must be a valid path to a file.
The `add_line_numbers` parameter controls whether to add line numbers to the output.
The `start_line` and `end_line` parameters can be used to specify a range of lines to view.
The line numbers count starts from 1.
The exact range will be [start_line, end_line), meaning that the `start_line` is inclusive and the `end_line` is exclusive.
If `end_line` is larger than the number of lines in the file, it will be adjusted to the last line.
The `start_line` must be smaller than the `end_line` and must be smaller than the number of lines in the file.
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        restart_required = self.model_provider == "openai"

        return [
            ToolParameter(
                name="file",
                type="string",
                description="The file to view.",
                required=True,
            ),
            ToolParameter(
                name="add_line_numbers",
                type="boolean",
                description="Whether to add line numbers to the output.",
                required=True,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="The first line to view (inclusive).",
                required=True,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="The last line to view (exclusive).",
                required=True,
            )
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        file_to_view = str(arguments["file"]) if "file" in arguments else None

        if file_to_view is None:
            return ToolExecResult(
                error=f"No file provided for the {self.get_name()} tool",
                error_code=-1,
            )
        if not os.path.isfile(file_to_view):
            return ToolExecResult(
                error=f"Provided path '{file_to_view}' is not a file.",
                error_code=-1,
            )
        add_line_numbers = bool(arguments["add_line_numbers"]) if "add_line_numbers" in arguments else False
        start_line = int(arguments["start_line"]) if "start_line" in arguments else None
        if start_line is None:
            return ToolExecResult(
                error=f"No start_line provided for the {self.get_name()} tool",
                error_code=-1,
            )
        end_line = int(arguments["end_line"]) if "end_line" in arguments else None
        if end_line is None:
            return ToolExecResult(
                error=f"No end_line provided for the {self.get_name()} tool",
                error_code=-1,
            )
        if start_line < 1 or end_line <= start_line:
            return ToolExecResult(
                error=f"Invalid line range: start_line ({start_line}) must be less than end_line ({end_line}) and both must be positive.",
                error_code=-1,
            )
        try:
            output_lines = []
            with open(file_to_view, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                total_lines = len(lines)
                if start_line > total_lines:
                    return ToolExecResult(
                        error=f"start_line ({start_line}) is greater than the total number of lines in the file ({total_lines}).",
                        error_code=-1,
                    )
                if end_line > total_lines:
                    end_line = total_lines
                for i in range(start_line - 1, end_line):
                    line_content = lines[i].rstrip('\n')
                    if add_line_numbers:
                        output_lines.append(f"{i + 1}: {line_content}")
                    else:
                        output_lines.append(line_content)
            if not output_lines:
                return ToolExecResult(output="No lines to display in the specified range.")
            output = "\n".join(output_lines)
            return ToolExecResult(output=output)
        except Exception as e:
            return ToolExecResult(
                error=f"Error reading file '{file_to_view}': {e}",
                error_code=-1,
            )
        