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
import re
import mimetypes

class AtomicSearchTool(Tool):


    def __init__(self, model_provider: str | None = None, is_live: bool = False) -> None:
        super().__init__(model_provider)
        self._session: _BashSession | None = None
        self._is_live = is_live

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "search_tool"

    @override
    def get_description(self) -> str:
        return """Search a string in a file or a directory.
The tool will search for the given string in all files under the specified directory.
Syntax of python regular expression is supported.
You don't need to additionally escape the regular expression string.
Provide them as you would use in the r string in Python.
For example, to search for a definition starting with 'def' and ending with '(', you can use the following:
pattern: "def\\s+\\w+\\s*\\("
file_or_dir: "/path/to/file/or/dir"
The tool will interpret the pattern just like pattern = re.compile(r"def\\s+\\w+\\s*\\(") in Python.

If more than 20 matches are found, only the first 20 matches will be returned.
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        restart_required = self.model_provider == "openai"

        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="The regular expression pattern to search for.",
                required=True,
            ),
            ToolParameter(
                name="file_or_dir",
                type="string",
                description="The file or directory to search in.",
                required=True,
            )
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        pattern = str(arguments["pattern"]) if "pattern" in arguments else None
        file_or_dir = str(arguments["file_or_dir"]) if "file_or_dir" in arguments else None

        if pattern is None or file_or_dir is None:
            return ToolExecResult(
                error=f"Pattern or file/directory not provided for the {self.get_name()} tool",
                error_code=-1,
            )

        if not os.path.exists(file_or_dir):
            return ToolExecResult(
                error=f"Provided path '{file_or_dir}' does not exist.",
                error_code=-1,
            )

        if not os.path.isdir(file_or_dir) and not os.path.isfile(file_or_dir):
            return ToolExecResult(
                error=f"Provided path '{file_or_dir}' is neither a file nor a directory.",
                error_code=-1,
            )

        try:
            matches = []
            errors = []
            if os.path.isfile(file_or_dir):
                potential_mime_type, _ = mimetypes.guess_type(file_or_dir)
                if potential_mime_type is None or not potential_mime_type.startswith('text'):
                    potential_non_text = True
                else:
                    potential_non_text = False
                try:
                    with open(file_or_dir, 'r', encoding='utf-8') as f:
                        content_lines = f.readlines()
                except Exception as e:
                    if potential_non_text:
                        error_msg = f"File '{file_or_dir}' might not be a text file, it has type '{potential_mime_type}'."
                        error_msg += f" Error reading file: {str(e)}"
                        return ToolExecResult(
                            error=error_msg, error_code=-1
                        )

                for line_number, line in enumerate(content_lines, start=1):
                    if re.search(pattern, line):
                        matches.append(f"{file_or_dir}:{line_number}: {line.strip()}")
                    if len(matches) >= 20:
                        break
            elif os.path.isdir(file_or_dir):
                for root, _, files in os.walk(file_or_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        potential_mime_type, _ = mimetypes.guess_type(file_path)
                        if potential_mime_type is None or not potential_mime_type.startswith('text'):
                            potential_non_text = True
                        else:
                            potential_non_text = False
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content_lines = f.readlines()
                        except Exception as e:
                            if potential_non_text:
                                continue
                            errors.append(f"Error reading {file_path}: {e}")
                            continue
                        for line_number, line in enumerate(content_lines, start=1):
                            if re.search(pattern, line):
                                matches.append(f"{file_path}:{line_number}: {line.strip()}")
                            if len(matches) >= 20:
                                break
                        if len(matches) >= 20:
                            break
            if not matches:
                return ToolExecResult(output="No matches found.")
            potential_errors = "\n".join(errors) if errors else ""
            output = "\n".join(matches)
            if potential_errors:
                output += f"\n\nPotential errors encountered:\n{potential_errors}"
            return ToolExecResult(output=output)
        except Exception as e:
            return ToolExecResult(
                error=f"Error searching for pattern: {e}", error_code=-1
            )