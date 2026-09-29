# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tools used by the tool architecture study."""

from typing import Type

from .base import Tool, ToolCall, ToolExecutor, ToolResult
from .bash_tool import BashTool
from .task_done_tool import TaskDoneTool
from .atomic_edit_tool import AtomicEditTool
from .atomic_search_tool import AtomicSearchTool
from .atomic_view_tool import AtomicViewTool
from .atomic_insert_tool import AtomicInsertTool
from .atomic_create_tool import AtomicCreateTool
from .subagent_search_tool import SubagentSearchTool
from .hypotheses_tracking_tool import HypothesesTrackingTool
from .scratchpad_tool import ScratchpadTool

__all__ = [
    "Tool",
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "BashTool",
    "TaskDoneTool",
    "AtomicEditTool",
    "AtomicSearchTool",
    "AtomicViewTool",
    "AtomicInsertTool",
    "AtomicCreateTool",
    "SubagentSearchTool",
    "HypothesesTrackingTool",
    "ScratchpadTool",
]

tools_registry: dict[str, Type[Tool]] = {
    "bash": BashTool,
    "task_done": TaskDoneTool,
    "atomic_edit": AtomicEditTool,
    "atomic_search": AtomicSearchTool,
    "atomic_view": AtomicViewTool,
    "atomic_insert": AtomicInsertTool,
    "atomic_create": AtomicCreateTool,
    "find_code_context": SubagentSearchTool,
    "hypotheses_tracking": HypothesesTrackingTool,
    "scratchpad": ScratchpadTool,
}
