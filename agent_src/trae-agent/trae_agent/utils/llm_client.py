# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""LLM client for the study models."""

from enum import Enum

from ..tools.base import Tool
from .base_client import BaseLLMClient
from .config import ModelParameters, AgentAblationConfig
from .llm_basics import LLMMessage, LLMResponse
from .trajectory_recorder import TrajectoryRecorder


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI_COMPATIBLE = "openai-compatible"



class LLMClient:
    """Main LLM client that supports multiple providers."""

    def __init__(
        self,
        provider: str | LLMProvider,
        model_parameters: ModelParameters,
        abl_params: AgentAblationConfig = None,
    ):
        if isinstance(provider, str):
            provider = LLMProvider(provider)

        self.provider: LLMProvider = provider

        from .openai_compatible_client import OpenAICompatibleClient

        self.client: BaseLLMClient = OpenAICompatibleClient(model_parameters, abl_params)

    def instantiate_context_manager(self, task_id: str) -> None:
        """Instantiate the context manager for this client."""
        self.client.instantiate_context_manager(task_id)

    def set_trajectory_recorder(self, recorder: TrajectoryRecorder | None) -> None:
        """Set the trajectory recorder for the underlying client."""
        self.client.set_trajectory_recorder(recorder)

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.client.set_chat_history(messages)

    async def chat(
        self,
        messages: list[LLMMessage],
        model_parameters: ModelParameters,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to the LLM."""
        return await self.client.chat(messages, model_parameters, tools, reuse_history)

    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current client supports tool calling."""
        return hasattr(
            self.client, "supports_tool_calling"
        ) and self.client.supports_tool_calling(model_parameters)
