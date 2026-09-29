# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI-compatible client with tool integrations"""

import copy
import json
import os
import random
import time
import traceback
from typing import List, override

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,    
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput

from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition
import yaml
import tiktoken

from ..tools.base import Tool, ToolCall, ToolResult
from .base_client import BaseLLMClient
from .config import ModelParameters, AgentAblationConfig
from .credentials import resolve_api_key, redact_secret
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from .context_manager_interface import MessageBlockEntryType, ContextManager
from .mcp_chat_completion_context_manager import MCPChatCompletionContextManager

class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible client with tool schema generation."""

    def __init__(
        self, model_parameters: ModelParameters, abl_parameters: AgentAblationConfig
    ):
        super().__init__(model_parameters)
        self.abl_parameters = abl_parameters
        self.use_chat_completion = abl_parameters.use_chat_completion
        self.task_id: str | None = None
        self.api_key = resolve_api_key(model_parameters.api_key)

        if self.base_url is None:
            raise ValueError("OpenAI-compatible API base URL not provided.")

        self.client: openai.OpenAI = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.message_history: list[ChatCompletionMessageParam] = []
        # instead of saving all history in a flat list, we save them in blocks
        # each block represents one step; it is ([an action], env feedbacks)
        # for the first step, there is no 'action'.
        # note that we design to let each step begin with the action because o4-mini requires
        # a tool call and its results appear together in the conversation history
        # self.message_history_blocks: list[list[ChatCompletionMessageParam]] = []
        self.active_message_block: list[MessageBlockEntryType] = []
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o").encode
        self.context_mgr: ContextManager = None

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    async def _get_context(self) -> List[MessageBlockEntryType]:
        return await self.context_mgr.get_current_context()

    def instantiate_context_manager(self, task_id: str) -> None:
        """Instantiate the context manager."""
        self.task_id = task_id
        self.context_mgr = MCPChatCompletionContextManager()

    async def _chat_response_api(
        self, messages: list[LLMMessage], model_parameters: ModelParameters, tools: list[Tool] | None = None, reuse_history: bool = True
    ) -> LLMResponse:
        """Send chat messages to OpenAI with optional tool support."""
        openai_messages: ResponseInputParam = self.parse_messages(messages)

        tool_schemas = None
        if tools:
            tool_schemas = [
                FunctionToolParam(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.get_input_schema(),
                    strict=True,
                    type="function",
                )
                for tool in tools
            ]

        self.active_message_block.extend(openai_messages)
        # self.message_history_blocks.append(self.active_message_block)
        await self.context_mgr.add_message_block(self.active_message_block)
        self.active_message_block = []

        api_call_input: ResponseInputParam = []
        if reuse_history:
            ori_message_history = self.message_history
            current_context = await self._get_context()
            api_call_input.extend(current_context)
            ori_message_history.extend(openai_messages)
            # Don't need to extend openai_messages here, because they are already in message_history_blocks
            self.message_history = ori_message_history
        else:
            api_call_input.extend(openai_messages)
            self.message_history = api_call_input

        is_reasoning = False
        if "o3" in model_parameters.model or "o4" in model_parameters.model:
            is_reasoning = True

        response = None
        error_message = ""
        for i in range(model_parameters.max_retries):
            try:
                response = self.client.responses.create(
                    input=api_call_input,
                    model=model_parameters.model,
                    tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
                    temperature=(
                        model_parameters.temperature
                        if not is_reasoning
                        else openai.NOT_GIVEN
                    ),
                    top_p=model_parameters.top_p,
                    max_output_tokens=model_parameters.max_tokens,
                    # this is a must for gpt-oss-120b
                    tool_choice='auto',
                )
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {redact_secret(str(e), self.api_key)}\n"
                # Randomly sleep for 3-30 seconds
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(
                f"Failed to get response from OpenAI after max retries: {error_message}"
            )

        # self.message_history = api_call_input + response.output
        # self.message_history = api_call_input

        content = ""
        tool_calls: list[ToolCall] = []
        for output_block in response.output:
            if output_block.type == "function_call":
                try:
                    tool_calls.append(
                        ToolCall(
                            call_id=output_block.call_id,
                            name=output_block.name,
                            arguments=(
                                json.loads(output_block.arguments)
                                if output_block.arguments
                                else {}
                            ),
                            id=output_block.id,
                        )
                    )
                except Exception as e:
                    raise e
                self.message_history.append(output_block)
                self.active_message_block.append(output_block)
            elif output_block.type == "message":
                content = "".join(
                    content_block.text
                    for content_block in output_block.content
                    if content_block.type == "output_text"
                )
                # change to the simple format: {"role": "assistant", "content": content}
                self.message_history.append({"role": "assistant", "content": content})
                self.active_message_block.append({"role": "assistant", "content": content})
                # self.message_history.append(output_block)
                # self.active_message_block.append(output_block)
            elif output_block.type == "reasoning":
                item_dict = output_block.to_dict()
                del item_dict["status"]
                self.message_history.append(item_dict)
                self.active_message_block.append(item_dict)

        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_input_tokens=response.usage.input_tokens_details.cached_tokens,
                reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
            )

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.status,
            tool_calls=tool_calls if len(tool_calls) > 0 else None,
        )
        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai",
                model=model_parameters.model,
                tools=tools,
                context=api_call_input,
        )
        return llm_response

    async def _chat_chat_completion_api(
        self, messages: list[LLMMessage], model_parameters: ModelParameters, tools: list[Tool] | None = None, reuse_history: bool = True
    ) -> LLMResponse:
        openai_messages: list[ChatCompletionMessageParam]  = self.parse_messages_for_chat_completion(messages)

        tool_schemas = None
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.get_input_schema(),
                    ),
                    type="function",
                )
                for tool in tools
            ]

        self.active_message_block.extend(openai_messages)
        # self.message_history_blocks.append(self.active_message_block)
        await self.context_mgr.add_message_block(self.active_message_block)
        self.active_message_block = []

        api_call_input: list[ChatCompletionMessageParam] = []
        if reuse_history:
            ori_message_history = self.message_history
            current_context = await self._get_context()
            api_call_input.extend(current_context)
            ori_message_history.extend(openai_messages)
            # Don't need to extend openai_messages here, because they are already in message_history_blocks
            self.message_history = ori_message_history
        else:
            api_call_input.extend(openai_messages)
            self.message_history = api_call_input

        is_reasoning = False
        if "o3" in model_parameters.model or "o4" in model_parameters.model:
            is_reasoning = True

        new_api_call_input = []
        for msg in api_call_input:
            if type(msg) != dict:
                new_api_call_input.append(msg)
            elif 'reasoning_content' not in msg:
                new_api_call_input.append(msg)
            else:
                new_msg = copy.deepcopy(msg)
                del new_msg['reasoning_content']
                new_api_call_input.append(new_msg)
        api_call_input = new_api_call_input

        response = None
        error_message = ""
        for i in range(model_parameters.max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=api_call_input,
                    model=model_parameters.model,
                    tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
                    temperature=(
                        model_parameters.temperature
                        if not is_reasoning
                        else openai.NOT_GIVEN
                    ),
                    frequency_penalty=model_parameters.frequency_penalty,
                    # not supported by bedrock models
                    # parallel_tool_calls=False,          
                    max_tokens=model_parameters.max_tokens,                    
                    # tool_choice='required',
                    # this must be 'auto' for all models requiring a specific tool parser!
                    tool_choice='auto',
                )
                # self.message_history = api_call_input + response.output
                # self.message_history = api_call_input
                if response.choices[0].message.tool_calls is None:
                    non_empty_tool_calls = []
                else:
                    non_empty_tool_calls = [tc for tc in response.choices[0].message.tool_calls if tc is not None]

                tool_calls: list[ToolCall] = []
                choice = response.choices[0]
                if len(non_empty_tool_calls) > 1:
                    print("Warning: more than one tool call in the response. We will only use the first one.")
                for tc in non_empty_tool_calls[:1]:                    
                    call_id = tc.id
                    name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except Exception as e:
                        # put the raw arguments in exception message
                        e.args += (tc.function.arguments, )
                        raise e
                    tool_calls.append(
                        ToolCall(
                            call_id=call_id,
                            name=name,
                            arguments=arguments,
                        )
                    )
                break
            except Exception as e:
                error_message += f"Model: {model_parameters.model}, Base URL: {redact_secret(str(self.base_url), self.api_key)}\n"
                # print stack trace
                stack_trace = redact_secret(traceback.format_exc(), self.api_key)
                error_message += f"Error {i + 1}: {redact_secret(str(e), self.api_key)}\n{stack_trace}\n"
                # Randomly sleep for 3-30 seconds
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(
                f"Failed to get response from OpenAI after max retries: {error_message}"
            )

        content = choice.message.content if choice.message.content else ''
        # reasoning_content = choice.message.reasoning_content if choice.message.reasoning_content else ''        
        # if reasoning content is an attribute of the choice.message, then we need to add it to the msg_to_append
        if hasattr(choice.message, 'reasoning_content'):
            reasoning_content = choice.message.reasoning_content
        else:
            reasoning_content = ''
        msg_to_append = {
            'role': 'assistant',
            'content': choice.message.content if choice.message.content else '',
            'reasoning_content': reasoning_content,
            'tool_calls': [tc.to_dict() for tc in non_empty_tool_calls[:1]],
        }
        self.message_history.append(msg_to_append)
        self.active_message_block.append(msg_to_append)

        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cache_read_input_tokens=0,
                reasoning_tokens=0,
            )       

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
        )
        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai",
                model=model_parameters.model,
                tools=tools,
                context=api_call_input,
        )
        return llm_response

    @override
    async def chat(
        self,
        messages: list[LLMMessage],
        model_parameters: ModelParameters,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to OpenAI with optional tool support."""
        if self.use_chat_completion:
            return await self._chat_chat_completion_api(messages, model_parameters, tools, reuse_history)
        else:
            return await self._chat_response_api(messages, model_parameters, tools, reuse_history)

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""

        if "o1-mini" in model_parameters.model:
            return False

        tool_capable_models = [
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.5",
            "o1",
            "o3",
            "o3-mini",
            "o4-mini",
            'Qwen3'
        ]
        return any(model in model_parameters.model for model in tool_capable_models)

    def _tool_call_result_to_string(self, tool_call_result: ToolResult) -> str:
        result_content: str = ""
        if tool_call_result.result is not None:
            result_content += str(tool_call_result.result)
        if tool_call_result.error:
            result_content += f"\nError: {tool_call_result.error}"
        result_content = result_content.strip()
        return result_content

    def parse_messages_for_chat_completion(self, messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if msg.tool_call:
                openai_messages.append(
                    ChatCompletionMessageFunctionToolCallParam(
                        id=msg.tool_call.call_id,
                        function=Function(
                            name=msg.tool_call.name,
                            arguments=json.dumps(msg.tool_call.arguments),
                        ),
                        type="function",
                    )                    
                )
            elif msg.tool_result:
                openai_messages.append(
                    ChatCompletionToolMessageParam(
                        content=self._tool_call_result_to_string(msg.tool_result),
                        role="tool",
                        tool_call_id=msg.tool_result.call_id,
                    )
                )
            elif msg.role == "system":
                openai_messages.append(
                    ChatCompletionSystemMessageParam(content=msg.content, role="system")
                )
            elif msg.role == "user":
                openai_messages.append(
                    ChatCompletionUserMessageParam(content=msg.content, role="user")
                )
            elif msg.role == "assistant":
                openai_messages.append(
                    ChatCompletionAssistantMessageParam(content=msg.content, role="assistant")
                )
            else:
                raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_messages(self, messages: list[LLMMessage]) -> ResponseInputParam:
        """Parse the messages to OpenAI format."""
        openai_messages: ResponseInputParam = []
        for msg in messages:
            if msg.tool_result:
                openai_messages.append(self.parse_tool_call_result(msg.tool_result))
            elif msg.tool_call:
                openai_messages.append(self.parse_tool_call(msg.tool_call))
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    openai_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    openai_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_tool_call(self, tool_call: ToolCall) -> ResponseFunctionToolCallParam:
        """Parse the tool call from the LLM response."""
        return ResponseFunctionToolCallParam(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=json.dumps(tool_call.arguments),
            type="function_call",
        )

    def parse_tool_call_result(
        self, tool_call_result: ToolResult
    ) -> FunctionCallOutput:
        """Parse the tool call result from the LLM response to FunctionCallOutput format."""
        result_content: str = self._tool_call_result_to_string(tool_call_result)

        return FunctionCallOutput(
            type="function_call_output",  # Explicitly set the type field
            call_id=tool_call_result.call_id,
            output=result_content,
        )
