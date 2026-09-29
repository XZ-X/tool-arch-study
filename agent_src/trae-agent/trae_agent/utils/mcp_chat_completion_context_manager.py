from .context_manager_interface import ContextManager, MessageBlockEntryType
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,    
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam
)
from typing import List, Any, Dict, Optional
from pydantic import BaseModel
import asyncio
import json
import uuid
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import yaml

class MCPChatCompletionContextManager(ContextManager):
    def __init__(self, run_id: Optional[str] = None, meta_info: Optional[Dict[str, Any]] = None) -> None:
        mcp_url = yaml.safe_load(open("mcp_url.yaml", "r"))["url"]
        self.mcp_url = mcp_url
        self.run_id = run_id or f"run_{uuid.uuid4().hex}"
        self._initialized: bool = False
        # Keep a local copy for get_all_message_blocks()
        self._message_blocks: List[List[ChatCompletionMessageParam]] = []
        self.meta_info = meta_info

    async def add_message_block(self, messages: List[ChatCompletionMessageParam]) -> None:
        if len(messages) == 0:
            return
        self._message_blocks.append(list(messages))
        await self._ensure_initialized()
        # Convert any pydantic models to plain dict for JSON transport
        await self._call_tool(
            "add_new_message_block",
            arguments={"run_id": self.run_id, "messages": messages},
        )
    
    async def get_all_message_blocks(self) -> List[List[ChatCompletionMessageParam]]:
        return self._message_blocks
    
    async def get_current_context(self) -> List[ChatCompletionMessageParam]:
        await self._ensure_initialized()
        rsp = await self._call_tool(
            "get_context",
            arguments={"run_id": self.run_id},
        )
        # Response content is a list of content parts; each .text is a JSON-encoded list
        contexts: List[ChatCompletionMessageParam] = []
        if hasattr(rsp, "content") and rsp.content:
            for c in rsp.content:
                try:
                    part = json.loads(c.text)
                    if isinstance(part, list):
                        contexts.extend(part)
                    else:
                        # Fallback if server returned a single object
                        contexts.append(part)
                except Exception:
                    # If parsing fails, skip this chunk
                    continue
        return contexts

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        # Always set context_manager_type to "dummy" for now
        if self.meta_info is None:
            self.meta_info = {"context_manager_type": "dummy"}
        await self._call_tool(
            "set_run_info",
            arguments={"run_id": self.run_id, "run_info": self.meta_info},
        )
        self._initialized = True

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        # Create a short-lived session per call to avoid lifecycle complexity
        async with streamablehttp_client(
            url=self.mcp_url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            sse_read_timeout=600,
        ) as (reader, writer, _):
            session = ClientSession(reader, writer, read_timeout_seconds=None)
            async with session as s:
                await s.initialize()
                return await s.call_tool(
                    tool_name,
                    arguments=arguments,
                    read_timeout_seconds=None,
                )
