import asyncio
import os
from typing import override, List, Any, Dict, Optional
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter
import yaml
import json
import traceback


cid = None
def get_my_container_id() -> str:
    global cid
    if cid is None:
        host_name = open('/etc/hostname', 'r').read().strip()
        cid = host_name
    return cid


class SubagentSearchTool(Tool):

    def __init__(self, model_provider: str | None = None, is_live: bool = False):
        mcp_url = yaml.safe_load(open("mcp_url.yaml", "r"))["ctx_tool"]
        self.mcp_url = mcp_url
        self.is_live = is_live
        self.cid = get_my_container_id()        
        self.description = open("trae-agent/trae_agent/tools/descriptions/code_context_tool.md", "r").read().strip()

    async def _call_mcp(self, tool_name: str, arguments: Dict[str, Any]):
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

    
    async def _query(self, query: str) -> str:
        rsp = await self._call_mcp("search_context", {"docker_id": self.cid, "query": query})
        rsp_txt = rsp.content[0].text
        rsp_json = json.loads(rsp_txt)
        return rsp_json["context"]

    
    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "find_code_context"

    @override
    def get_description(self) -> str:
        return self.description

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The query to search for code context.",
            )
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        query = arguments["query"]
        try:
            context = await self._query(query)
        except Exception as e:
            # with stack trace
            err_msg = f"Error searching for code context: {e}\n{traceback.format_exc()}"
            return ToolExecResult(error=err_msg, error_code=-1)
        return ToolExecResult(output=context)