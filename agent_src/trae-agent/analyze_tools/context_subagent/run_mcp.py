from typing import Any, Dict, List, Tuple
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
import uuid
import json
import time
from pathlib import Path
import os
import sys
from datetime import datetime
import random
import string
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,    
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,    
)
from datetime import timedelta
import asyncio
import logging
import argparse
import traceback
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_tools.context_subagent.main_context_subagent import (  # noqa: E402
    configure_context_subagent,
    query_context_subagent,
)

mcp = FastMCP(
    "search-tool-server",
    stateless_http=True,  # optional: no session persistence
    json_response=True,  # key: respond with a single JSON object, no SSE stream
)

@mcp.tool()
async def search_context(docker_id: str, query: str) -> dict:
    try:
        ret = await query_context_subagent(docker_id, query)
    except Exception as e:
        logging.error(f"Error searching context for docker_id '{docker_id}' and query '{query}': {e}")
        raise e
    return {
        "context": ret,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the context subagent MCP server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=14389, help="Port to run the server on.")
    parser.add_argument(
        "--model",
        "--model-short-name",
        dest="model_short_name",
        type=str,
        default=os.environ.get("CONTEXT_SUBAGENT_MODEL", "Qwen3Coder-30B"),
        help=(
            "Model alias from models.yaml for the search "
            "subagent. Defaults to CONTEXT_SUBAGENT_MODEL or Qwen3Coder-30B."
        ),
    )
    args = parser.parse_args()
    configure_context_subagent(args.model_short_name)
    uvicorn.run(
        mcp.streamable_http_app(), host=args.host, port=args.port, log_level="info"
    )
