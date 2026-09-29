from dataclasses import dataclass
from pathlib import Path
from trae_agent.utils.credentials import resolve_api_key
from typing import Any, List, Tuple

from docker import from_env
from analyze_tools.context_subagent.context_subagent_tools import (
    CodeSearchContext,
    ContextSearchResult,
    record_relevant_context,
    remove_relevant_context,
    update_relevant_context,
    bash,
    merge_overlapping_context_spans,
    resolve_repo_root,
)
import os
import asyncio
import traceback
import shlex
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    RunConfig,
    ModelSettings,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    StopAtTools,
    RunContextWrapper,
)
from agents.lifecycle import RunHooksBase
from openai import AsyncOpenAI

import yaml
from analyze_tools.context_subagent.trace_logger import SearchSubagentTraceLogger

system_prompt = (Path(__file__).parent / "prompts/context_subagent_sys_prompt.md").read_text()

set_tracing_disabled(True)
set_default_openai_api("chat_completions")

DEFAULT_MODEL_SHORT_NAME = os.environ.get("CONTEXT_SUBAGENT_MODEL", "Qwen3Coder-30B")


@dataclass(frozen=True)
class SearchSubagentModelRuntime:
    model_short_name: str
    model_name: str
    base_url: str
    model: OpenAIChatCompletionsModel


_model_runtime: SearchSubagentModelRuntime | None = None




def configure_context_subagent(model_short_name: str) -> SearchSubagentModelRuntime:
    global _model_runtime

    with open("models.yaml") as config_file:
        llm_config = yaml.safe_load(config_file)["models"]
    if model_short_name not in llm_config:
        available = ", ".join(sorted(llm_config))
        raise ValueError(
            f"Unknown context subagent model alias '{model_short_name}'. "
            f"Available aliases: {available}"
        )

    model_config = llm_config[model_short_name]
    model_name = model_config["model"]
    base_url = model_config["base_url"]
    api_key = resolve_api_key(model_config.get("api_key", "openai-compatible"))
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_default_openai_client(client)
    _model_runtime = SearchSubagentModelRuntime(
        model_short_name=model_short_name,
        model_name=model_name,
        base_url=base_url,
        model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
    )
    print(
        "Using context subagent model: "
        f"{model_short_name} ({model_name}), base_url: {base_url}, api_key: <redacted>"
    )
    return _model_runtime


def get_context_subagent_runtime() -> SearchSubagentModelRuntime:
    if _model_runtime is None:
        return configure_context_subagent(DEFAULT_MODEL_SHORT_NAME)
    return _model_runtime

@function_tool
async def task_done(ctx: RunContextWrapper[CodeSearchContext], final_output: str) -> str:
    """
Call this tool when you have finished your task.
    """
    tool_call_index = (
        ctx.context.trace_logger.next_tool_step()
        if ctx.context.trace_logger is not None
        else None
    )
    ctx.context.log_event(
        "tool_start",
        tool_call_index=tool_call_index,
        tool="task_done",
        arguments={"final_output": final_output},
    )
    ctx.context.log_event(
        "tool_end",
        tool_call_index=tool_call_index,
        tool="task_done",
        output=final_output,
        error=False,
    )
    return final_output


class SearchSubagentTraceHooks(RunHooksBase[CodeSearchContext, Agent]):
    async def on_agent_start(
        self, context: RunContextWrapper[CodeSearchContext], agent: Agent
    ) -> None:
        context.context.log_event("agent_start", agent=agent.name)

    async def on_agent_end(
        self, context: RunContextWrapper[CodeSearchContext], agent: Agent, output: Any
    ) -> None:
        context.context.log_event("agent_end", agent=agent.name, output=output)

    async def on_llm_start(
        self,
        context: RunContextWrapper[CodeSearchContext],
        agent: Agent,
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        llm_step_index = (
            context.context.trace_logger.next_llm_step()
            if context.context.trace_logger is not None
            else 0
        )
        context.context.current_llm_step_index = llm_step_index
        context.context.log_event(
            "llm_start",
            llm_step_index=llm_step_index,
            agent=agent.name,
            system_prompt_chars=len(system_prompt or ""),
            input_items_count=len(input_items),
            input_items=input_items,
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[CodeSearchContext],
        agent: Agent,
        response: Any,
    ) -> None:
        usage = None
        if context.context.trace_logger is not None and getattr(response, "usage", None):
            usage = context.context.trace_logger.add_usage(response.usage)
        context.context.log_event(
            "llm_end",
            llm_step_index=context.context.current_llm_step_index,
            agent=agent.name,
            response_id=getattr(response, "response_id", None),
            usage=usage,
            output=getattr(response, "output", None),
        )

async def _run_one_query(
    docker_id: str, query: str
) -> Tuple[str, List[dict], CodeSearchContext]:
    runtime = get_context_subagent_runtime()
    trace_logger = SearchSubagentTraceLogger(
        docker_id=docker_id,
        query=query,
        model_short_name=runtime.model_short_name,
        model_name=runtime.model_name,
        base_url=runtime.base_url,
    )
    repo_root = resolve_repo_root(docker_id)
    ctx = CodeSearchContext(
        docker_id=docker_id,
        repo_root=repo_root,
        trace_logger=trace_logger,
    )
    instructions = system_prompt.replace("/testbed", repo_root)
    agent = Agent[CodeSearchContext](
        name="Code Search Subagent",
        instructions=instructions,
        tools=[
            record_relevant_context,
            remove_relevant_context,
            update_relevant_context,
            bash,
            task_done,
        ],
        model=runtime.model,
        tool_use_behavior=StopAtTools(stop_at_tool_names=["task_done"]),
    )
    model_settings = ModelSettings(
        temperature=0.0,        
    )
    run_config = RunConfig(model_settings=model_settings)
    try:
        ret = await Runner.run(
            agent,
            query,
            run_config=run_config,
            max_turns=50,
            context=ctx,
            hooks=SearchSubagentTraceHooks(),
        )
    except Exception as e:
        trace_logger.close(
            status="error",
            error=str(e),
            traceback=traceback.format_exc(),
            spans=ctx.get_sorted_span_list(),
        )
        raise
    return (ret.final_output, ret.to_input_list(), ctx)

docker_client = from_env()
def _read_file_snippet_from_docker(docker_id: str, path: str, line_start: int, line_end: int) -> str:
    container = docker_client.containers.get(docker_id)
    quoted_path = shlex.quote(path)
    file_cmd = f"cat -n {quoted_path} | head -n {line_end} | tail -n {line_end - line_start + 1}"
    cmd = [
        "bash", "-lc", file_cmd
    ]
    return_code, output = container.exec_run(cmd)
    output_str= output.decode("utf-8")
    if return_code != 0:
        return None
    else:
        return output_str
    
    

def _pretty_print_as_agent_results(final_output: str, ctx: CodeSearchContext):
    ctx_strings = ""
    docker_id = ctx.docker_id
    merged_spans = merge_overlapping_context_spans(ctx)
    for span in merged_spans:
        file_content = _read_file_snippet_from_docker(docker_id, span.filename, span.start_line, span.end_line)
        ctx.log_event(
            "snippet_read",
            filename=span.filename,
            start_line=span.start_line,
            end_line=span.end_line,
            success=file_content is not None,
            output_chars=len(file_content) if file_content is not None else 0,
        )
        if file_content is None:
            continue
        ctx_strings += f"<ContextID id={span.id}>\n"
        ctx_strings += f"Reasoning: {span.comment}\n"
        ctx_strings += f"File and Lines: {span.filename}:{span.start_line}-{span.end_line}\n"
        ctx_strings += f"Content:\n{file_content}\n"
        ctx_strings += f"</ContextID>\n"

    return final_output + "\n\n" + ctx_strings

async def query_context_subagent(docker_id: str, query: str) -> str:
    try:
        ret = await _run_one_query(docker_id, query)
    except Exception as e:
        return "Your query is too complex. Please try to be more specific."
    final_output = ret[0]
    ctx = ret[2]
    try:
        ret_str = _pretty_print_as_agent_results(final_output, ctx)
    except Exception as e:
        if ctx.trace_logger is not None:
            ctx.trace_logger.close(
                status="error",
                final_output=final_output,
                error=str(e),
                traceback=traceback.format_exc(),
                spans=ctx.get_sorted_span_list(),
            )
        return "Your query is too complex. Please try to be more specific."
    if ctx.trace_logger is not None:
        merged_spans = merge_overlapping_context_spans(ctx)
        ctx.trace_logger.close(
            status="ok",
            final_output=final_output,
            merged_spans=merged_spans,
            final_context_chars=len(ret_str),
        )
    return ret_str

# ret = asyncio.run(run_one_query(docker_id="79e3ab", query="what is the E3054 rule?"))
# final_output = ret[0]
# ctx = ret[2]
# ret_str = _pretty_print_as_agent_results(final_output, ctx)
