from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from agents import function_tool
from agents import RunContextWrapper
import asyncio
import docker
import os
import shlex
import time
from docker.errors import NotFound, APIError

class ContextSpan(BaseModel):
    id: str = Field(..., description="Stable identifier for this context span")
    filename: str = Field(..., description="Repo-relative path")
    start_line: int = Field(..., ge=1, description="1-indexed start line, inclusive")
    end_line: int = Field(..., ge=1, description="1-indexed end line, inclusive")
    comment: Optional[str] = Field(None, description="Optional short note on why it's relevant")



class ContextSearchResult(BaseModel):
    results: List[ContextSpan] = Field(default_factory=list)


DEFAULT_REPO_ROOT = "/testbed"
REPO_ROOT_CANDIDATES = tuple(
    candidate
    for candidate in os.environ.get(
        "CONTEXT_SUBAGENT_REPO_ROOT_CANDIDATES",
        "/testbed:/app:/workspace:/repo",
    ).split(":")
    if candidate
)


def _normalize_fpath(path: str, repo_root: str = DEFAULT_REPO_ROOT) -> str:
    path = path.strip()
    if not path:
        return repo_root
    if path.startswith("/"):
        return path
    root = repo_root.rstrip("/")
    path = path.lstrip("/")
    if path.startswith(root.lstrip("/") + "/"):
        return "/" + path
    return f"{root}/{path}"


def resolve_repo_root(docker_id: str) -> str:
    container = _DOCKER_CLIENT.containers.get(docker_id)
    for candidate in REPO_ROOT_CANDIDATES:
        quoted = shlex.quote(candidate)
        res = container.exec_run(
            cmd=["bash", "-lc", f"test -d {quoted}/.git || test -d {quoted}"],
            workdir="/",
            stdout=True,
            stderr=True,
            demux=False,
        )
        if getattr(res, "exit_code", 1) == 0:
            return candidate.rstrip("/") or DEFAULT_REPO_ROOT
    return DEFAULT_REPO_ROOT


def _ensure_repo_root(ctx: "CodeSearchContext") -> str:
    if not ctx.repo_root:
        ctx.repo_root = resolve_repo_root(ctx.docker_id)
        ctx.log_event("repo_root_resolved", repo_root=ctx.repo_root)
    return ctx.repo_root

@dataclass
class CodeSearchContext:
    docker_id: str = Field(..., description="Docker container id")
    repo_root: str = DEFAULT_REPO_ROOT
    next_id: int = 1
    spans_by_id: Dict[str, ContextSpan] = field(default_factory=dict)
    trace_logger: Any | None = None
    current_llm_step_index: int = 0

    def new_id(self) -> str:
        cid = f"ctx-{self.next_id:04d}"
        self.next_id += 1
        return cid

    def _sort_key(self, k: str) -> int:
        try:
            return int(k.split("-")[-1])
        except Exception:
            return 10**9

    def get_sorted_span_list(self) -> List[ContextSpan]:
        sorted_ctx_entries = sorted(self.spans_by_id.keys(), key=self._sort_key)
        return [self.spans_by_id[k] for k in sorted_ctx_entries]

    def format_list(self, prefix: str = "existing contexts list:", highlight_id: str | None = None) -> str:
        # stable order: by numeric portion of id if possible

        parts: list[str] = []
        for s in self.get_sorted_span_list():
            cid = s.id
            comment = s.comment or ""
            highlight = " (new added context)" if (highlight_id and cid == highlight_id) else ""
            # Example format: ctx-0001, xxx/y.py:3:20, definition of func yyy (new added context)
            item = f"{s.id}, {s.filename}:{s.start_line}:{s.end_line}"
            if comment:
                item += f", {comment}"
            item += highlight
            parts.append(item)

        if not parts:
            return f"{prefix} <empty>"
        return f"{prefix} " + ", ".join(parts)

    def log_event(self, event_type: str, **payload: Any) -> None:
        if self.trace_logger is None:
            return
        payload.setdefault("llm_step_index", self.current_llm_step_index)
        self.trace_logger.log_event(event_type, **payload)


def merge_overlapping_context_spans(ctx: CodeSearchContext) -> List[ContextSpan]:
    """
    Merge overlapping (or touching) spans within the same file, and assign new ids as "0", "1", ...

    - Only spans with the same `filename` are eligible to merge together.
    - Treats spans as closed intervals [start_line, end_line].
    - Merges spans that overlap OR touch (e.g., [10, 20] and [21, 30] -> [10, 30]).
    - Returns new ContextSpan objects (does not mutate input).
    - `comment` is combined (non-empty comments joined by " | ").
    """
    spans = list(ctx.spans_by_id.values())
    if not spans:
        return []

    repo_root = getattr(ctx, "repo_root", DEFAULT_REPO_ROOT) or DEFAULT_REPO_ROOT
    spans.sort(key=lambda s: (_normalize_fpath(s.filename, repo_root), s.start_line, s.end_line))

    merged: List[ContextSpan] = []

    cur = spans[0].model_copy()
    cur.filename = _normalize_fpath(cur.filename, repo_root)
    for s in spans[1:]:
        if _normalize_fpath(s.filename, repo_root) != _normalize_fpath(cur.filename, repo_root):
            merged.append(cur)
            cur = s.model_copy()
            cur.filename = _normalize_fpath(cur.filename, repo_root)
            continue

        if s.start_line <= cur.end_line + 1:  # overlap or touch
            cur.end_line = max(cur.end_line, s.end_line)
            comments = [c for c in (cur.comment, s.comment) if c]
            cur.comment = " | ".join(dict.fromkeys(comments)) if comments else None
        else:
            merged.append(cur)
            cur = s.model_copy()
            cur.filename = _normalize_fpath(cur.filename)

    merged.append(cur)

    # Re-id sequentially
    for i, span in enumerate(merged):
        span.id = str(i)
    return merged


@function_tool
def record_relevant_context(
    ctx: RunContextWrapper[CodeSearchContext],
    filename: str,
    start_line: int,
    end_line: int,
    comment: str | None = None,
) -> str:
    """
    Adds a new span with an auto-generated id and returns ALL contexts as a string.
    """
    new_id = ctx.context.new_id()
    ctx.context.spans_by_id[new_id] = ContextSpan(
        id=new_id,
        filename=filename,
        start_line=start_line,
        end_line=end_line,
        comment=comment,
    )
    output = ctx.context.format_list(highlight_id=new_id)
    ctx.context.log_event(
        "span_recorded",
        tool_call_index=(
            ctx.context.trace_logger.next_tool_step()
            if ctx.context.trace_logger is not None
            else None
        ),
        tool="record_relevant_context",
        arguments={
            "filename": filename,
            "start_line": start_line,
            "end_line": end_line,
            "comment": comment,
        },
        span=ctx.context.spans_by_id[new_id],
        spans_after=ctx.context.get_sorted_span_list(),
        output=output,
    )
    return output



@function_tool
def remove_relevant_context(
    ctx: RunContextWrapper[CodeSearchContext],
    id: str,
) -> str:
    """
    Removes a span by id and returns ALL contexts as a string.
    """
    removed = ctx.context.spans_by_id.pop(id, None)
    if removed is None:
        output = ctx.context.format_list(prefix=f"existing contexts list: (no-op; id not found: {id})")
        ctx.context.log_event(
            "span_remove_noop",
            tool_call_index=(
                ctx.context.trace_logger.next_tool_step()
                if ctx.context.trace_logger is not None
                else None
            ),
            tool="remove_relevant_context",
            arguments={"id": id},
            id=id,
            spans_after=ctx.context.get_sorted_span_list(),
            output=output,
        )
        return output
    output = ctx.context.format_list(prefix=f"existing contexts list: (removed {id})")
    ctx.context.log_event(
        "span_removed",
        tool_call_index=(
            ctx.context.trace_logger.next_tool_step()
            if ctx.context.trace_logger is not None
            else None
        ),
        tool="remove_relevant_context",
        arguments={"id": id},
        id=id,
        removed=removed,
        spans_after=ctx.context.get_sorted_span_list(),
        output=output,
    )
    return output

@function_tool
def update_relevant_context(
    ctx: RunContextWrapper[CodeSearchContext],
    id: str,
    filename: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    comment: str | None = None,
) -> str:
    """
    Patches an existing span by id using only primitive args.
    Any arg left as None is not changed.

    Returns ALL contexts as a string.
    """
    existing = ctx.context.spans_by_id.get(id)
    if existing is None:
        output = ctx.context.format_list(prefix=f"existing contexts list: (no-op; id not found: {id})")
        ctx.context.log_event(
            "span_update_noop",
            tool_call_index=(
                ctx.context.trace_logger.next_tool_step()
                if ctx.context.trace_logger is not None
                else None
            ),
            tool="update_relevant_context",
            arguments={
                "id": id,
                "filename": filename,
                "start_line": start_line,
                "end_line": end_line,
                "comment": comment,
            },
            spans_after=ctx.context.get_sorted_span_list(),
            output=output,
        )
        return output

    data = existing.model_dump()

    if filename is not None:
        data["filename"] = filename
    if start_line is not None:
        data["start_line"] = start_line
    if end_line is not None:
        data["end_line"] = end_line
    if comment is not None:
        data["comment"] = comment

    # Ensure a valid range if both are present and reversed
    if data["start_line"] > data["end_line"]:
        data["start_line"], data["end_line"] = data["end_line"], data["start_line"]

    data["id"] = id  # keep stable
    ctx.context.spans_by_id[id] = ContextSpan(**data)

    output = ctx.context.format_list(prefix=f"existing contexts list: (updated {id})")
    ctx.context.log_event(
        "span_updated",
        tool_call_index=(
            ctx.context.trace_logger.next_tool_step()
            if ctx.context.trace_logger is not None
            else None
        ),
        tool="update_relevant_context",
        arguments={
            "id": id,
            "filename": filename,
            "start_line": start_line,
            "end_line": end_line,
            "comment": comment,
        },
        id=id,
        before=existing,
        after=ctx.context.spans_by_id[id],
        spans_after=ctx.context.get_sorted_span_list(),
        output=output,
    )
    return output

_DOCKER_CLIENT = docker.from_env()

def _exec_in_container_sync(
    docker_id: str,
    command: str,
    workdir: str = DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """
    Synchronous helper: run a shell command inside a running container using docker-py.
    Returns combined output (stdout+stderr) as text.

    Uses: container.exec_run(["bash","-lc", command], workdir=workdir, demux=False)
    """
    container = _DOCKER_CLIENT.containers.get(docker_id)

    # docker-py exec_run supports:
    # - cmd: str | list[str]
    # - workdir: str
    # - stdout/stderr: bool
    # - demux: bool (False => combined stream)
    # - environment, user, privileged, etc.
    start_time = time.time()
    res = container.exec_run(
        cmd=["bash", "-lc", command],
        workdir=workdir,
        stdout=True,
        stderr=True,
        demux=False,
    )
    elapsed_seconds = time.time() - start_time

    # res.output is bytes (or str depending on version/config); normalize.
    output = res.output
    if output is None:
        text = ""
    elif isinstance(output, (bytes, bytearray)):
        text = output.decode("utf-8", errors="replace")
    else:
        text = str(output)

    # If exit_code != 0, still return output (useful for grep misses, etc.)
    # You can optionally annotate errors:
    exit_code = getattr(res, "exit_code", 0)
    if exit_code != 0:
        text = f"(exit_code={exit_code})\n{text}"

    # Optional truncation to keep tool output manageable
    original_chars = len(text)
    truncated = False
    if len(text) > 5000:
        text = text[:5000] + "\n...[truncated]..."
        truncated = True

    return {
        "output": text,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed_seconds,
        "original_output_chars": original_chars,
        "output_chars": len(text),
        "truncated": truncated,
    }


@function_tool
async def bash(
    ctx: RunContextWrapper["CodeSearchContext"],
    command: str,
    timeout_s: int = 15,
) -> str:
    """
    Execute `command` inside the docker container (ctx.context.docker_id),
    using the detected repository root as cwd.
    """
    docker_id = (getattr(ctx.context, "docker_id", "") or "").strip()
    if not docker_id:
        return "ERROR: docker_id is not set on CodeSearchContext"
    repo_root = _ensure_repo_root(ctx.context)

    tool_call_index = (
        ctx.context.trace_logger.next_tool_step()
        if ctx.context.trace_logger is not None
        else None
    )
    ctx.context.log_event(
        "tool_start",
        tool_call_index=tool_call_index,
        tool="bash",
        arguments={"command": command, "timeout_s": timeout_s},
    )
    try:
        # docker-py is blocking; run it in a thread so we don't block the event loop
        coro = asyncio.to_thread(_exec_in_container_sync, docker_id, command, repo_root)
        exec_result = await asyncio.wait_for(coro, timeout=timeout_s)
        ctx.context.log_event(
            "docker_exec",
            tool_call_index=tool_call_index,
            docker_exec_index=(
                ctx.context.trace_logger.next_docker_exec_step()
                if ctx.context.trace_logger is not None
                else None
            ),
            command=command,
            workdir=repo_root,
            timeout_s=timeout_s,
            exit_code=exec_result["exit_code"],
            elapsed_seconds=round(exec_result["elapsed_seconds"], 3),
            original_output_chars=exec_result["original_output_chars"],
            output_chars=exec_result["output_chars"],
            truncated=exec_result["truncated"],
            output_preview=exec_result["output"],
        )
        ctx.context.log_event(
            "tool_end",
            tool_call_index=tool_call_index,
            tool="bash",
            output_chars=exec_result["output_chars"],
            output_preview=exec_result["output"],
            error=False,
        )
        return exec_result["output"]
    except asyncio.TimeoutError:
        output = f"ERROR: command timed out after {timeout_s}s"
        ctx.context.log_event(
            "tool_end",
            tool_call_index=tool_call_index,
            tool="bash",
            output=output,
            error=True,
            error_type="timeout",
        )
        return output
    except NotFound:
        output = f"ERROR: container not found: {docker_id}"
        ctx.context.log_event(
            "tool_end",
            tool_call_index=tool_call_index,
            tool="bash",
            output=output,
            error=True,
            error_type="container_not_found",
        )
        return output
    except APIError as e:
        output = f"ERROR: docker API error: {e.explanation if hasattr(e, 'explanation') else str(e)}"
        ctx.context.log_event(
            "tool_end",
            tool_call_index=tool_call_index,
            tool="bash",
            output=output,
            error=True,
            error_type="docker_api_error",
        )
        return output
    except Exception as e:
        output = f"ERROR: unexpected error: {type(e).__name__}: {e}"
        ctx.context.log_event(
            "tool_end",
            tool_call_index=tool_call_index,
            tool="bash",
            output=output,
            error=True,
            error_type=type(e).__name__,
        )
        return output
