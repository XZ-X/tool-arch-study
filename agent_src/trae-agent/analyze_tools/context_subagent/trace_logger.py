from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel


_COUNTERS_LOCK = threading.Lock()
_SESSION_COUNTERS: dict[str, int] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def _next_invocation_index(docker_id: str) -> int:
    with _COUNTERS_LOCK:
        next_index = _SESSION_COUNTERS.get(docker_id, 0) + 1
        _SESSION_COUNTERS[docker_id] = next_index
        return next_index


def _jsonable(value: Any, *, max_string: int = 20000) -> Any:
    if isinstance(value, str):
        if len(value) > max_string:
            return {
                "text": value[:max_string],
                "truncated": True,
                "original_chars": len(value),
            }
        return value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"), max_string=max_string)
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v, max_string=max_string)
            for k, v in value.items()
            if not _looks_sensitive(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, max_string=max_string) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"), max_string=max_string)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(), max_string=max_string)
        except Exception:
            pass
    return repr(value)


def _looks_sensitive(key: str) -> bool:
    lowered = key.lower()
    return "api_key" in lowered or "authorization" in lowered or "secret" in lowered


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return ""
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


class SearchSubagentTraceLogger:
    def __init__(
        self,
        *,
        docker_id: str,
        query: str,
        model_short_name: str,
        model_name: str,
        base_url: str,
        log_root: str | Path = "analyze_tools/context_subagent/logs",
    ):
        self.docker_id = docker_id
        self.session_id = docker_id[:12] if docker_id else "unknown"
        self.query = query
        self.model_short_name = model_short_name
        self.model_name = model_name
        self.base_url = _safe_url(base_url)
        self.invocation_index = _next_invocation_index(docker_id)
        self.started_at = time.time()
        date_dir = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query_hash = _short_hash(query)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.trace_id = (
            f"{timestamp}_{_safe_slug(self.session_id)}_"
            f"{self.invocation_index:04d}_{query_hash}"
        )

        self.log_root = Path(log_root)
        self.trace_dir = self.log_root / date_dir / "traces"
        self.session_dir = self.log_root / date_dir / "sessions"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.trace_path = self.trace_dir / f"{self.trace_id}.jsonl"
        self.summary_path = self.trace_dir / f"{self.trace_id}.summary.json"
        self.session_index_path = self.session_dir / f"{_safe_slug(self.session_id)}.jsonl"

        self.event_index = 0
        self.llm_call_index = 0
        self.tool_call_index = 0
        self.docker_exec_index = 0
        self._event_lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_input_tokens = 0
        self.total_reasoning_tokens = 0

        self.log_event(
            "request_start",
            docker_id=docker_id,
            query=query,
            model_short_name=model_short_name,
            model=model_name,
            base_url=self.base_url,
            invocation_index=self.invocation_index,
        )

    def log_event(self, event_type: str, **payload: Any) -> None:
        with self._event_lock:
            self.event_index += 1
            record = {
                "event_index": self.event_index,
                "timestamp": _utc_now(),
                "type": event_type,
                "trace_id": self.trace_id,
                "session_id": self.session_id,
                **_jsonable(payload),
            }
            with self.trace_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def next_llm_step(self) -> int:
        self.llm_call_index += 1
        return self.llm_call_index

    def next_tool_step(self) -> int:
        self.tool_call_index += 1
        return self.tool_call_index

    def next_docker_exec_step(self) -> int:
        self.docker_exec_index += 1
        return self.docker_exec_index

    def add_usage(self, usage: Any) -> dict[str, int]:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)
        reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_input_tokens += cached_tokens
        self.total_reasoning_tokens += reasoning_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_read_input_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    def close(self, *, status: str, final_output: str | None = None, **payload: Any) -> None:
        elapsed_seconds = time.time() - self.started_at
        summary = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "docker_id": self.docker_id,
            "query": self.query,
            "status": status,
            "started_at": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            "ended_at": _utc_now(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "model_short_name": self.model_short_name,
            "model": self.model_name,
            "base_url": self.base_url,
            "invocation_index": self.invocation_index,
            "llm_calls": self.llm_call_index,
            "tool_calls": self.tool_call_index,
            "docker_exec_calls": self.docker_exec_index,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_input_tokens": self.total_cached_input_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "trace_path": self.trace_path.as_posix(),
            "summary_path": self.summary_path.as_posix(),
            **_jsonable(payload),
        }
        if final_output is not None:
            summary["final_output"] = _jsonable(final_output)

        self.log_event("request_end", **summary)
        with self.summary_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        with self.session_index_path.open("a") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
