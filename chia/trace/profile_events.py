"""Stable, privacy-safe events for external agent profilers.

The ordinary Chia profiler trace remains the transport: these records are
JSON-serializable objects in the same JSONL file.  The schema deliberately
contains identifiers, timings and counters, but never prompts, tool arguments,
tool results, file contents, environment values, or credentials.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator, Optional

PROFILE_SCHEMA_VERSION = "1.0"
PROFILE_EVENT_TYPES = frozenset({"llm_request", "tool_activity", "agent_start", "agent_end"})


@dataclass(frozen=True)
class ProfileContext:
    """Identity inherited by telemetry emitted inside one Chia call."""

    run_id: str = ""
    call_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    agent_id: str = ""
    parent_agent_id: str = ""


def new_request_id() -> str:
    """Return an opaque request id without encoding prompt or user data."""

    return uuid.uuid4().hex


def base_event(event_type: str, context: ProfileContext, **fields) -> dict:
    """Build a versioned event, dropping unset values for compact JSONL."""

    if event_type not in PROFILE_EVENT_TYPES:
        raise ValueError(f"unsupported profile event type: {event_type}")
    event = {
        "schema": "chia.agent_profile",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "type": event_type,
        "ts": time.time(),
        **asdict(context),
        **fields,
    }
    return {key: value for key, value in event.items() if value not in (None, "")}


def llm_request_event(
    context: ProfileContext,
    *,
    provider: str,
    model: str,
    backend: str,
    status: str,
    request_id: str = "",
    attempt: int = 1,
    duration_s: Optional[float] = None,
    ttft_s: Optional[float] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    cost_usd: Optional[float] = None,
    cost_source: str = "unavailable",
    billing_mode: str = "per_token",
    retry: bool = False,
) -> dict:
    """Build one inference attempt record using subset-safe token buckets."""

    return base_event(
        "llm_request", context, provider=provider, model=model, backend=backend,
        status=status, request_id=request_id or new_request_id(), attempt=attempt,
        duration_s=duration_s, ttft_s=ttft_s, input_tokens=int(input_tokens),
        output_tokens=int(output_tokens), cache_read_tokens=int(cache_read_tokens),
        cache_write_tokens=int(cache_write_tokens), reasoning_tokens=int(reasoning_tokens),
        cost_usd=cost_usd, cost_source=cost_source, billing_mode=billing_mode,
        retry=bool(retry),
    )


def tool_activity_event(
    context: ProfileContext,
    *,
    tool_name: str,
    category: str = "tool",
    status: str = "completed",
    duration_s: Optional[float] = None,
    request_id: str = "",
) -> dict:
    """Build a tool span without tool inputs, outputs, commands, or file paths."""

    return base_event(
        "tool_activity", context, tool_name=tool_name, category=category,
        status=status, duration_s=duration_s, request_id=request_id or new_request_id(),
    )


def agent_event(
    event_type: str,
    context: ProfileContext,
    *,
    name: str,
    status: str = "running",
    model: str = "",
) -> dict:
    """Build an agent lifecycle record."""

    if event_type not in ("agent_start", "agent_end"):
        raise ValueError("agent event must be agent_start or agent_end")
    return base_event(event_type, context, name=name, status=status, model=model)


@contextmanager
def agent_scope(profiler, *, name: str, model: str = "", agent_id: str = "",
                parent_agent_id: str = "") -> Iterator[str]:
    """Emit balanced agent lifecycle events around an orchestration scope."""

    resolved = agent_id or uuid.uuid4().hex
    previous = profiler.profile_context()
    context = ProfileContext(**{
        **asdict(previous), "agent_id": resolved, "parent_agent_id": parent_agent_id,
    })
    profiler.set_profile_context(context)
    profiler.log_profile_event(agent_event("agent_start", context, name=name, model=model))
    status = "completed"
    try:
        yield resolved
    except BaseException:
        status = "failed"
        raise
    finally:
        profiler.log_profile_event(agent_event(
            "agent_end", context, name=name, model=model, status=status,
        ))
        profiler.set_profile_context(previous)
