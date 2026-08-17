"""Typed records for a Codex run — the frozen-contract §3 shapes.

These are the durable, serializable facts a Chia Codex invocation produces:
one :class:`CodexAttempt` per subprocess launch (including every failed or
timed-out retry) and one :class:`CodexRunResult` aggregating them. They are the
hand-off surface the AET importer reads and the streaming sink records against.

The two token/tool shapes named in §3 — ``CodexTurnUsage`` and
``CodexToolCall`` — are the same objects the stream parser already produces
(:class:`chia.models.codex_events.TurnUsage` / ``ToolCall``); they are
re-exported here under their contract names so there is exactly one parser and
one definition of "how a turn's tokens add up".

Two invariants the whole design turns on:

* **Every attempt is kept, with its usage.** A retry never erases the tokens an
  earlier failed attempt already burned (contract §5: "charge every attempt the
  provider reports usage for"). :meth:`CodexRunResult.total_usage` sums across
  *all* attempts, and unreported turns contribute unknown-ness, not zero.
* **Unknown is not zero.** A turn whose usage never arrived (a ``turn.failed``)
  is ``reported=False`` and stays out of any confident total — see
  :mod:`chia.models.codex_events`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chia.models.codex_events import (
    ParsedStream,
    ToolCall,
    TurnUsage,
    sum_usage,
)

# The frozen-contract §3 names are the stream parser's own types. One parser,
# one definition of the arithmetic.
CodexTurnUsage = TurnUsage
CodexToolCall = ToolCall

#: Argv tokens whose *following* value is a secret to redact from a stored argv.
_SECRET_FLAG_VALUES = frozenset({
    "--api-key", "--apikey", "--token", "--auth", "--authorization",
    "--openai-api-key", "--bearer", "--password",
})
#: Argv tokens that are themselves secret-ish and should be masked wholesale.
_SECRET_SUBSTRINGS = ("api_key", "apikey", "authorization", "bearer", "secret", "token=")


def redact_argv(argv: list[str] | tuple[str, ...]) -> list[str]:
    """Return *argv* with obvious secrets masked (contract §7: no secrets stored).

    Masks the value after a known secret flag, and any single token that itself
    looks like ``key=secret``. Conservative by construction: it never drops a
    token (so the command stays reproducible in shape), only masks values.
    """
    out: list[str] = []
    mask_next = False
    for tok in argv:
        if mask_next:
            out.append("***REDACTED***")
            mask_next = False
            continue
        low = tok.lower()
        if low in _SECRET_FLAG_VALUES:
            out.append(tok)
            mask_next = True
            continue
        if "=" in tok and any(s in low for s in _SECRET_SUBSTRINGS):
            key, _, _ = tok.partition("=")
            out.append(f"{key}=***REDACTED***")
            continue
        out.append(tok)
    return out


@dataclass
class CodexAttempt:
    """One ``codex exec`` subprocess launch — kept even when it failed.

    ``turns`` and ``tools`` come straight from the streamed events; ``timeout``,
    ``signal`` and ``exit_code`` come from the process; ``failure_class`` is the
    caller's classification (rate_limit, auth, server, timeout, ...), left
    ``None`` on success.
    """

    index: int
    cmd_argv: list[str] = field(default_factory=list)
    thread_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    signal: str | None = None
    timeout: bool = False
    retry_reason: str | None = None
    turns: list[CodexTurnUsage] = field(default_factory=list)
    tools: list[CodexToolCall] = field(default_factory=list)
    raw_event_path: str | None = None
    stderr_path: str | None = None
    final_output: str | None = None
    agent_messages: list[str] = field(default_factory=list)
    failure_class: str | None = None

    @property
    def usage(self) -> CodexTurnUsage:
        """This attempt's tokens summed across its turns (unknown stays unknown)."""
        return sum_usage(self.turns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cmd_argv": list(self.cmd_argv),
            "thread_id": self.thread_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "timeout": self.timeout,
            "retry_reason": self.retry_reason,
            "turns": [t.as_dict() for t in self.turns],
            "tools": [t.as_dict() for t in self.tools],
            "raw_event_path": self.raw_event_path,
            "stderr_path": self.stderr_path,
            "final_output": self.final_output,
            "agent_messages": list(self.agent_messages),
            "failure_class": self.failure_class,
        }


@dataclass
class CodexRunResult:
    """All attempts of one logical Codex call plus the resolved run metadata."""

    thread_id: str | None = None
    attempts: list[CodexAttempt] = field(default_factory=list)
    resolved_model: str | None = None
    requested_model: str | None = None
    agent_version: str = ""
    status: str = "unknown"
    active_wall_s: float = 0.0
    prompt_sha256: str = ""
    price_snapshot_id: str | None = None

    @property
    def total_usage(self) -> CodexTurnUsage:
        """Usage summed across every attempt — retries included (contract §5)."""
        return sum_usage(t for a in self.attempts for t in a.turns)

    @property
    def usage_complete(self) -> bool:
        """True only when every turn across every attempt reported its tokens."""
        turns = [t for a in self.attempts for t in a.turns]
        return bool(turns) and all(t.reported for t in turns)

    @property
    def final_output(self) -> str:
        """Last agent message of the last attempt that produced one."""
        for attempt in reversed(self.attempts):
            if attempt.agent_messages:
                return attempt.agent_messages[-1]
            if attempt.final_output:
                return attempt.final_output
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "attempts": [a.as_dict() for a in self.attempts],
            "resolved_model": self.resolved_model,
            "requested_model": self.requested_model,
            "agent_version": self.agent_version,
            "status": self.status,
            "active_wall_s": self.active_wall_s,
            "prompt_sha256": self.prompt_sha256,
            "price_snapshot_id": self.price_snapshot_id,
            "total_usage": self.total_usage.as_dict(),
            "usage_complete": self.usage_complete,
        }


def attempt_from_parsed(
    parsed: ParsedStream,
    *,
    index: int,
    cmd_argv: list[str] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    exit_code: int | None = None,
    signal: str | None = None,
    timeout: bool = False,
    retry_reason: str | None = None,
    raw_event_path: str | None = None,
    stderr_path: str | None = None,
    final_output: str | None = None,
    failure_class: str | None = None,
) -> CodexAttempt:
    """Build a :class:`CodexAttempt` from a parsed stream plus process facts."""
    return CodexAttempt(
        index=index,
        cmd_argv=redact_argv(cmd_argv or []),
        thread_id=parsed.thread_id,
        started_at=started_at,
        ended_at=ended_at,
        exit_code=exit_code,
        signal=signal,
        timeout=timeout,
        retry_reason=retry_reason,
        turns=list(parsed.turn_usage),
        tools=list(parsed.tool_calls),
        raw_event_path=raw_event_path,
        stderr_path=stderr_path,
        final_output=final_output if final_output is not None else parsed.final_text,
        agent_messages=list(parsed.agent_messages),
        failure_class=failure_class,
    )
