"""Typed model of the ``codex exec --json`` event stream.

Frozen against **codex-cli 0.147.0** by capturing a live run, not by reading the
documentation. The fixtures in ``tests/fixtures/codex/`` are that capture, byte
for byte, and the parser here is written to those bytes. Re-capture and re-pin
when the CLI version moves; a schema guessed from prose is how a backend starts
silently mis-measuring.

The stream is one JSON object per line::

    {"type": "thread.started", "thread_id": "01a0..."}
    {"type": "turn.started"}
    {"type": "item.started",   "item": {"id": "item_0", "type": "command_execution", ...}}
    {"type": "item.completed", "item": {"id": "item_0", "type": "command_execution", ...}}
    {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "..."}}
    {"type": "turn.completed", "usage": {...}}

Four properties of that stream drive every design choice below.

1. **Usage appears only on ``turn.completed``.** A ``turn.failed`` carries an
   ``error`` and *no* usage at all, so the tokens a failed turn consumed are not
   merely un-summed — they are unknowable from the stream. They are therefore
   recorded as unknown (``None``), never as ``0``. "Free" and "unmeasured" are
   different facts and must not be stored in the same field.

2. **Cached and reasoning counters are SUBSETS.** ``cached_input_tokens`` and
   ``cache_write_input_tokens`` are already inside ``input_tokens``, and
   ``reasoning_output_tokens`` is already inside ``output_tokens``. Adding them
   double-counts. :class:`TurnUsage` exposes ``uncached_input_tokens`` so callers
   never have to remember which way the arithmetic goes.

3. **The events carry no timestamps.** Not one field anywhere in the stream. Any
   timing an experiment reports is therefore the *reader's* arrival time, which
   is why capture must tee line by line as the lines arrive (see
   :mod:`chia.models.codex_capture`) rather than parse a buffer afterwards.

4. **Event and item types are two separate vocabularies.** ``type`` names the
   envelope (``item.completed``) and ``item.type`` names the payload
   (``command_execution``). Substring-matching one against the other — asking
   whether ``"tool" in type`` — is what makes a parser accept a spelling the CLI
   never emits while silently dropping one it does. Both vocabularies are
   matched exactly here, and anything unrecognized is preserved verbatim instead
   of being dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------------------
# The measured vocabularies. Exact strings, matched exactly.
# ---------------------------------------------------------------------------

#: Envelope types observed in 0.147.0.
EVENT_THREAD_STARTED = "thread.started"
EVENT_TURN_STARTED = "turn.started"
EVENT_TURN_COMPLETED = "turn.completed"
EVENT_TURN_FAILED = "turn.failed"
EVENT_ITEM_STARTED = "item.started"
EVENT_ITEM_UPDATED = "item.updated"
EVENT_ITEM_COMPLETED = "item.completed"
EVENT_ERROR = "error"

KNOWN_EVENT_TYPES = frozenset({
    EVENT_THREAD_STARTED,
    EVENT_TURN_STARTED,
    EVENT_TURN_COMPLETED,
    EVENT_TURN_FAILED,
    EVENT_ITEM_STARTED,
    EVENT_ITEM_UPDATED,
    EVENT_ITEM_COMPLETED,
    EVENT_ERROR,
})

#: ``item.type`` values observed in 0.147.0.
ITEM_COMMAND_EXECUTION = "command_execution"
ITEM_AGENT_MESSAGE = "agent_message"
ITEM_ERROR = "error"
ITEM_REASONING = "reasoning"
ITEM_FILE_CHANGE = "file_change"
ITEM_MCP_TOOL_CALL = "mcp_tool_call"
ITEM_WEB_SEARCH = "web_search"
ITEM_TODO_LIST = "todo_list"

#: Item types that represent a tool invocation of some kind.
TOOL_ITEM_TYPES = frozenset({
    ITEM_COMMAND_EXECUTION,
    ITEM_MCP_TOOL_CALL,
    ITEM_WEB_SEARCH,
    ITEM_FILE_CHANGE,
})

#: The five usage fields, in the CLI's own spelling. ``cache_write_input_tokens``
#: is the one a previous alias table missed (it looked for
#: ``cache_creation_input_tokens``, an Anthropic spelling), which silently
#: dropped every cache write.
USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def _opt_int(value: Any) -> int | None:
    """Coerce a reported count to ``int``, preserving *unreported* as ``None``.

    ``bool`` is rejected because ``isinstance(True, int)`` is true and a boolean
    in a token field means the payload is not what we think it is.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


@dataclass(frozen=True)
class TurnUsage:
    """Token usage for one turn, with unreported fields preserved as ``None``.

    ``reported`` distinguishes "the provider told us nothing" (a failed turn)
    from "the provider told us zero". Every consumer that turns tokens into
    money must branch on it.
    """

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    source_event: str = ""
    reported: bool = False

    @property
    def provider_reported(self) -> bool:
        """Spec-name alias for :attr:`reported` (the frozen contract §3 name)."""
        return self.reported

    @classmethod
    def unreported(cls, source_event: str) -> "TurnUsage":
        """A turn whose usage the provider never reported (e.g. ``turn.failed``)."""
        return cls(source_event=source_event, reported=False)

    @classmethod
    def from_payload(cls, usage: Any, *, source_event: str) -> "TurnUsage":
        if not isinstance(usage, dict):
            return cls.unreported(source_event)
        values = {name: _opt_int(usage.get(name)) for name in USAGE_FIELDS}
        return cls(
            **values,
            source_event=source_event,
            reported=any(v is not None for v in values.values()),
        )

    @property
    def uncached_input_tokens(self) -> int | None:
        """Input tokens that were neither a cache read nor a cache write.

        ``None`` when ``input_tokens`` is unknown. The cache subsets default to 0
        when absent — an unreported cache field means no cache activity was
        attributed, which is different from an unreported *total*. Clamped at 0
        so a provider inconsistency cannot produce a negative billable count.
        """
        if self.input_tokens is None:
            return None
        read = self.cached_input_tokens or 0
        write = self.cache_write_input_tokens or 0
        return max(self.input_tokens - read - write, 0)

    @property
    def non_reasoning_output_tokens(self) -> int | None:
        """Output tokens that were not reasoning tokens."""
        if self.output_tokens is None:
            return None
        return max(self.output_tokens - (self.reasoning_output_tokens or 0), 0)

    def as_dict(self) -> dict[str, Any]:
        """Serializable form. ``None`` is preserved, never coerced to ``0``."""
        return {
            **{name: getattr(self, name) for name in USAGE_FIELDS},
            "uncached_input_tokens": self.uncached_input_tokens,
            "source_event": self.source_event,
            "reported": self.reported,
        }


def add_usage(left: TurnUsage, right: TurnUsage) -> TurnUsage:
    """Sum two usage records field-wise, keeping unknown-ness contagious per field.

    ``None + 5`` is ``5``: a turn that reported nothing must not erase a turn
    that reported something. But the result's ``reported`` is true only if some
    field actually came from the provider, so a sum over exclusively-unreported
    turns stays unreported rather than becoming a confident zero.
    """
    out: dict[str, Any] = {}
    for name in USAGE_FIELDS:
        a, b = getattr(left, name), getattr(right, name)
        out[name] = a if b is None else (b if a is None else a + b)
    sources = [s for s in (left.source_event, right.source_event) if s]
    return TurnUsage(
        **out,
        source_event="+".join(sources),
        reported=left.reported or right.reported,
    )


def sum_usage(items: Iterable[TurnUsage]) -> TurnUsage:
    """Fold :func:`add_usage` over *items*; empty yields an unreported record."""
    total = TurnUsage.unreported("")
    for item in items:
        total = add_usage(total, item)
    return total


@dataclass
class ToolCall:
    """One tool invocation, assembled from its ``item.started``/``item.completed`` pair."""

    item_id: str
    item_type: str
    command: str | None = None
    exit_code: int | None = None
    status: str | None = None
    aggregated_output: str | None = None
    #: ``file_change`` items carry a list of ``{path, kind}`` dicts.
    changes: list[Any] | None = None
    started_at: str | None = None
    completed_at: str | None = None
    #: Anything in the item payload we do not model, kept so nothing is lost.
    extra: dict[str, Any] = field(default_factory=dict)
    #: The last-seen item payload verbatim, so a consumer can recover any field.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """Spec-name alias for :attr:`item_type` (frozen contract §3 name)."""
        return self.item_type

    @property
    def failed(self) -> bool:
        """True only when an exit code was reported and it was non-zero."""
        return self.exit_code is not None and self.exit_code != 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "command": self.command,
            "exit_code": self.exit_code,
            "status": self.status,
            "changes": self.changes,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_bytes": len(self.aggregated_output or ""),
            **({"extra": self.extra} if self.extra else {}),
        }


_MODELLED_ITEM_KEYS = frozenset({
    "id", "type", "command", "exit_code", "status", "aggregated_output",
    "text", "changes",
})


@dataclass
class ParsedStream:
    """Everything a single ``codex exec`` invocation's stream tells us.

    Deliberately records *observations*, not judgements: it says whether a
    ``turn.completed`` was seen and what usage came with it, and leaves "did this
    attempt succeed" to the caller, which also knows the exit code and whether
    the process was killed.
    """

    thread_id: str | None = None
    turns_started: int = 0
    turns_completed: int = 0
    turns_failed: int = 0
    turn_usage: list[TurnUsage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    agent_messages: list[str] = field(default_factory=list)
    reasoning_texts: list[str] = field(default_factory=list)
    #: ``error`` envelopes and ``item.type == "error"`` payloads, in order.
    errors: list[str] = field(default_factory=list)
    #: Envelope types seen that this version does not model.
    unknown_event_types: list[str] = field(default_factory=list)
    #: ``item.type`` values seen that this version does not model.
    unknown_item_types: list[str] = field(default_factory=list)
    #: Lines that were not valid JSON, verbatim (truncated by the caller if needed).
    unparsed_lines: list[str] = field(default_factory=list)
    events_seen: int = 0

    @property
    def total_usage(self) -> TurnUsage:
        """Usage summed across turns. Unreported turns do not contribute zeros."""
        return sum_usage(self.turn_usage)

    @property
    def usage_complete(self) -> bool:
        """True when every turn that started also reported usage.

        False means at least one turn's tokens are unaccounted for — the honest
        signal that a spend figure derived from this stream is a lower bound.
        """
        if self.turns_started == 0:
            return False
        reported = sum(1 for u in self.turn_usage if u.reported)
        return reported >= self.turns_started

    @property
    def final_text(self) -> str:
        """The last agent message, which is what ``--output-last-message`` writes."""
        return self.agent_messages[-1] if self.agent_messages else ""


def parse_line(line: str) -> dict[str, Any] | None:
    """Parse one JSONL line, or ``None`` when it is not a JSON object.

    A JSON scalar (``"5"``, ``"null"``) is not an event and is reported as
    unparsed rather than being wrapped into a fake event.
    """
    text = line.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_events(events: Iterable[dict[str, Any]]) -> ParsedStream:
    """Fold a sequence of already-decoded event dicts into a :class:`ParsedStream`."""
    out = ParsedStream()
    pending: dict[str, ToolCall] = {}

    for event in events:
        out.events_seen += 1
        etype = event.get("type")
        if not isinstance(etype, str):
            out.unknown_event_types.append(repr(etype))
            continue

        if etype == EVENT_THREAD_STARTED:
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                out.thread_id = thread_id
        elif etype == EVENT_TURN_STARTED:
            out.turns_started += 1
        elif etype == EVENT_TURN_COMPLETED:
            out.turns_completed += 1
            out.turn_usage.append(
                TurnUsage.from_payload(event.get("usage"), source_event=etype)
            )
        elif etype == EVENT_TURN_FAILED:
            out.turns_failed += 1
            # No usage is carried here; record the turn as unmeasured, not free.
            out.turn_usage.append(TurnUsage.unreported(etype))
            out.errors.append(_error_text(event.get("error")))
        elif etype == EVENT_ERROR:
            out.errors.append(_error_text(event.get("message") or event))
        elif etype in (EVENT_ITEM_STARTED, EVENT_ITEM_UPDATED, EVENT_ITEM_COMPLETED):
            _absorb_item(event, etype, out, pending)
        else:
            out.unknown_event_types.append(etype)

    # A tool call whose completion never arrived (the process died mid-command)
    # is still a tool call that ran. Keep it, with status left as observed.
    out.tool_calls.extend(pending.values())
    out.tool_calls.sort(key=lambda t: (t.started_at or "", t.item_id))
    return out


def _absorb_item(
    event: dict[str, Any],
    etype: str,
    out: ParsedStream,
    pending: dict[str, ToolCall],
) -> None:
    item = event.get("item")
    if not isinstance(item, dict):
        out.unknown_item_types.append(repr(item))
        return
    itype = item.get("type")
    if not isinstance(itype, str):
        out.unknown_item_types.append(repr(itype))
        return
    item_id = str(item.get("id") or f"anon_{out.events_seen}")

    if itype == ITEM_AGENT_MESSAGE:
        if etype == EVENT_ITEM_COMPLETED:
            text = item.get("text")
            out.agent_messages.append(text if isinstance(text, str) else "")
        return

    if itype == ITEM_REASONING:
        if etype == EVENT_ITEM_COMPLETED:
            text = item.get("text")
            if isinstance(text, str) and text:
                out.reasoning_texts.append(text)
        return

    if itype == ITEM_ERROR:
        out.errors.append(_error_text(item.get("message") or item))
        return

    if itype in TOOL_ITEM_TYPES:
        call = pending.pop(item_id, None) or ToolCall(item_id=item_id, item_type=itype)
        changes = item.get("changes")
        call = replace(
            call,
            command=_first_str(item.get("command"), call.command),
            exit_code=_opt_int(item.get("exit_code")) if item.get("exit_code") is not None else call.exit_code,
            status=_first_str(item.get("status"), call.status),
            aggregated_output=_first_str(item.get("aggregated_output"), call.aggregated_output),
            changes=changes if isinstance(changes, list) else call.changes,
            extra={**call.extra, **{k: v for k, v in item.items() if k not in _MODELLED_ITEM_KEYS}},
            raw={**call.raw, **item},
        )
        if etype == EVENT_ITEM_COMPLETED:
            out.tool_calls.append(call)
        else:
            pending[item_id] = call
        return

    out.unknown_item_types.append(itype)


def _first_str(value: Any, fallback: str | None) -> str | None:
    return value if isinstance(value, str) else fallback


def _error_text(value: Any) -> str:
    """Render an error payload as text without losing its structure.

    Codex nests a JSON *string* inside ``error.message`` for HTTP failures, so
    the useful text is one level deeper than it looks; the whole payload is kept
    when it is not a plain string.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def iter_lines(text: str) -> Iterator[dict[str, Any] | None]:
    """Yield a parsed event per line of *text* (``None`` for unparsable lines)."""
    for line in text.splitlines():
        yield parse_line(line)


def parse_stream_text(text: str) -> ParsedStream:
    """Parse a whole JSONL blob, recording unparsable lines rather than dropping them.

    Used for replay of a captured ``events.raw.jsonl``; live capture feeds
    :func:`parse_events` incrementally instead.
    """
    events: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        event = parse_line(line)
        if event is None:
            unparsed.append(line)
        else:
            events.append(event)
    out = parse_events(events)
    out.unparsed_lines.extend(unparsed)
    return out
