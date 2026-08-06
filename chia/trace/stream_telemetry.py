"""Derive per-call activity telemetry from a Claude Code event stream.

chia already runs the CLI with ``--output-format stream-json --verbose``, so every tool
call, its result, and its wall-clock duration arrive on stdout. Until now
:meth:`ClaudeCodeLLM._process_event_line` rendered each line into a human-readable
transcript and dropped it, and the only thing that survived a call was one folded usage
dict. Downstream, that made three questions unanswerable from a chia run:

* what did the agent spend its time *doing* (read / write / shell / waiting on a tool),
* how many tool calls did it make, and how many of those failed,
* which model served which part of the run.

None of that needed new capture — the events were already in hand — and none of it needed
a new parser: :mod:`aet.tracking.claude_stream` parses exactly this format, and its
``parse_timestamped_stream`` docstring describes the very loop chia's ``drain_stdout``
already runs. This module is the wiring between the two.

**What crosses the process boundary.** :meth:`ClaudeCodeLLM.prompt` may run on any Ray
worker, so whatever this produces travels back to the driver inside the profiler's
metadata dict. It is therefore a *summary*, derived where the data already is: activity
bands (interval + category), tool counts, and per-model token totals. Raw event lines,
tool inputs and tool outputs never leave the worker — they are the parts that carry
prompt text, file contents and credentials, and none of them are needed to draw a figure.

Everything degrades to ``{}`` when aet is not installed, because the ``aet`` extra is
optional and a missing figure must never fail a run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Key under which the summary travels in the profiler metadata dict. Namespaced so it
# cannot collide with a usage field: everything else in that dict is a scalar.
METADATA_KEY = "stream_telemetry"

# A band shorter than this is dropped. The activity view is a share-of-time stack, and a
# sub-millisecond interval contributes nothing to it while still costing a row in every
# downstream artifact.
MIN_BAND_S = 1e-3


def _classifier(config: Optional[Any] = None):
    """aet's activity classifier, or ``None`` when aet is unavailable.

    The classifier is configurable on purpose (aet's own docstring is explicit that "a Bash
    call running verilator is a long tool-wait" is repo-specific knowledge that must live in
    a config object). chia passes no config, so callers get aet's generic defaults; a suite
    with domain rules supplies its own.
    """
    try:
        from aet.trajectory.classify import ActivityClassifier
    except Exception:
        return None
    return ActivityClassifier(config)


def bands_from_tool_calls(tool_calls: Sequence[Any], *, config: Optional[Any] = None,
                          t_offset: float = 0.0) -> List[dict]:
    """Activity bands for *tool_calls*, as plain dicts ready for :class:`aet.ActivityBand`.

    :param tool_calls: ``aet.tracking.claude_stream.ToolCall`` objects.
    :param config: Optional ``ActivityConfig``; aet's defaults when omitted.
    :param t_offset: Seconds to add to each band, to place a per-call stream on a run-wide
        clock. A run is many ``prompt`` calls and each stream starts at its own zero.
    :rtype: List[dict]

    A zero-duration call still produces no band: it is an instant, not an interval, and
    drawing it as one would put area under the activity curve that no time was spent in.
    """
    clf = _classifier(config)
    if clf is None:
        return []
    out: List[dict] = []
    for tc in tool_calls:
        duration = float(getattr(tc, "duration_s", 0.0) or 0.0)
        if duration < MIN_BAND_S:
            continue
        start = float(getattr(tc, "start_offset_s", 0.0) or 0.0) + t_offset
        name = str(getattr(tc, "name", "") or "")
        try:
            category, weight = clf.classify(name, getattr(tc, "input", None) or {})
        except Exception as exc:                      # a classifier rule is caller-supplied
            logger.debug("activity classification failed for %r: %s", name, exc)
            continue
        out.append({
            "t0_s": start,
            "t1_s": start + duration,
            "category": category,
            # The tool NAME is kept (it is a fixed vocabulary: Read, Bash, Edit...). The tool
            # INPUT is not: that is where file contents and prompt text live.
            "tool_name": name,
            "weight": float(weight),
            "is_error": bool(getattr(tc, "is_error", False)),
        })
    return out


def summarize_stream(events: Sequence[Tuple[float, str]], *,
                     config: Optional[Any] = None) -> Dict[str, Any]:
    """Summarise one call's ``(monotonic_s, raw_json_line)`` events.

    :param events: The stream as collected in ``drain_stdout``.
    :type events: Sequence[Tuple[float, str]]
    :rtype: Dict[str, Any]

    Returns ``{}`` when aet is absent, when the stream carried no events, or when parsing
    fails — a telemetry summary is never worth raising into a caller's result path. The
    returned dict is JSON-serialisable and small: one entry per tool call and per model,
    not per event.
    """
    if not events:
        return {}
    try:
        from aet.tracking.claude_stream import parse_timestamped_stream
    except Exception:
        return {}
    try:
        parsed = parse_timestamped_stream(list(events))
    except Exception as exc:
        logger.debug("stream telemetry unavailable (parse failed): %s", exc)
        return {}

    tool_calls = list(getattr(parsed, "tool_calls", ()) or ())
    summary: Dict[str, Any] = {
        "bands": bands_from_tool_calls(tool_calls, config=config),
        "tool_calls": len(tool_calls),
        "tool_errors": sum(1 for tc in tool_calls if getattr(tc, "is_error", False)),
        "tools_used": sorted({str(getattr(tc, "name", "")) for tc in tool_calls
                              if getattr(tc, "name", "")}),
        "num_turns": int(getattr(parsed, "num_turns", 0) or 0),
        # ``has_result_event`` False means the stream was cut short, so a consumer knows the
        # counts are a floor rather than a total.
        "complete": bool(getattr(parsed, "has_result_event", False)),
    }

    per_model = []
    try:
        for mu in parsed.per_model_usage():
            per_model.append({
                "model": str(getattr(mu, "model", "") or ""),
                "input_tokens": int(getattr(mu, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(mu, "output_tokens", 0) or 0),
                "cache_read_input_tokens": int(getattr(mu, "cache_read_input_tokens", 0) or 0),
                "cache_creation_input_tokens":
                    int(getattr(mu, "cache_creation_input_tokens", 0) or 0),
                "activity_share": float(getattr(mu, "activity_share", 0.0) or 0.0),
            })
    except Exception as exc:
        logger.debug("per-model usage unavailable: %s", exc)
    if per_model:
        # This is the answer to "which tier served which call". chia's own metadata folds
        # every model the CLI touched into one `model` field; this does not.
        summary["per_model"] = per_model
    return summary


def merge_into_metadata(meta: Optional[dict], summary: Dict[str, Any]) -> dict:
    """Attach *summary* to a profiler metadata dict under :data:`METADATA_KEY`.

    Returns the dict (created when *meta* is ``None``). An empty summary is not attached,
    so a run without aet installed produces metadata byte-identical to before this existed.
    """
    if meta is None:
        meta = {}
    if summary:
        meta[METADATA_KEY] = summary
    return meta
