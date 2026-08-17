"""Tests for the streaming Codex recorder (:class:`chia.trace.aet_sink.CodexAetRecorder`).

Unlike the aggregate ``record_run_usage`` path, this exercises the incremental
per-turn / per-tool / per-attempt contract. The durable local JSONL is written
independently of whether aet is importable, so these run fully offline; the aet
mirror (when present) is best-effort and never asserted here.
"""

from __future__ import annotations

import json

from chia.trace.aet_sink import CodexAetRecorder
from chia.models.codex_events import ToolCall, TurnUsage
from chia.models.codex_records import CodexAttempt, CodexRunResult


def _lines(path):
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def test_recorder_streams_turn_tool_attempt_to_local_jsonl(tmp_path):
    rec = CodexAetRecorder(run_dir=tmp_path, run_id="r1", model="gpt-x", enabled=True)
    agent = tmp_path / "agent"

    rec.record_thread("01a0-THREAD")
    rec.record_turn(TurnUsage(
        input_tokens=100, cached_input_tokens=20, cache_write_input_tokens=5,
        output_tokens=10, reasoning_output_tokens=3,
        source_event="turn.completed", reported=True,
    ))
    rec.record_tool(ToolCall(item_id="i0", item_type="command_execution",
                             command="ls", exit_code=0, status="completed"))
    rec.record_attempt(CodexAttempt(index=0, thread_id="01a0-THREAD"))
    rec.finish(CodexRunResult(thread_id="01a0-THREAD", status="completed"), status="completed")

    usage = _lines(agent / "usage.jsonl")
    assert usage[0]["input_tokens"] == 100
    assert usage[0]["uncached_input_tokens"] == 100 - 20 - 5  # subset math preserved
    tools = _lines(agent / "tools.jsonl")
    assert tools[0]["item_type"] == "command_execution"
    attempts = _lines(agent / "attempts.jsonl")
    assert attempts[0]["index"] == 0
    assert (agent / "session.jsonl").exists()
    assert (agent / "run_result.json").exists()


def test_recorder_skips_unreported_turn(tmp_path):
    rec = CodexAetRecorder(run_dir=tmp_path, run_id="r2", enabled=True)
    # A failed turn carries no usage -> nothing to record (unknown != zero).
    assert rec.record_turn(TurnUsage.unreported("turn.failed")) is False
    assert not (tmp_path / "agent" / "usage.jsonl").exists()


def test_recorder_disabled_is_noop(tmp_path):
    rec = CodexAetRecorder(run_dir=tmp_path, run_id="r3", enabled=False)
    assert rec.record_thread("T") is False
    assert rec.record_turn(TurnUsage(input_tokens=1, output_tokens=1, reported=True)) is False
    assert not (tmp_path / "agent").exists()


def test_recorder_fail_open_without_run_dir():
    # No run dir resolvable -> everything no-ops, never raises.
    rec = CodexAetRecorder(run_dir=None, run_id="r4", enabled=True)
    assert rec.record_turn(TurnUsage(input_tokens=1, output_tokens=1, reported=True)) is False
    assert rec.finish() is False


def test_recorder_multiple_turns_accumulate_in_order(tmp_path):
    rec = CodexAetRecorder(run_dir=tmp_path, run_id="r5", enabled=True)
    rec.record_turn(TurnUsage(input_tokens=100, output_tokens=10, source_event="turn.completed", reported=True))
    rec.record_turn(TurnUsage(input_tokens=200, output_tokens=20, source_event="turn.completed", reported=True))
    usage = _lines(tmp_path / "agent" / "usage.jsonl")
    assert [u["input_tokens"] for u in usage] == [100, 200]
