"""Parser + typed-record tests for the ``codex exec --json`` stream (0.147.0).

The two ``real_*`` fixtures are byte-for-byte captures of the installed
codex-cli 0.147.0; they are ground truth. The synthetic fixtures cover the paths
a single probe cannot (multi-turn, null cache-write, unknown event/item types,
error/failed turns, a trailing non-JSON line, and same-thread resume).
"""

from __future__ import annotations

import pathlib

from chia.models import codex_events as ev
from chia.models.codex_records import (
    CodexAttempt,
    CodexRunResult,
    CodexToolCall,
    CodexTurnUsage,
    attempt_from_parsed,
    redact_argv,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "codex"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


# --------------------------------------------------------------------------
# Real captures — ground truth for the frozen contract.
# --------------------------------------------------------------------------

def test_real_simple_agent_message_parses_with_subset_accounting():
    parsed = ev.parse_stream_text(_load("real_simple_agent_message.jsonl"))
    assert parsed.thread_id == "01a01160-7e52-7153-a3f1-a3ee492ab99e"
    assert parsed.turns_started == 1
    assert parsed.turns_completed == 1
    assert parsed.final_text == "PONG"

    usage = parsed.total_usage
    assert usage.input_tokens == 17713
    assert usage.cached_input_tokens == 9984
    assert usage.output_tokens == 6
    # cached is a SUBSET of input; uncached = input - cache_read - cache_write.
    assert usage.uncached_input_tokens == 17713 - 9984 - 0
    assert usage.provider_reported is True


def test_real_tool_and_filechange_parses_tools_and_usage():
    parsed = ev.parse_stream_text(_load("real_tool_and_filechange.jsonl"))
    assert parsed.thread_id == "01a01161-68a0-71e3-9afb-6ec7a839f084"
    assert parsed.final_text == "FINISHED"
    assert parsed.agent_messages[0].startswith("I")  # first agent_message

    kinds = [t.kind for t in parsed.tool_calls]
    assert "file_change" in kinds and "command_execution" in kinds

    file_change = next(t for t in parsed.tool_calls if t.kind == "file_change")
    assert file_change.changes and file_change.changes[0]["kind"] == "add"

    cmd = next(t for t in parsed.tool_calls if t.kind == "command_execution")
    assert cmd.exit_code == 0 and cmd.status == "completed"
    assert cmd.aggregated_output == "done\n"

    usage = parsed.total_usage
    assert usage.input_tokens == 53494
    assert usage.cached_input_tokens == 37120
    assert usage.uncached_input_tokens == 53494 - 37120


# --------------------------------------------------------------------------
# Synthetic full-coverage fixture.
# --------------------------------------------------------------------------

def test_synthetic_full_covers_all_paths():
    parsed = ev.parse_stream_text(_load("synthetic_full.jsonl"))
    # Two turns, both completed, plus a reasoning item and two tools.
    assert parsed.turns_started == 2
    assert parsed.turns_completed == 2
    assert parsed.reasoning_texts and "inspect the file" in parsed.reasoning_texts[0]
    kinds = sorted(t.kind for t in parsed.tool_calls)
    assert kinds == ["command_execution", "file_change"]

    # Unknown event and item types are preserved, never dropped.
    assert "cosmic.ray" in parsed.unknown_event_types
    assert "quantum_teleport" in parsed.unknown_item_types
    # The trailing non-JSON line is kept verbatim.
    assert any("not-json" in u for u in parsed.unparsed_lines)
    # The explicit error envelope is captured.
    assert any("transient upstream hiccup" in e for e in parsed.errors)


def test_synthetic_full_subset_math_and_null_preservation():
    parsed = ev.parse_stream_text(_load("synthetic_full.jsonl"))
    turns = [u for u in parsed.turn_usage if u.reported]
    # Turn 1 reported cache_write=50; turn 2 reported cache_write=null (absent).
    assert turns[0].cache_write_input_tokens == 50
    assert turns[1].cache_write_input_tokens is None  # null preserved, not 0

    total = parsed.total_usage
    assert total.input_tokens == 1300          # 1000 + 300
    assert total.cached_input_tokens == 500    # 200 + 300
    assert total.cache_write_input_tokens == 50  # 50 + (None -> ignored)
    assert total.output_tokens == 130          # 120 + 10
    assert total.reasoning_output_tokens == 40  # 40 + 0
    # reasoning is a subset of output; non-reasoning = output - reasoning.
    assert total.non_reasoning_output_tokens == 130 - 40
    # uncached = input - cache_read - cache_write, clamped >= 0.
    assert total.uncached_input_tokens == 1300 - 500 - 50


def test_turn_failed_records_unknown_not_zero():
    parsed = ev.parse_stream_text(_load("exec_turn_failed_unsupported_model.jsonl"))
    assert parsed.turns_failed == 1
    # A failed turn carries no usage: unknown, never a confident zero.
    assert parsed.turn_usage[0].reported is False
    assert parsed.turn_usage[0].input_tokens is None
    assert parsed.usage_complete is False
    assert any("not supported" in e for e in parsed.errors)


def test_unparsed_line_not_dropped_and_scalar_is_not_an_event():
    # A bare JSON scalar is not an event object; it must be treated as unparsed.
    parsed = ev.parse_stream_text('5\n{"type":"turn.started"}\n"a string"\n')
    assert parsed.turns_started == 1
    assert set(parsed.unparsed_lines) == {"5", '"a string"'}


# --------------------------------------------------------------------------
# Typed records (frozen contract §3).
# --------------------------------------------------------------------------

def test_attempt_and_runresult_from_parsed_streams():
    p1 = ev.parse_stream_text(_load("real_tool_and_filechange.jsonl"))
    a1 = attempt_from_parsed(
        p1, index=0, cmd_argv=["codex", "exec", "--json"],
        exit_code=0, raw_event_path="/x/a0.jsonl",
    )
    assert isinstance(a1, CodexAttempt)
    assert a1.thread_id == p1.thread_id
    assert a1.final_output == "FINISHED"
    assert all(isinstance(t, CodexTurnUsage) for t in a1.turns)
    assert all(isinstance(t, CodexToolCall) for t in a1.tools)

    run = CodexRunResult(thread_id=p1.thread_id, attempts=[a1], status="completed")
    assert run.total_usage.input_tokens == 53494
    assert run.usage_complete is True
    assert run.final_output == "FINISHED"
    d = run.as_dict()
    assert d["total_usage"]["cached_input_tokens"] == 37120


def test_runresult_keeps_failed_attempt_usage_across_retry():
    # A failed attempt that still reported usage must survive into the total,
    # alongside the succeeding attempt (contract §5: charge every attempt).
    failed = CodexAttempt(
        index=0,
        turns=[CodexTurnUsage(input_tokens=100, output_tokens=10, source_event="turn.completed", reported=True)],
        failure_class="server_error",
    )
    ok = CodexAttempt(
        index=1,
        turns=[CodexTurnUsage(input_tokens=200, output_tokens=20, source_event="turn.completed", reported=True)],
    )
    run = CodexRunResult(attempts=[failed, ok], status="completed")
    assert run.total_usage.input_tokens == 300   # not reset to 200
    assert run.total_usage.output_tokens == 30
    assert len(run.attempts) == 2


def test_redact_argv_masks_secrets_without_dropping_tokens():
    argv = ["codex", "exec", "--api-key", "sk-super-secret", "-c", "authorization=Bearer xyz", "-"]
    out = redact_argv(argv)
    assert out[3] == "***REDACTED***"
    assert out[5] == "authorization=***REDACTED***"
    assert out[0] == "codex" and out[-1] == "-"
    assert len(out) == len(argv)  # shape preserved


def test_resume_fixture_shares_thread_id():
    first = ev.parse_stream_text(_load("synthetic_full.jsonl"))
    resumed = ev.parse_stream_text(_load("synthetic_resume.jsonl"))
    assert resumed.thread_id == first.thread_id  # same-thread continuation
