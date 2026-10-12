"""Streaming-capture tests for :mod:`chia.models.codex_capture`.

These exercise the *real* :class:`subprocess.Popen` path against tiny Python
child scripts, because the whole value of the module — that a killed run still
leaves a parseable raw JSONL file — cannot be proven by a mocked subprocess.
Offline: no ``codex`` binary is involved, only ``sys.executable``.
"""

from __future__ import annotations

import os
import sys

from chia.models.codex_capture import stream_codex
from chia.models.codex_events import parse_stream_text


def _child(body: str) -> list[str]:
    return [sys.executable, "-u", "-c", body]


def test_normal_run_tees_events_and_feeds_stdin(tmp_path):
    raw = tmp_path / "events.raw.jsonl"
    err = tmp_path / "stderr.log"
    # Child echoes stdin back inside an agent_message so we can prove stdin flow.
    body = (
        "import sys, json\n"
        "prompt = sys.stdin.read().strip()\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'T1'}))\n"
        "print(json.dumps({'type': 'turn.started'}))\n"
        "print(json.dumps({'type': 'item.completed', 'item': {'id': 'i0', 'type': 'agent_message', 'text': prompt}}))\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 5, 'cached_input_tokens': 1, 'cache_write_input_tokens': 0, 'output_tokens': 2, 'reasoning_output_tokens': 0}}))\n"
    )
    res = stream_codex(_child(body), input_text="PING", raw_path=str(raw), stderr_path=str(err), timeout=30)
    assert res.returncode == 0
    assert res.timed_out is False
    assert res.lines_written == 4
    parsed = parse_stream_text(raw.read_text())
    assert parsed.thread_id == "T1"
    assert parsed.final_text == "PING"  # stdin was delivered to the child
    assert parsed.total_usage.input_tokens == 5


def test_timeout_kills_but_raw_jsonl_is_salvageable(tmp_path):
    raw = tmp_path / "events.raw.jsonl"
    err = tmp_path / "stderr.log"
    # Emit two complete lines, flush, then hang forever mid-"turn".
    body = (
        "import sys, json, time\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'T2'}))\n"
        "print(json.dumps({'type': 'turn.started'}))\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    res = stream_codex(_child(body), input_text="", raw_path=str(raw), stderr_path=str(err), timeout=1.0)
    assert res.timed_out is True
    assert res.returncode is not None and res.returncode != 0
    # The salvage guarantee: the file on disk parses cleanly despite the kill.
    text = raw.read_text()
    assert text.count("\n") == 2
    parsed = parse_stream_text(text)
    assert parsed.thread_id == "T2"
    assert parsed.turns_started == 1
    assert parsed.turns_completed == 0  # died before turn.completed


def test_partial_line_no_newline_survives_a_mid_write_kill(tmp_path):
    raw = tmp_path / "events.raw.jsonl"
    err = tmp_path / "stderr.log"
    # One complete line, then a PARTIAL line with no trailing newline, then hang.
    # readline() would block forever on the newline-less fragment; the fd reader
    # must instead salvage it verbatim when the process is killed for timeout.
    body = (
        "import sys, json, time\n"
        "sys.stdout.write(json.dumps({'type': 'thread.started', 'thread_id': 'T5'}) + '\\n')\n"
        "sys.stdout.write('{\"type\":\"turn.started\"')\n"  # deliberately no newline
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    res = stream_codex(_child(body), input_text="", raw_path=str(raw), stderr_path=str(err), timeout=1.0)
    assert res.timed_out is True
    text = raw.read_text()
    # The partial fragment is on disk byte-for-byte (no newline was invented).
    assert text.endswith('{"type":"turn.started"')
    # It is kept as an unparsed record, never dropped.
    assert '{"type":"turn.started"' in res.unparsed_lines
    # And the complete line before it still parses.
    parsed = parse_stream_text(text)
    assert parsed.thread_id == "T5"


def test_corrupt_line_is_kept_not_dropped(tmp_path):
    raw = tmp_path / "events.raw.jsonl"
    err = tmp_path / "stderr.log"
    body = (
        "import sys, json\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'T3'}))\n"
        "print('this is not json')\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 3, 'output_tokens': 1}}))\n"
    )
    res = stream_codex(_child(body), input_text="", raw_path=str(raw), stderr_path=str(err), timeout=30)
    assert res.returncode == 0
    assert res.unparsed_lines == ["this is not json"]
    assert len(res.events) == 2
    parsed = parse_stream_text(raw.read_text())
    assert parsed.unparsed_lines == ["this is not json"]
    assert parsed.total_usage.input_tokens == 3


def test_per_line_callback_and_timestamped_file(tmp_path):
    raw = tmp_path / "events.raw.jsonl"
    err = tmp_path / "stderr.log"
    ts = tmp_path / "events.timestamped.jsonl"
    seen: list[tuple[str, bool]] = []
    body = (
        "import json\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'T4'}))\n"
        "print('nope')\n"
    )
    stream_codex(
        _child(body), input_text="", raw_path=str(raw), stderr_path=str(err),
        timeout=30, timestamped_path=str(ts),
        on_event=lambda line, ev: seen.append((line, ev is not None)),
    )
    assert seen[0][1] is True and "thread.started" in seen[0][0]
    assert seen[1] == ("nope", False)
    assert os.path.exists(ts)
    assert ts.read_text().count("\n") == 2


def test_stderr_is_captured_to_file(tmp_path):
    raw = tmp_path / "events.raw.jsonl"
    err = tmp_path / "stderr.log"
    body = (
        "import sys, json\n"
        "sys.stderr.write('boom on stderr\\n')\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 1}}))\n"
    )
    res = stream_codex(_child(body), input_text="", raw_path=str(raw), stderr_path=str(err), timeout=30)
    assert "boom on stderr" in res.stderr_text
    assert "boom on stderr" in err.read_text()
