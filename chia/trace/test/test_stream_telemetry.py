"""Tests for :mod:`chia.trace.stream_telemetry` and the activity path into the run dir.

The claim under test is narrow and checkable: chia already receives every tool call, its
result and its wall-clock duration, and until now dropped all of it at the point of
ingestion. These tests pin the three things that makes true, and the two that must not
change as a result:

* a tool call becomes an activity band with a category and a duration;
* the summary that crosses the Ray boundary carries no prompt text, tool input or tool
  output — only the derived shape;
* per-call streams are placed on a run-wide clock, not stacked at t=0;
* a run without aet, or without the CLI backend, produces exactly what it produced before;
* ``metrics/trajectory.json`` is written, because it is the only run-dir artifact that can
  carry bands at all.
"""

from __future__ import annotations

import json

import pytest

from chia.base.usage import TokenUsage
from chia.trace.aet_sink import (
    CallUsage,
    RunUsage,
    _activity_bands,
    _write_trajectory_json,
)
from chia.trace.stream_telemetry import (
    METADATA_KEY,
    merge_into_metadata,
    summarize_stream,
)

pytest.importorskip("aet.tracking.claude_stream")

SECRET = "sk-ant-supersecret-0123456789"


def _stream(*, tool="Read", tool_input=None, secret_in_output=False):
    """A minimal Claude Code stream: one assistant tool_use, its result, one result event."""
    tool_input = tool_input or {"file_path": "/repo/main.py"}
    out = "the file contents" + (f" {SECRET}" if secret_in_output else "")
    lines = [
        {"type": "system", "subtype": "init", "session_id": "s1"},
        {"type": "assistant", "message": {
            "model": "claude-haiku-4-5", "usage": {"input_tokens": 10, "output_tokens": 4},
            "content": [{"type": "tool_use", "id": "t1", "name": tool, "input": tool_input}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": out}]}},
        {"type": "assistant", "message": {
            "model": "claude-haiku-4-5", "usage": {"input_tokens": 2, "output_tokens": 7},
            "content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "subtype": "success", "total_cost_usd": 0.01,
         "num_turns": 2, "duration_ms": 3000, "session_id": "s1",
         "usage": {"input_tokens": 12, "output_tokens": 11}},
    ]
    # Timestamps: the tool_use at t=1.0, its result at t=3.5 — a 2.5 s call.
    stamps = [0.0, 1.0, 3.5, 4.0, 4.2]
    return [(t, json.dumps(ev)) for t, ev in zip(stamps, lines)]


# --------------------------------------------------------------- the summary
def test_a_tool_call_becomes_a_band_with_a_category_and_a_duration():
    """The headline. This is the data that used to be rendered to prose and dropped."""
    summary = summarize_stream(_stream(tool="Read"))
    assert summary["tool_calls"] == 1
    bands = summary["bands"]
    assert len(bands) == 1
    assert bands[0]["category"] == "read"
    assert bands[0]["t1_s"] - bands[0]["t0_s"] == pytest.approx(2.5)


def test_the_classifier_separates_the_activities():
    """A band is only useful if different tools land in different lanes."""
    cats = {tool: summarize_stream(_stream(tool=tool))["bands"][0]["category"]
            for tool in ("Read", "Edit", "Bash")}
    assert cats["Read"] == "read"
    assert cats["Edit"] == "write"
    assert cats["Bash"] == "bash"
    assert len(set(cats.values())) == 3


def test_an_instantaneous_call_produces_no_band():
    """A zero-length interval drawn as one puts area under the activity curve that no time
    was spent in."""
    events = _stream()
    same_time = [(1.0, line) for _, line in events]
    assert summarize_stream(same_time)["bands"] == []


def test_the_summary_carries_no_prompt_tool_input_or_tool_output():
    """THE constraint on this design.

    The summary crosses a Ray boundary and lands in a run directory people commit. Tool
    inputs carry file contents and prompts; tool results carry command output, which is
    where a leaked credential would be. Only the derived shape may travel — and the tool
    *name* is a fixed vocabulary, so it is allowed.
    """
    summary = summarize_stream(_stream(tool_input={"file_path": f"/keys/{SECRET}.pem"},
                                       secret_in_output=True))
    blob = json.dumps(summary)
    assert SECRET not in blob
    assert "the file contents" not in blob
    assert "file_path" not in blob
    assert summary["tools_used"] == ["Read"]        # the name survives; nothing else does


def test_tool_errors_are_counted_separately_from_tool_calls():
    events = _stream()
    events[2] = (3.5, json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "boom", "is_error": True}]}}))
    summary = summarize_stream(events)
    assert summary["tool_calls"] == 1 and summary["tool_errors"] == 1
    assert summary["bands"][0]["is_error"] is True


def test_an_empty_stream_summarises_to_nothing():
    assert summarize_stream([]) == {}


def test_an_unparseable_stream_is_empty_not_an_exception():
    """Telemetry is a figure. A figure must never be able to fail a run."""
    assert summarize_stream([(0.0, "not json"), (1.0, "{oops")]).get("bands", []) == []


def test_an_empty_summary_leaves_the_metadata_untouched():
    """Without the aet extra the metadata must be byte-identical to what it was before."""
    meta = {"model": "m", "cost_usd": 0.1}
    assert merge_into_metadata(dict(meta), {}) == meta
    assert METADATA_KEY not in merge_into_metadata(dict(meta), {})


# ------------------------------------------------------- placing on the clock
def _call(ts, *, duration, bands, cid):
    return CallUsage(call_id=cid, func="f", ts=ts,
                     usage=TokenUsage(input_tokens=1, output_tokens=1, cost_usd=0.01,
                                      cost_source="billed"),
                     duration_s=duration, telemetry={"bands": bands})


def test_each_calls_bands_are_offset_onto_the_run_clock():
    """Every stream numbers its tools from its own t=0.

    Without the offset a run of N prompts stacks N sets of bands on top of each other at
    the start of the figure, and the activity view describes a run that never happened.

    The origin is the one the trajectory points use (``RunUsage.clock_origin``) — the first
    call's *start* — so bands and the spend curve share an x-axis and nothing lands at
    negative time. Call a ran 98→100 and call b ran 108→110, so on a clock zeroed at 98 the
    two 0.5 s-in bands land at 0.5 and 10.5.
    """
    band = [{"t0_s": 0.5, "t1_s": 1.5, "category": "read"}]
    run = RunUsage(calls=(
        _call(100.0, duration=2.0, bands=band, cid="a"),     # ran 98.0 → 100.0
        _call(110.0, duration=2.0, bands=band, cid="b"),     # ran 108.0 → 110.0
    ))
    got = _activity_bands(run)
    assert [round(b["t0_s"], 3) for b in got] == [0.5, 10.5]
    assert [round(b["t1_s"], 3) for b in got] == [1.5, 11.5]


def test_the_first_calls_own_activity_is_not_lost():
    """The regression this clock change exists for.

    Zeroing the clock at the first *completion* put every tool the first call ran at
    negative time — a single-call run lost its entire activity view, and an N-call run
    lost 1/N of it. Nothing may fall off the front of the run.
    """
    run = RunUsage(calls=(
        _call(100.0, duration=10.0, bands=[{"t0_s": 0.5, "t1_s": 1.0, "category": "read"},
                                           {"t0_s": 9.5, "t1_s": 9.9, "category": "bash"}],
              cid="a"),
    ))
    got = _activity_bands(run)
    assert [b["category"] for b in got] == ["read", "bash"]
    assert got[0]["t0_s"] == pytest.approx(0.5)
    assert got[1]["t1_s"] == pytest.approx(9.9)


def test_a_run_recorded_without_durations_keeps_its_old_clock():
    """Runs collected before per-call durations existed must reconstruct unchanged."""
    run = RunUsage(calls=(
        CallUsage(call_id="a", func="f", ts=100.0,
                  usage=TokenUsage(input_tokens=1, cost_usd=0.0, cost_source="billed")),
        CallUsage(call_id="b", func="f", ts=110.0,
                  usage=TokenUsage(input_tokens=1, cost_usd=0.0, cost_source="billed")),
    ))
    assert run.clock_origin() == 100.0
    assert [p["t_s"] for p in run.trajectory()] == [0.0, 10.0]


# ------------------------------------------------------------ the run dir
def test_the_fast_path_artifact_is_written_and_carries_the_bands(tmp_path):
    """``logs/`` cannot hold bands — aet's own reconstruction documents them as out of
    reach there — so ``metrics/trajectory.json`` is the only way an activity view can
    reach a reader."""
    from aet.trajectory.model import RunTrajectory

    points = [{"index": 0, "t_s": 0.0, "cum_input": 10, "cum_output": 4, "cum_cache": 100,
               "cum_cache_read": 80, "cum_cache_creation": 20, "cum_cost": 0.01,
               "provisional_cost": False},
              {"index": 1, "t_s": 10.0, "cum_input": 20, "cum_output": 8, "cum_cache": 200,
               "cum_cache_read": 160, "cum_cache_creation": 40, "cum_cost": 0.02,
               "provisional_cost": False}]
    summary = {"run_id": "r1", "duration_s": 10.0, "num_rounds": 1, "final_cost_usd": 0.02,
               "final_input_tokens": 20, "final_output_tokens": 8, "final_cache_tokens": 200}
    bands = [{"t0_s": 0.5, "t1_s": 1.5, "category": "read", "tool_name": "Read",
              "weight": 1.0, "is_error": False}]

    assert _write_trajectory_json(tmp_path, RunUsage(), summary, points, bands) is True

    back = RunTrajectory.from_run_dir(tmp_path)      # the reader aet plot uses
    assert len(back.points) == 2
    assert len(back.bands) == 1
    assert back.bands[0].category == "read"
    assert back.points[-1].cum_cache_read_tokens == 160
    assert back.tests_total() is None                # chia runs no oracle; claims no score


def test_no_partial_artifact_is_left_behind(tmp_path):
    """A half-written trajectory read as a figure would be a fabricated observation."""
    _write_trajectory_json(tmp_path, RunUsage(), {"run_id": "r", "duration_s": 1.0},
                           [{"index": 0, "t_s": 0.0, "cum_input": 1, "cum_output": 1,
                             "cum_cache": 0, "cum_cache_read": 0, "cum_cache_creation": 0,
                             "cum_cost": 0.0}], [])
    assert list(tmp_path.rglob("*.partial")) == []
    assert (tmp_path / "metrics" / "trajectory.json").is_file()


def test_no_points_writes_nothing(tmp_path):
    assert _write_trajectory_json(tmp_path, RunUsage(), {}, [], []) is False
    assert not (tmp_path / "metrics").exists()
