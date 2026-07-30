"""Tests for the guarded chia -> aet usage sink (:mod:`chia.trace.aet_sink`)."""

from __future__ import annotations

import json

import pytest

from chia.trace import aet_sink

aet = pytest.importorskip("aet", reason="aet not installed; sink is a no-op")


def _metric_names(run_dir) -> list[str]:
    path = run_dir / "logs" / "metrics.jsonl"
    lines = path.read_text().strip().splitlines()
    return [json.loads(line)["name"] for line in lines]


USAGE = {
    "input_tokens": 120,
    "output_tokens": 60,
    "cache_read_input_tokens": 100,
    "cache_creation_input_tokens": 40,
    "cost_usd": 0.0123,
    "num_turns": 4,
    "model": "anthropic.claude-sonnet-4-6",
}


def test_sink_writes_expected_metrics(tmp_path):
    ok = aet_sink.record_run_usage(
        USAGE,
        model="anthropic.claude-sonnet-4-6",
        run_id="run1",
        run_dir=tmp_path,
        project="proj",
        suite="suite",
        target="chia",
        method="agent",
        seed=0,
        enabled=True,  # explicit opt-in, bypassing the env flag
    )
    assert ok is True

    names = _metric_names(tmp_path)
    assert "gen_ai.usage.input_tokens" in names
    assert "gen_ai.usage.output_tokens" in names
    assert "gen_ai.usage.cache_read.input_tokens" in names
    assert "gen_ai.usage.cache_creation.input_tokens" in names
    assert "aet.agent.cost_usd" in names
    assert "aet.agent.num_turns" in names
    # Per-model breakdown recorded too.
    assert any(n.startswith("per_model.") and n.endswith(".cost_usd") for n in names)


def test_sink_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("CHIA_AET_SINK", raising=False)
    ok = aet_sink.record_run_usage(USAGE, run_dir=tmp_path, enabled=None)
    assert ok is False
    assert not (tmp_path / "logs").exists()


def test_sink_enabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIA_AET_SINK", "1")
    ok = aet_sink.record_run_usage(
        USAGE, run_id="r", run_dir=tmp_path, seed=0,
    )
    assert ok is True
    assert (tmp_path / "logs" / "metrics.jsonl").exists()


def test_sink_noop_without_token_counts(tmp_path):
    ok = aet_sink.record_run_usage(
        {"model": "x", "num_turns": 1}, run_dir=tmp_path, enabled=True,
    )
    assert ok is False


def test_sink_noop_without_run_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("CHIA_AET_RUN_DIR", raising=False)
    ok = aet_sink.record_run_usage(USAGE, enabled=True)
    assert ok is False


def test_sink_omits_cost_when_absent(tmp_path):
    usage = dict(USAGE)
    usage.pop("cost_usd")
    ok = aet_sink.record_run_usage(usage, run_dir=tmp_path, enabled=True, seed=0)
    assert ok is True
    names = _metric_names(tmp_path)
    # No cost metric fabricated when cost is unknown.
    assert "aet.agent.cost_usd" not in names
