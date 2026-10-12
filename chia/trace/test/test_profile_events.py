import json

import pytest

from chia.base.llm_call import QueryResult
from chia.base.usage import TokenUsage
from chia.models.claude import ClaudeCodeLLM
from chia.trace.aet_sink import export_profile_jsonl
from chia.trace.profile_events import (
    PROFILE_SCHEMA_VERSION,
    ProfileContext,
    agent_event,
    llm_request_event,
    tool_activity_event,
)


class _ProfiledFakeLLM(ClaudeCodeLLM):
    """Use the shared accounting implementation without invoking a CLI."""


def test_llm_event_is_subset_safe_and_privacy_safe():
    event = llm_request_event(
        ProfileContext(run_id="r", call_id="c", agent_id="a"),
        provider="openai", model="gpt-test", backend="CodexLLM", status="completed",
        input_tokens=100, cache_read_tokens=80, cache_write_tokens=10,
        output_tokens=20, reasoning_tokens=5,
    )

    assert event["schema_version"] == PROFILE_SCHEMA_VERSION
    assert event["input_tokens"] == 100
    assert event["cache_read_tokens"] == 80
    assert event["reasoning_tokens"] == 5
    assert not ({"prompt", "arguments", "result", "environment", "credentials"} & event.keys())
    json.dumps(event)


def test_tool_event_contains_only_identity_timing_and_status():
    event = tool_activity_event(
        ProfileContext(call_id="c"), tool_name="Bash", category="tool",
        duration_s=1.25, status="failed",
    )
    assert event["tool_name"] == "Bash"
    assert event["duration_s"] == 1.25
    assert "command" not in event
    assert "output" not in event


def test_agent_event_rejects_non_lifecycle_type():
    with pytest.raises(ValueError):
        agent_event("agent_pause", ProfileContext(), name="worker")


def test_external_export_includes_only_closed_schema_events(tmp_path):
    safe = llm_request_event(
        ProfileContext(call_id="c"), provider="openai", model="gpt",
        backend="CodexLLM", status="completed",
    )
    out = tmp_path / "profile.jsonl"
    assert export_profile_jsonl([
        {"type": "complete", "extra": {"prompt": "secret"}}, safe,
    ], out) == 1
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records == [safe]
    assert "secret" not in out.read_text()


def test_claude_harness_prefers_explicit_non_anthropic_provider(monkeypatch):
    captured = []

    class FakeProfiler:
        enabled = True

        @staticmethod
        def profile_context():
            return ProfileContext(call_id="c")

        @staticmethod
        def log_profile_event(event):
            captured.append(event)

    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: FakeProfiler())
    llm = ClaudeCodeLLM(model="openai/gpt-test", retries=1)
    llm.begin_retry_ledger()
    llm._emit_usage_event(
        TokenUsage(input_tokens=1, model="openai/gpt-test"),
        meta={"provider": "openai"}, status="completed", attempt=1, retry=False,
    )
    assert captured[0]["provider"] == "openai"


def test_retry_trace_events_sum_to_the_cumulative_result_once(monkeypatch):
    captured = []

    class FakeProfiler:
        enabled = True

        @staticmethod
        def profile_context():
            return ProfileContext(call_id="c")

        @staticmethod
        def log_profile_event(event):
            captured.append(event)

        @staticmethod
        def log_event(*_args, **_kwargs):
            pass

    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: FakeProfiler())
    llm = _ProfiledFakeLLM(model="openai/gpt-test", retries=2)
    llm.begin_retry_ledger()
    llm.note_retry(0, RuntimeError("retry"), meta={"input_tokens": 100})
    result = llm.attach_usage(
        result=QueryResult(result="", returncode=0, stderr="", stream_result=""),
        meta={"input_tokens": 30, "output_tokens": 5, "provider": "openai"},
    )

    requests = [event for event in captured if event["type"] == "llm_request"]
    assert len(requests) == 2
    assert [event["input_tokens"] for event in requests] == [100, 30]
    assert sum(event["input_tokens"] for event in requests) == result.usage.input_tokens == 130


def test_all_failed_accounting_does_not_emit_a_synthetic_request(monkeypatch):
    captured = []

    class FakeProfiler:
        enabled = True

        @staticmethod
        def profile_context():
            return ProfileContext(call_id="c")

        @staticmethod
        def log_profile_event(event):
            captured.append(event)

        @staticmethod
        def log_event(*_args, **_kwargs):
            pass

    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: FakeProfiler())
    llm = _ProfiledFakeLLM(model="openai/gpt-test", retries=1)
    llm.begin_retry_ledger()
    llm.note_retry(0, RuntimeError("terminal"), meta={"input_tokens": 12})
    result = llm.attach_usage(
        result=QueryResult(result="", returncode=1, stderr="failed", stream_result=""),
        meta={},
    )

    requests = [event for event in captured if event["type"] == "llm_request"]
    assert len(requests) == 1
    assert requests[0]["attempt"] == 1
    assert result.usage.input_tokens == 12
