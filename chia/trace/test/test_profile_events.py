import json
import time
from io import StringIO

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


def test_multi_model_cli_totals_are_not_attributed_to_the_primary_model(monkeypatch):
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
    llm = ClaudeCodeLLM(model="sonnet", retries=1)
    llm.begin_retry_ledger()
    llm.attach_usage(
        result=QueryResult(result="ok", returncode=0, stderr="", stream_result=""),
        meta={"input_tokens": 16, "output_tokens": 9,
              "model_usage": {"sonnet": {"input_tokens": 13},
                              "fixture/weak": {"input_tokens": 3}}},
    )
    assert captured[0]["model"] == "mixed"
    assert captured[0]["provider"] == "multiple"
    assert captured[0]["input_tokens"] == 16
    assert captured[0]["output_tokens"] == 9


def test_opencode_child_sessions_become_disjoint_per_agent_trace_records(monkeypatch):
    captured = []

    class FakeProfiler:
        enabled = True

        @staticmethod
        def profile_context():
            return ProfileContext(call_id="c", agent_id="lead-scope")

        @staticmethod
        def log_profile_event(event):
            captured.append(event)

    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: FakeProfiler())
    llm = _ProfiledFakeLLM(model="fixture/strong", retries=1)
    llm.begin_retry_ledger()
    llm.attach_usage(
        result=QueryResult(result="ok", returncode=0, stderr="", stream_result=""),
        meta={"input_tokens": 30, "output_tokens": 6,
              "subagent_usage_complete": True,
              "agent_usage": [
                  {"session_id": "root", "agent": "lead", "model": "fixture/strong",
                   "input_tokens": 20, "output_tokens": 4},
                  {"session_id": "child", "parent_session_id": "root",
                   "agent": "explorer", "model": "fixture/weak",
                   "input_tokens": 10, "output_tokens": 2},
              ]},
    )
    assert len(captured) == 2
    assert [event["agent_id"] for event in captured] == ["opencode:root", "opencode:child"]
    assert captured[1]["parent_agent_id"] == "opencode:root"
    assert [event["model"] for event in captured] == ["fixture/strong", "fixture/weak"]
    assert sum(event["input_tokens"] for event in captured) == 30
    assert sum(event["output_tokens"] for event in captured) == 6


def test_opencode_timed_messages_and_tool_span_keep_request_totals(monkeypatch):
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
    llm = _ProfiledFakeLLM(model="fixture/strong", retries=1)
    llm.begin_retry_ledger()
    llm.attach_usage(
        result=QueryResult(result="ok", returncode=0, stderr="", stream_result=""),
        meta={"input_tokens": 20, "output_tokens": 4,
              "subagent_usage_complete": True,
              "agent_usage": [{
                  "session_id": "root", "model": "fixture/strong",
                  "input_tokens": 20, "output_tokens": 4,
                  "num_turns": 2,
                  "requests": [
                      {"request_id": "one", "end_ts": 2.0, "duration_s": 1.0,
                       "input_tokens": 10, "output_tokens": 2},
                      {"request_id": "two", "end_ts": 4.0, "duration_s": 0.5,
                       "input_tokens": 10, "output_tokens": 2},
                  ],
                  "tools": [{"name": "task", "status": "completed",
                             "end_ts": 3.0, "duration_s": 1.0}],
              }]},
    )
    requests = [event for event in captured if event["type"] == "llm_request"]
    tools = [event for event in captured if event["type"] == "tool_activity"]
    assert [(event["request_id"], event["ts"], event["duration_s"])
            for event in requests] == [("one", 2.0, 1.0), ("two", 4.0, 0.5)]
    assert sum(event["input_tokens"] for event in requests) == 20
    assert len(tools) == 1 and tools[0]["ts"] == 3.0


def test_cli_result_folds_all_models_and_cache_classes_without_aet():
    llm = ClaudeCodeLLM(model="sonnet")
    result = {
        "type": "result", "subtype": "success", "total_cost_usd": 0.05,
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "modelUsage": {
            "sonnet": {"inputTokens": 10, "outputTokens": 2,
                       "cacheReadInputTokens": 70, "cacheCreationInputTokens": 20},
            "fixture/weak": {"inputTokens": 7, "outputTokens": 3,
                             "cacheReadInputTokens": 30, "cacheCreationInputTokens": 0},
        },
    }
    llm._process_event_line(json.dumps(result), StringIO(), [])
    assert llm._last_metadata["input_tokens"] == 17
    assert llm._last_metadata["output_tokens"] == 5
    assert llm._last_metadata["cache_read_input_tokens"] == 100
    assert llm._last_metadata["cache_creation_input_tokens"] == 20
    assert llm._last_metadata["model_usage"]["fixture/weak"]["input_tokens"] == 7


def test_claude_stream_child_task_id_produces_balanced_lifecycle(monkeypatch):
    captured = []

    class FakeProfiler:
        enabled = True

        @staticmethod
        def profile_context():
            return ProfileContext(call_id="call", agent_id="lead")

        @staticmethod
        def log_profile_event(event):
            captured.append(event)

    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: FakeProfiler())
    llm = ClaudeCodeLLM(model="sonnet")
    now = time.monotonic()
    llm._record_child_profiles([
        (now, json.dumps({"type": "system", "subtype": "task_started",
                          "task_id": "child-1", "subagent_type": "explorer",
                          "spawn_depth": 1, "session_id": "session"})),
        (now + 0.1, json.dumps({"type": "system", "subtype": "task_notification",
                                "task_id": "child-1", "status": "completed",
                                "usage": {"total_tokens": 100}})),
    ])
    assert [event["type"] for event in captured] == ["agent_start", "agent_end"]
    assert [event["agent_id"] for event in captured] == ["claude:child-1"] * 2
    assert all(event["parent_agent_id"] == "lead" for event in captured)
    assert captured[0]["name"] == captured[1]["name"] == "explorer"
    assert captured[1]["status"] == "completed"
    assert captured[1]["ts"] - captured[0]["ts"] == pytest.approx(0.1)
    assert "usage" not in captured[1]


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
