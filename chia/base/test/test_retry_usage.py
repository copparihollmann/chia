"""Tests for retry-attempt token accounting.

A retried call is billed for every attempt, but each backend clears its
per-attempt metadata at the top of the next attempt — so before the ledger, the
tokens burned by attempts that raised were gone by the time a result existed to
report them on, and a retry left nothing behind but a ``logger.warning``.

Two layers:

* **Ledger unit tests** — :meth:`LLMCallBase.note_retry` /
  :meth:`LLMCallBase.attach_usage` in isolation, including the zero-token case
  that must not degrade a billed total.
* **End-to-end through a real backend** — ``BedrockLLM.prompt`` driven by a fake
  boto3 that fails the first attempt after billing it, asserting the returned
  usage covers both attempts. This is the number that was wrong.
"""
import types

import pytest

from chia.base.llm_call import LLMCallBase, QueryResult
from chia.base.usage import RetryAttempt, TokenUsage


UNPRICED_MODEL = "chia-test-model-with-no-price-9e3f"


class _FakeLLM(LLMCallBase):
    """Minimal concrete backend, for driving the ledger without a provider."""

    def __init__(self, model=UNPRICED_MODEL, retries=3):
        super().__init__(system_message="")
        self.model = model
        self.retries = retries
        self._last_metadata = {}

    def prompt(self, user_message, tools=None):  # pragma: no cover - never called
        raise NotImplementedError


def _result():
    return QueryResult(result="", returncode=0, stderr="", stream_result="")


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_a_failed_attempts_tokens_reach_the_final_usage():
    """The defect: attempt 1 burned 100 input tokens and raised, so a total that
    reports only attempt 2 under-counts real spend."""
    llm = _FakeLLM()

    llm.begin_retry_ledger()
    llm._last_metadata = {"input_tokens": 100, "output_tokens": 10}
    llm.note_retry(0, RuntimeError("boom"))

    result = llm.attach_usage(_result(), {"input_tokens": 30, "output_tokens": 5})

    assert result.usage.input_tokens == 130
    assert result.usage.output_tokens == 15


def test_the_failed_attempts_are_itemised_on_the_result():
    llm = _FakeLLM()
    llm.begin_retry_ledger()
    llm._last_metadata = {"input_tokens": 100}
    llm.note_retry(0, ValueError("first"))
    llm._last_metadata = {"input_tokens": 200}
    llm.note_retry(1, ValueError("second"), backoff_s=5.0)

    result = llm.attach_usage(_result(), {"input_tokens": 1})

    assert [a.attempt for a in result.retry_attempts] == [1, 2]
    assert [a.error_type for a in result.retry_attempts] == ["ValueError", "ValueError"]
    assert [a.backoff_s for a in result.retry_attempts] == [0.0, 5.0]
    assert result.retry_attempts[1].usage.input_tokens == 200


def test_attempt_index_is_stored_one_based():
    """Backends count attempts from 0; a human-facing record counts from 1, and
    the existing warnings already say "attempt 1/3"."""
    llm = _FakeLLM()
    llm.begin_retry_ledger()

    assert llm.note_retry(0, RuntimeError("x")).attempt == 1


def test_a_zero_token_attempt_does_not_degrade_a_billed_total():
    """A call that died before the provider reported anything burned nothing chia
    can see. Folding its empty usage in would drop cost_source from billed to
    estimated for no reason."""
    llm = _FakeLLM()
    llm.begin_retry_ledger()
    llm._last_metadata = {}
    llm.note_retry(0, RuntimeError("died early"))

    result = llm.attach_usage(_result(), {"input_tokens": 10, "cost_usd": 0.5})

    assert result.usage.cost_usd == pytest.approx(0.5)
    assert result.usage.cost_source == "billed"
    # ...but the attempt is still recorded, so the retry is visible.
    assert len(result.retry_attempts) == 1


def test_an_unpriced_failed_attempt_poisons_the_cost_rather_than_being_ignored():
    """When a failed attempt genuinely burned tokens at an unknown price, the total
    is unknown — reporting the successful attempt's cost as the total would
    understate it silently."""
    llm = _FakeLLM(model=UNPRICED_MODEL)
    llm.begin_retry_ledger()
    llm._last_metadata = {"input_tokens": 100}
    llm.note_retry(0, RuntimeError("boom"))

    result = llm.attach_usage(_result(), {"input_tokens": 10, "cost_usd": 0.5})

    assert result.usage.input_tokens == 110
    assert result.usage.cost_usd is None
    assert result.usage.cost_source == "unavailable"


def test_note_retry_defaults_to_the_instances_last_metadata():
    llm = _FakeLLM()
    llm.begin_retry_ledger()
    llm._last_metadata = {"input_tokens": 42}

    assert llm.note_retry(0, RuntimeError("x")).usage.input_tokens == 42


def test_begin_retry_ledger_clears_a_previous_calls_attempts():
    """A call that propagated an exception must not leak its attempts into the
    next call on the same instance."""
    llm = _FakeLLM()
    llm.begin_retry_ledger()
    llm._last_metadata = {"input_tokens": 100}
    llm.note_retry(0, RuntimeError("stale"))

    llm.begin_retry_ledger()

    assert llm.retry_attempts == ()
    assert llm.attach_usage(_result(), {}).usage.input_tokens == 0


def test_note_retry_works_before_begin_retry_ledger():
    """Defensive: a backend that forgets the reset still accounts correctly rather
    than raising AttributeError inside an except branch."""
    llm = _FakeLLM()
    llm._last_metadata = {"input_tokens": 5}

    llm.note_retry(0, RuntimeError("x"))

    assert len(llm.retry_attempts) == 1


def test_retry_attempts_defaults_to_empty_on_a_result():
    assert _result().retry_attempts == ()


def test_note_retry_never_raises_when_the_profiler_misbehaves(monkeypatch):
    """It is called from inside an except branch; a telemetry failure there would
    replace the real error with a confusing one."""
    llm = _FakeLLM()
    llm.begin_retry_ledger()

    import chia.trace.profiler as profiler_mod
    monkeypatch.setattr(
        profiler_mod, "get_profiler",
        lambda: (_ for _ in ()).throw(RuntimeError("collector down")),
    )

    assert llm.note_retry(0, RuntimeError("x")).attempt == 1


def test_note_retry_emits_a_profiler_event(monkeypatch):
    """Retries used to be logger.warning only, so a retry rate could not be
    measured after the fact."""
    events = []

    class _FakeProfiler:
        def log_event(self, event_type, **kwargs):
            events.append((event_type, kwargs))

    import chia.trace.profiler as profiler_mod
    monkeypatch.setattr(profiler_mod, "get_profiler", lambda: _FakeProfiler())

    llm = _FakeLLM(retries=3)
    llm.begin_retry_ledger()
    llm._last_metadata = {"input_tokens": 100, "output_tokens": 10}
    llm.note_retry(1, TimeoutError("too slow"), backoff_s=10.0)

    assert len(events) == 1
    event_type, payload = events[0]
    assert event_type == "llm_retry"
    assert payload["attempt"] == 2
    assert payload["total_attempts"] == 3
    assert payload["error_type"] == "TimeoutError"
    assert payload["backoff_s"] == 10.0
    assert payload["input_tokens"] == 100
    assert payload["backend"] == "_FakeLLM"


# ---------------------------------------------------------------------------
# RetryAttempt
# ---------------------------------------------------------------------------


def test_retry_attempt_truncates_a_huge_error_body():
    """A provider error can be an entire HTML page; this is destined for a
    profiler event, not a debug log."""
    record = RetryAttempt(attempt=1, error_type="ServerError", error="x" * 5000)

    assert len(record.error) < 600
    assert record.error.endswith("[truncated]")


def test_retry_attempt_defaults_to_an_all_zero_usage():
    record = RetryAttempt(attempt=1, error_type="ServerError", error="boom")

    assert isinstance(record.usage, TokenUsage)
    assert record.usage.total_tokens == 0


def test_retry_attempt_as_event_is_flat():
    record = RetryAttempt(
        attempt=2, error_type="ServerError", error="503", backoff_s=5.0,
        usage=TokenUsage(input_tokens=7, cost_usd=0.1, cost_source="billed"),
    )

    event = record.as_event()

    assert event["attempt"] == 2
    assert event["input_tokens"] == 7
    assert event["cost_usd"] == 0.1
    assert event["cost_source"] == "billed"


# ---------------------------------------------------------------------------
# End to end through a real backend
# ---------------------------------------------------------------------------


def _bedrock_with_responses(monkeypatch, outcomes):
    """Install a fake boto3 whose ``converse`` walks *outcomes*: each entry is
    either a response dict or an exception instance to raise."""
    from chia.models import bedrock as bedrock_mod

    mod = types.ModuleType("boto3")
    state = {"i": 0}

    class _FakeClient:
        def converse(self, **kwargs):
            outcome = outcomes[state["i"]]
            state["i"] += 1
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    mod.client = lambda name, **kw: _FakeClient()
    monkeypatch.setitem(__import__("sys").modules, "boto3", mod)
    return bedrock_mod


def _converse_ok(text, in_tok, out_tok):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": in_tok, "outputTokens": out_tok,
                  "totalTokens": in_tok + out_tok},
    }


def test_bedrock_retry_usage_covers_every_attempt(monkeypatch):
    """The end-to-end regression: attempt 1 is billed 1000 input tokens and then
    the service 500s. Reporting only attempt 2's 20 tokens under-counts the call
    by 50x."""
    bedrock_mod = _bedrock_with_responses(monkeypatch, [
        _converse_ok("partial", 1000, 100),
        _converse_ok("PONG", 20, 5),
    ])

    llm = bedrock_mod.BedrockLLM(model=UNPRICED_MODEL, region="us-east-1", retries=3)

    # Fail the first attempt *after* its usage was recorded, exactly as a
    # truncated/failed response does: the provider billed the tokens either way.
    original_classify = llm._run_converse
    calls = {"n": 0}

    def _run_then_fail(user_message, tools):
        result = original_classify(user_message, tools)
        calls["n"] += 1
        if calls["n"] == 1:
            raise bedrock_mod.ServerError("node", "503", "service unavailable")
        return result

    monkeypatch.setattr(llm, "_run_converse", _run_then_fail)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    cli = llm.prompt("ping", tools=[])

    assert cli.success is True
    assert cli.result == "PONG"
    assert cli.usage.input_tokens == 1020
    assert cli.usage.output_tokens == 105
    assert [a.error_type for a in cli.retry_attempts] == ["ServerError"]
    assert cli.retry_attempts[0].usage.input_tokens == 1000


def test_bedrock_total_failure_still_reports_what_was_burned(monkeypatch):
    """A run that failed every attempt is not free. Before, the exhausted-loop
    result carried no accounting at all, so a grid's spend total silently omitted
    its failures."""
    bedrock_mod = _bedrock_with_responses(monkeypatch, [
        _converse_ok("x", 100, 10),
        _converse_ok("x", 100, 10),
        _converse_ok("x", 100, 10),
    ])

    llm = bedrock_mod.BedrockLLM(model=UNPRICED_MODEL, region="us-east-1", retries=3)
    original = llm._run_converse

    def _always_fail(user_message, tools):
        original(user_message, tools)
        raise bedrock_mod.ServerError("node", "503", "down")

    monkeypatch.setattr(llm, "_run_converse", _always_fail)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    cli = llm.prompt("ping", tools=[])

    assert cli.success is False
    assert cli.returncode == -1
    assert cli.usage.input_tokens == 300
    assert len(cli.retry_attempts) == 3
