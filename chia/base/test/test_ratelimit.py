"""Tests for waiting out a usage limit.

Clock and sleep are injected, so nothing here actually waits — every duration below is
an assertion about the number that *would* have been slept.

Two layers:

* **The primitive** — every bound, and every refusal message, since a run that stopped
  after three waits is a different situation from one that stopped after five hours and
  the operator has to be able to tell which.
* **Through ClaudeCodeLLM** — that a limit is waited out and the call then succeeds, that
  a waited limit does not consume a retry attempt, and that with no policy the historical
  behaviour (propagate immediately) is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chia.base.ratelimit import (
    CLAUDE_SESSION_RESOURCE,
    RateLimitPolicy,
    RateLimitWaiter,
    RateLimitWaitExhausted,
)

NOW = datetime(2026, 8, 5, 12, 0, 0, tzinfo=timezone.utc)


def _policy(**kwargs):
    slept = []
    policy = RateLimitPolicy(
        sleep=slept.append,
        now=lambda: NOW,
        **kwargs,
    )
    return policy, slept


def _limit():
    return RuntimeError("You've hit your limit")


# ---------------------------------------------------------------------------
# seconds_until
# ---------------------------------------------------------------------------


def test_the_wait_is_the_distance_to_the_reset_plus_padding():
    """Waking at the exact reset second reliably hits the limit once more, because the
    provider's clock and this one are not the same clock."""
    policy, _ = _policy(max_waits=1, pad_s=30.0)
    waiter = RateLimitWaiter(policy)

    seconds = waiter.seconds_until(NOW + timedelta(minutes=45))

    assert seconds == pytest.approx(45 * 60 + 30)


def test_a_reset_already_in_the_past_means_no_wait():
    """Retrying immediately is correct; sleeping would idle a worker for nothing."""
    policy, _ = _policy(max_waits=1)
    waiter = RateLimitWaiter(policy)

    assert waiter.seconds_until(NOW - timedelta(minutes=5)) == 0.0


def test_a_naive_timestamp_is_read_as_utc():
    """chia's own parsers produce aware UTC, but a caller's may not, and subtracting a
    naive from an aware datetime raises rather than returning a wrong number."""
    policy, _ = _policy(max_waits=1, pad_s=0.0)
    waiter = RateLimitWaiter(policy)

    naive = (NOW + timedelta(minutes=10)).replace(tzinfo=None)

    assert waiter.seconds_until(naive) == pytest.approx(600)


def test_no_reset_time_means_no_wait():
    policy, _ = _policy(max_waits=1)

    assert RateLimitWaiter(policy).seconds_until(None) == 0.0


# ---------------------------------------------------------------------------
# The bounds
# ---------------------------------------------------------------------------


def test_waiting_is_off_by_default():
    """The historical behaviour: a limit propagates. Right for an interactive call,
    wrong for a grid — so it is opt-in rather than a silent change."""
    policy = RateLimitPolicy()
    assert policy.enabled is False

    with pytest.raises(RateLimitWaitExhausted) as exc:
        RateLimitWaiter(policy).wait(_limit(), NOW + timedelta(minutes=5))

    assert "disabled" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_wait_sleeps_the_computed_duration():
    policy, slept = _policy(max_waits=2, pad_s=10.0)
    waiter = RateLimitWaiter(policy)

    waited = waiter.wait(_limit(), NOW + timedelta(minutes=30))

    assert waited == pytest.approx(30 * 60 + 10)
    assert slept == [pytest.approx(30 * 60 + 10)]
    assert waiter.waits == 1


def test_an_already_reset_window_is_retried_without_sleeping():
    policy, slept = _policy(max_waits=1)
    waiter = RateLimitWaiter(policy)

    assert waiter.wait(_limit(), NOW - timedelta(minutes=1)) == 0.0
    assert slept == []
    # Still counted, so a provider that keeps reporting a stale reset cannot spin.
    assert waiter.waits == 1


def test_the_attempt_budget_is_enforced():
    policy, _ = _policy(max_waits=2)
    waiter = RateLimitWaiter(policy)
    waiter.wait(_limit(), NOW + timedelta(minutes=1))
    waiter.wait(_limit(), NOW + timedelta(minutes=1))

    with pytest.raises(RateLimitWaitExhausted) as exc:
        waiter.wait(_limit(), NOW + timedelta(minutes=1))

    assert "waited out 2 limit(s)" in str(exc.value)


def test_the_total_time_budget_is_enforced():
    """An account out of quota for the month must not park a worker indefinitely — this
    is what separates a normal five-hour window from that."""
    policy, _ = _policy(max_waits=10, max_total_wait_s=3600, pad_s=0.0)
    waiter = RateLimitWaiter(policy)
    waiter.wait(_limit(), NOW + timedelta(minutes=50))

    with pytest.raises(RateLimitWaitExhausted) as exc:
        waiter.wait(_limit(), NOW + timedelta(minutes=50))

    assert "over the" in str(exc.value) and "ceiling" in str(exc.value)


def test_an_implausibly_distant_reset_is_refused():
    """A reset a week out is more likely a clock skew or a parse error than a real
    window, and parking a worker on it would be worse than failing."""
    policy, _ = _policy(max_waits=5, max_single_wait_s=6 * 3600)
    waiter = RateLimitWaiter(policy)

    with pytest.raises(RateLimitWaitExhausted) as exc:
        waiter.wait(_limit(), NOW + timedelta(days=7))

    assert "clock skew" in str(exc.value)


def test_a_missing_reset_time_is_refused_rather_than_guessed():
    """Guessing a duration would invent the one fact this primitive depends on."""
    policy, _ = _policy(max_waits=3)

    with pytest.raises(RateLimitWaitExhausted) as exc:
        RateLimitWaiter(policy).wait(_limit(), None)

    assert "nothing to wait for" in str(exc.value)


def test_the_original_error_is_always_chained():
    """So the operator still sees the provider's own message alongside chia's reason."""
    policy, _ = _policy(max_waits=0)
    original = _limit()

    with pytest.raises(RateLimitWaitExhausted) as exc:
        RateLimitWaiter(policy).wait(original, NOW + timedelta(minutes=1))

    assert exc.value.__cause__ is original


def test_the_session_resource_is_a_whole_unit_gate():
    """Distinct from claude_creds at 0.01: that says "this worker has credentials", this
    says "this worker may run one more session right now", so N units admits exactly N.
    A subscription seat is one concurrency slot; fanning ten calls at it produces ten
    rate limits rather than one."""
    from chia.models.claude import ClaudeCodeLLM

    assert CLAUDE_SESSION_RESOURCE == "claude_session"
    creds_gate = ClaudeCodeLLM.prompt._chia_options["resources"]
    assert creds_gate == {"claude_creds": 0.01}
    assert CLAUDE_SESSION_RESOURCE not in creds_gate

    gated = ClaudeCodeLLM.prompt.options(
        resources={CLAUDE_SESSION_RESOURCE: 1, "claude_creds": 0.01})
    assert gated is not None


# ---------------------------------------------------------------------------
# Through ClaudeCodeLLM
# ---------------------------------------------------------------------------


def _llm(policy=None, retries=3):
    from chia.models.claude import ClaudeCodeLLM

    return ClaudeCodeLLM(backend="cli", log_stream=False, retries=retries,
                         rate_limit_policy=policy)


def _rate_limit_error(llm, reset_in_minutes=30):
    from chia.models.claude import RateLimitError

    return RateLimitError(node_id="n1", reset_time=NOW + timedelta(minutes=reset_in_minutes),
                          raw_message="You've hit your limit")


def test_without_a_policy_a_limit_still_propagates(monkeypatch):
    """Unchanged default behaviour: this feature must not silently start waiting."""
    from chia.models.claude import RateLimitError

    llm = _llm(policy=None)

    def _boom(user, tools):
        raise _rate_limit_error(llm)

    monkeypatch.setattr(llm, "_run_claude", _boom)

    with pytest.raises(RateLimitError):
        llm.prompt("hi", tools=[])


def test_with_a_policy_the_limit_is_waited_out_and_the_call_succeeds(monkeypatch):
    from chia.models.claude import ClaudeCodeQueryResult

    policy, slept = _policy(max_waits=2, pad_s=5.0)
    llm = _llm(policy=policy)
    calls = {"n": 0}
    ok = ClaudeCodeQueryResult(result="PONG", returncode=0, stderr="",
                              stream_result="")

    def _limited_then_ok(user, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limit_error(llm)
        return ok

    monkeypatch.setattr(llm, "_run_claude", _limited_then_ok)

    result = llm.prompt("hi", tools=[])

    assert result.result == "PONG"
    assert result.success is True
    assert slept == [pytest.approx(30 * 60 + 5)]


def test_a_waited_limit_does_not_consume_a_retry_attempt(monkeypatch):
    """Otherwise three windows would exhaust a three-attempt budget without the model
    ever being reached — which is precisely the failure this exists to remove."""
    from chia.models.claude import ClaudeCodeQueryResult

    policy, slept = _policy(max_waits=5, pad_s=0.0)
    llm = _llm(policy=policy, retries=2)
    calls = {"n": 0}
    ok = ClaudeCodeQueryResult(result="PONG", returncode=0, stderr="",
                              stream_result="")

    def _limited_thrice(user, tools):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise _rate_limit_error(llm, reset_in_minutes=1)
        return ok

    monkeypatch.setattr(llm, "_run_claude", _limited_thrice)

    result = llm.prompt("hi", tools=[])

    assert result.result == "PONG"
    assert len(slept) == 3          # three windows waited...
    assert calls["n"] == 4          # ...on a retries=2 budget


def test_an_exhausted_wait_budget_surfaces_with_the_provider_error_attached(monkeypatch):
    from chia.models.claude import RateLimitError

    policy, _ = _policy(max_waits=1, pad_s=0.0)
    llm = _llm(policy=policy, retries=5)

    def _always_limited(user, tools):
        raise _rate_limit_error(llm, reset_in_minutes=1)

    monkeypatch.setattr(llm, "_run_claude", _always_limited)

    with pytest.raises(RateLimitWaitExhausted) as exc:
        llm.prompt("hi", tools=[])

    assert isinstance(exc.value.__cause__, RateLimitError)


def test_other_retryable_errors_are_unaffected_by_the_policy(monkeypatch):
    """The policy must only change the rate-limit branch; a server error still uses
    exponential backoff and still consumes attempts."""
    from chia.models.claude import ServerError

    policy, _ = _policy(max_waits=3)
    llm = _llm(policy=policy, retries=2)
    calls = {"n": 0}

    def _always_500(user, tools):
        calls["n"] += 1
        raise ServerError(node_id="n1", raw_message="503")

    monkeypatch.setattr(llm, "_run_claude", _always_500)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    result = llm.prompt("hi", tools=[])

    assert result.success is False
    assert calls["n"] == 2
