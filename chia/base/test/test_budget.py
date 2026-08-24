"""Tests for the pre-flight spend ceiling.

Pure arithmetic over literal usage lists: no Ray, no provider, no clock.

The two cases that matter most are the ones a naive implementation gets wrong:

* ``test_a_grid_that_would_cross_the_cap_is_refused`` — the floor-only check passes
  whenever the ledger is under the cap and then permits a grid of any size. That is how
  a $11 cap on an empty ledger became $15.55 of spend downstream.
* ``test_subscription_quota_does_not_count_against_a_dollar_cap`` — a seat's quota is
  not a card charge, and a budget that summed the two would refuse a grid that costs
  nothing.
"""

import pytest

from chia.base.budget import (
    MIN_REAL_CALL_USD,
    BudgetExceeded,
    BudgetStatus,
    check_budget,
    estimate_per_call,
    summarise,
)
from chia.base.usage import TokenUsage


def _metered(cost, tokens=1000):
    return TokenUsage(input_tokens=tokens, cost_usd=cost, cost_source="billed")


def _quota(cost, tokens=1000):
    return TokenUsage(input_tokens=tokens, cost_usd=cost, cost_source="billed",
                      billing_mode="subscription")


def _unpriced(tokens=1000):
    return TokenUsage(input_tokens=tokens)


# ---------------------------------------------------------------------------
# Summarising what has been spent
# ---------------------------------------------------------------------------


def test_metered_spend_is_summed():
    status = summarise([_metered(0.10), _metered(0.20), _metered(0.05)], cap_usd=10.0)

    assert status.metered_usd == pytest.approx(0.35)
    assert status.priced_calls == 3
    assert status.headroom_usd == pytest.approx(9.65)
    assert status.over is False


def test_subscription_quota_is_reported_beside_metered_spend_not_inside_it():
    status = summarise([_metered(0.10), _quota(9.99)], cap_usd=1.0)

    assert status.metered_usd == pytest.approx(0.10)
    assert status.subscription_usd == pytest.approx(9.99)
    assert status.over is False


def test_an_unpriced_call_makes_the_total_a_floor():
    status = summarise([_metered(0.10), _unpriced()], cap_usd=10.0)

    assert status.unpriced_calls == 1
    assert status.is_complete is False


def test_a_zero_token_call_is_not_an_unpriced_call():
    """Nothing happened, so there is nothing to be unsure about."""
    status = summarise([_metered(0.10), TokenUsage()], cap_usd=10.0)

    assert status.unpriced_calls == 0
    assert status.is_complete is True


def test_an_empty_history_is_a_clean_slate():
    status = summarise([], cap_usd=10.0)

    assert status.metered_usd == 0.0
    assert status.per_call_usd is None
    assert status.over is False


# ---------------------------------------------------------------------------
# The per-call estimate
# ---------------------------------------------------------------------------


def test_the_estimate_is_the_median_not_the_mean():
    """One runaway call must not set the budget for the rest."""
    estimate = estimate_per_call([_metered(0.10), _metered(0.10), _metered(100.0)])

    assert estimate == pytest.approx(0.10)


def test_trivially_cheap_calls_are_excluded_from_the_estimate():
    """A replay or a refusal costs almost nothing; including them would make the
    estimate optimistic in exactly the situation where optimism is expensive."""
    estimate = estimate_per_call([
        _metered(MIN_REAL_CALL_USD / 100), _metered(MIN_REAL_CALL_USD / 100),
        _metered(0.50), _metered(0.50), _metered(0.50),
    ])

    assert estimate == pytest.approx(0.50)


def test_subscription_calls_do_not_set_the_metered_estimate():
    assert estimate_per_call([_quota(5.0), _metered(0.20)]) == pytest.approx(0.20)


def test_no_history_means_no_estimate():
    assert estimate_per_call([]) is None
    assert estimate_per_call([_unpriced()]) is None


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_spend_already_at_the_cap_is_refused():
    with pytest.raises(BudgetExceeded) as exc:
        check_budget(1.0, [_metered(1.0)])

    assert "at or over" in str(exc.value)
    assert "$1.00" in str(exc.value)


def test_a_grid_that_would_cross_the_cap_is_refused():
    """THE case. A floor-only check passes here — $2 spent against a $10 cap — and then
    permits 100 more calls at $0.50 each. One grid is the whole spend."""
    history = [_metered(0.50), _metered(0.50), _metered(0.50), _metered(0.50)]

    with pytest.raises(BudgetExceeded) as exc:
        check_budget(10.0, history, projected_calls=100)

    message = str(exc.value)
    assert "100 call(s)" in message
    assert "projects to $52.00" in message
    assert "already metered" in message
    # The estimate's provenance is stated, so raising the cap is a deliberate act.
    assert "median of 4 recorded call(s)" in message
    assert "16 call(s) would fit" in message


def test_a_grid_that_fits_is_allowed():
    status = check_budget(10.0, [_metered(0.50)] * 4, projected_calls=10)

    assert status.metered_usd == pytest.approx(2.0)
    assert status.per_call_usd == pytest.approx(0.50)


def test_an_explicit_per_call_estimate_overrides_history():
    """For a grid unlike anything in the history — a different model, a longer task."""
    with pytest.raises(BudgetExceeded):
        check_budget(10.0, [_metered(0.01)] * 4, projected_calls=10,
                     per_call_usd=5.0)


def test_an_overridden_estimate_does_not_claim_to_come_from_history():
    with pytest.raises(BudgetExceeded) as exc:
        check_budget(1.0, [], projected_calls=10, per_call_usd=5.0)

    assert "recorded call" not in str(exc.value)


def test_the_first_grid_is_allowed_despite_having_nothing_to_estimate_from():
    """Refusing it would make the feature impossible to start using. The returned status
    says the estimate is missing, so a caller can report that rather than imply a
    projection was checked."""
    status = check_budget(10.0, [], projected_calls=500)

    assert status.per_call_usd is None
    assert status.affordable_calls() is None


def test_subscription_quota_does_not_count_against_a_dollar_cap():
    """A budget that summed the two would refuse a grid that costs no money at all."""
    status = check_budget(1.0, [_quota(500.0)], projected_calls=100)

    assert status.metered_usd == 0.0
    assert status.subscription_usd == pytest.approx(500.0)


def test_the_over_cap_message_mentions_quota_and_unpriced_calls_when_present():
    """A refusal a caller cannot act on is a refusal they will work around."""
    with pytest.raises(BudgetExceeded) as exc:
        check_budget(1.0, [_metered(2.0), _quota(3.0), _unpriced()])

    message = str(exc.value)
    assert "subscription-equivalent" in message
    assert "no known price" in message
    assert "floor" in message


def test_a_non_positive_cap_is_a_programming_error_not_a_refusal():
    """A cap of zero refuses every dispatch; that is almost always a mis-passed
    argument, and reporting it as BudgetExceeded would send debugging the wrong way."""
    for cap in (0.0, -1.0):
        with pytest.raises(ValueError) as exc:
            check_budget(cap, [])
        assert "positive" in str(exc.value)


def test_projection_is_skipped_when_no_calls_are_projected():
    """A status query is not a dispatch, so it must not refuse."""
    status = check_budget(10.0, [_metered(0.50)] * 4)

    assert status.metered_usd == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# affordable_calls
# ---------------------------------------------------------------------------


def test_affordable_calls_uses_the_headroom():
    status = BudgetStatus(cap_usd=10.0, metered_usd=4.0, per_call_usd=0.5)

    assert status.affordable_calls() == 12


def test_affordable_calls_is_zero_at_the_cap_not_negative():
    status = BudgetStatus(cap_usd=10.0, metered_usd=12.0, per_call_usd=0.5)

    assert status.headroom_usd == 0.0
    assert status.affordable_calls() == 0
