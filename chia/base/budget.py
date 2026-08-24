"""A spend ceiling that refuses a fan-out *before* it is dispatched.

chia will happily fan five hundred agent calls onto a Ray cluster with no notion of what
they cost. On a shared grant that is a footgun, and both downstream consumers of chia
built their own ceiling rather than do without one.

**Why pre-flight rather than mid-run.** A per-token backend gives no natural checkpoint:
by the time a mid-grid check fired, the calls that crossed the line would already be
billed. A check before dispatch can only overshoot by at most one grid — and a grid's
size is knowable in advance, because it is a list of rows.

**Why "already spent" is not enough.** A floor-only check passes whenever the ledger is
below the cap and then permits a grid of any size. Downstream that failed in exactly the
way it sounds: a cap of $11 on an empty ledger, an estimate of $7, and $15.55 spent. One
grid *is* the whole spend, so the projection has to include the grid about to run.

**Why the estimate comes from history rather than a price table.** Both are available and
the recorded per-call cost is the honest one: it already includes the retries, the cache
misses and the tool loops this workload actually produces. A price table costs the tokens
you predict, not the tokens you spend.

Two accounting rules carry over from :mod:`chia.base.usage`, because a budget that broke
either would be worse than none:

* **Subscription quota does not count against a dollar cap.** It is reported beside the
  metered figure and never added to it — a seat's quota is not a card charge.
* **An unpriced call makes the total a floor, not a fact.** Those calls are counted and
  surfaced; a cap check that treated them as $0 would silently authorise a grid whose
  cost it could not see.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from chia.base.usage import TokenUsage, sum_usages

#: Calls below this are almost certainly replays, refusals or errors rather than real
#: work. Letting them drag the median down is how an estimate built from history repeats
#: the mistake it exists to prevent.
MIN_REAL_CALL_USD = 0.01


class BudgetExceeded(RuntimeError):
    """A dispatch was refused because it would cross the spend ceiling.

    Never raised silently and never downgraded to a warning: the point of a ceiling is
    that something stops. The message always states what is already spent, what the
    grid projects to, and where the per-call estimate came from, so raising the cap is a
    deliberate act rather than a guess.
    """


@dataclass(frozen=True)
class BudgetStatus:
    """What has been spent, and what it is safe to dispatch.

    :param cap_usd: The ceiling being enforced.
    :param metered_usd: Real per-token spend recorded so far.
    :param subscription_usd: Dollar-equivalent of subscription quota consumed. Reported
        for visibility and **not** counted against ``cap_usd``.
    :param priced_calls: Calls whose cost is known.
    :param unpriced_calls: Calls with token counts but no known price, which make
        ``metered_usd`` a floor rather than a total.
    :type cap_usd: float
    :type metered_usd: float
    :type subscription_usd: float
    :type priced_calls: int
    :type unpriced_calls: int
    """

    cap_usd: float
    metered_usd: float = 0.0
    subscription_usd: float = 0.0
    priced_calls: int = 0
    unpriced_calls: int = 0
    per_call_usd: Optional[float] = None

    @property
    def over(self) -> bool:
        """Whether metered spend has already reached the cap."""
        return self.metered_usd >= self.cap_usd

    @property
    def headroom_usd(self) -> float:
        """Cap minus metered spend, floored at zero."""
        return max(0.0, self.cap_usd - self.metered_usd)

    @property
    def is_complete(self) -> bool:
        """Whether every recorded call had a known price.

        ``False`` means ``metered_usd`` is a lower bound. A caller reporting a total
        should say so rather than presenting a floor as a fact.
        """
        return self.unpriced_calls == 0

    def affordable_calls(self, per_call_usd: Optional[float] = None) -> Optional[int]:
        """How many more calls fit under the cap, or ``None`` with no estimate.

        :param per_call_usd: Override for the historical estimate.
        :type per_call_usd: Optional[float]
        :rtype: Optional[int]
        """
        rate = per_call_usd if per_call_usd is not None else self.per_call_usd
        if not rate:
            return None
        return int(self.headroom_usd // rate)


def _usages(usages: Iterable[TokenUsage]) -> List[TokenUsage]:
    return [u for u in usages if isinstance(u, TokenUsage)]


def summarise(
    usages: Iterable[TokenUsage],
    cap_usd: float,
) -> BudgetStatus:
    """Fold recorded *usages* into a :class:`BudgetStatus`.

    :param usages: One :class:`~chia.base.usage.TokenUsage` per completed call — from
        ``QueryResult.usage``, or from :func:`chia.trace.aet_sink.collect_run_usage`.
    :param cap_usd: The ceiling to report against.
    :type usages: Iterable[TokenUsage]
    :type cap_usd: float
    :rtype: BudgetStatus

    Pure: no Ray, no filesystem, no clock, so the arithmetic can be tested against a
    literal list.
    """
    items = _usages(usages)
    metered = [u for u in items if u.billing_mode == "per_token"]
    quota = [u for u in items if u.billing_mode == "subscription"]

    metered_total = sum(u.cost_usd for u in metered if u.cost_usd is not None)
    quota_total = sum(u.cost_usd for u in quota if u.cost_usd is not None)
    priced = sum(1 for u in metered if u.cost_usd is not None)
    # A call with no tokens at all is not an unpriced call — nothing happened.
    unpriced = sum(1 for u in metered if u.cost_usd is None and u.total_tokens)

    return BudgetStatus(
        cap_usd=cap_usd,
        metered_usd=float(metered_total),
        subscription_usd=float(quota_total),
        priced_calls=priced,
        unpriced_calls=unpriced,
        per_call_usd=estimate_per_call(metered),
    )


def estimate_per_call(usages: Iterable[TokenUsage]) -> Optional[float]:
    """Median metered cost of a *real* call, or ``None`` when there is no history.

    :param usages: Recorded usages; non-metered ones are ignored.
    :type usages: Iterable[TokenUsage]
    :rtype: Optional[float]

    The median rather than the mean, because one runaway call should not set the budget
    for the rest. Calls under :data:`MIN_REAL_CALL_USD` are excluded: a replay or a
    refusal costs almost nothing and including them would make the estimate optimistic
    in exactly the situation where optimism is expensive.
    """
    costs = [u.cost_usd for u in _usages(usages)
             if u.billing_mode == "per_token" and u.cost_usd is not None
             and u.cost_usd >= MIN_REAL_CALL_USD]
    return statistics.median(costs) if costs else None


def check_budget(
    cap_usd: float,
    usages: Iterable[TokenUsage] = (),
    *,
    projected_calls: int = 0,
    per_call_usd: Optional[float] = None,
) -> BudgetStatus:
    """Refuse a dispatch that would cross *cap_usd*; otherwise return the status.

    :param cap_usd: The ceiling, in USD of metered spend.
    :param usages: Recorded usages so far.
    :param projected_calls: How many calls are about to be dispatched. Pass this — a
        check without it is the floor-only check that failed downstream.
    :param per_call_usd: Override the historical per-call estimate. Use when the grid
        about to run is unlike anything in the history.
    :type cap_usd: float
    :type usages: Iterable[TokenUsage]
    :type projected_calls: int
    :type per_call_usd: Optional[float]
    :rtype: BudgetStatus
    :raises BudgetExceeded: Spend has already reached the cap, or the projection
        crosses it.
    :raises ValueError: *cap_usd* is not positive — a cap of zero or less would refuse
        everything, which is almost always a mis-passed argument rather than an intent.

    With no history and no ``per_call_usd`` the projection cannot be computed, and the
    call is **allowed** with ``per_call_usd=None`` on the returned status. That is
    deliberate: the first grid has nothing to estimate from, and refusing it would make
    the feature impossible to start using. The returned status says the estimate is
    missing so a caller can report it.
    """
    if cap_usd <= 0:
        raise ValueError(
            f"cap_usd must be positive, got {cap_usd!r}. A non-positive cap refuses "
            f"every dispatch, which is almost never the intent."
        )

    status = summarise(usages, cap_usd)

    if status.over:
        raise BudgetExceeded(
            f"metered spend is ${status.metered_usd:.2f}, at or over the "
            f"${cap_usd:.2f} cap. Raise the cap deliberately if that is intended; "
            f"nothing is refused silently."
            + (f" (${status.subscription_usd:.2f} of subscription-equivalent is "
               f"reported beside it and deliberately does not count against the cap — "
               f"it is not a card charge.)" if status.subscription_usd else "")
            + (f" {status.unpriced_calls} call(s) had no known price, so the figure is "
               f"a floor." if status.unpriced_calls else "")
        )

    if projected_calls:
        rate = per_call_usd if per_call_usd is not None else status.per_call_usd
        if rate is not None:
            projected = status.metered_usd + rate * projected_calls
            if projected > cap_usd:
                affordable = status.affordable_calls(rate)
                raise BudgetExceeded(
                    f"{projected_calls} call(s) at ~${rate:.4f} each projects to "
                    f"${projected:.2f}, over the ${cap_usd:.2f} cap "
                    f"(${status.metered_usd:.2f} already metered). "
                    + (f"The estimate is the median of {status.priced_calls} recorded "
                       f"call(s) from this run, not a price-table guess. "
                       if per_call_usd is None else "")
                    + f"About {affordable} call(s) would fit. Raise the cap "
                    f"deliberately, or dispatch fewer."
                )
    return status
