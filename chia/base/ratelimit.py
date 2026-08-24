"""Waiting out a usage limit, as a loop primitive rather than as each caller's problem.

chia detects a rate limit and raises. That is the right default for an interactive call
and the wrong one for a grid: a subscription's window resets on a clock, so the work is
not lost, it is *deferred* — and a framework that raises leaves every consumer to
rediscover the same wait-and-resume loop. Both downstream consumers did, and one of them
lost 18 rows across three grids to an exhausted five-hour window before it did.

Two halves, and the second is what makes the first cheap:

**Wait until the window resets.** The provider tells you when: the Claude CLI's
``rate_limit_event`` carries a reset timestamp, and :class:`~chia.models.claude.RateLimitError`
carries it as ``reset_time``. A fixed backoff would either wake too early (and burn a
retry confirming the limit is still there) or too late (and idle a cluster).

**Resume, do not restart.** chia already has the better half of this:
``resume_session=True`` gives the CLI ``--session-id`` on the first call and ``--resume``
after, so a continuation sends a few kilobytes of new turns instead of re-sending the
whole prompt. Without it, waiting out a limit and then re-sending a 185 KB corpus pays
for the same input twice — the second time at the fresh-input rate, because the cache
window has also expired.

**Bounded, always.** :attr:`RateLimitPolicy.max_waits` and
:attr:`RateLimitPolicy.max_total_wait_s` exist because "wait for the window" is
unbounded in principle: an account that is out of quota for the month would otherwise
park a Ray worker forever. A policy that cannot be exhausted is not a policy.

Concurrency
-----------

Waiting is not enough on its own. A subscription seat is a *single* concurrency slot:
fanning ten calls at it produces ten rate limits rather than one. Gate them with
:data:`CLAUDE_SESSION_RESOURCE` — a whole-unit custom Ray resource, so N units on the
cluster admits exactly N concurrent sessions::

    llm.prompt.options(resources={CLAUDE_SESSION_RESOURCE: 1}).chia_remote(llm, msg)

This is deliberately distinct from ``claude_creds``, which chia already uses at 0.01
per call — a fractional gate that says "this worker has credentials", not "this worker
may run one more session right now".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger("chia.ratelimit")

#: Whole-unit Ray resource that bounds concurrent agent sessions. Distinct from
#: ``claude_creds`` (0.01 per call, "this worker has credentials"): this one means
#: "this worker may run one more session right now", so N units admits exactly N.
CLAUDE_SESSION_RESOURCE = "claude_session"


class RateLimitWaitExhausted(RuntimeError):
    """The wait budget ran out before the usage window reset.

    Carries the original error as ``__cause__`` so a caller still sees the provider's
    own message, and states which bound was hit — a run that stopped because it had
    waited three times is a different situation from one that stopped because it had
    waited five hours, and the operator needs to know which.
    """


@dataclass
class RateLimitPolicy:
    """How long a call may wait for a usage window to reset, and how often.

    :param max_waits: How many separate limits may be waited out in one call. ``0``
        disables waiting entirely, which is the historical behaviour: propagate.
    :param max_total_wait_s: Ceiling on the *summed* wait across those attempts. A
        five-hour subscription window is normal; a month-long quota exhaustion is not,
        and this is what tells them apart.
    :param max_single_wait_s: Ceiling on any one wait. A reset timestamp far in the
        future is more likely a clock skew or a parse error than a real window, and
        parking a worker on it would be worse than failing.
    :param pad_s: Slack added to each computed wait. Waking at the exact reset
        second reliably hits the limit once more, because the provider's clock and
        this one are not the same clock.
    :param sleep: Injected for tests, so the suite does not actually wait.
    :param now: Injected for tests; must return an aware UTC datetime.
    :type max_waits: int
    :type max_total_wait_s: float
    :type max_single_wait_s: float
    :type pad_s: float
    :type sleep: Callable[[float], None]
    :type now: Callable[[], datetime]
    """

    max_waits: int = 0
    max_total_wait_s: float = 6 * 60 * 60
    max_single_wait_s: float = 5.5 * 60 * 60
    pad_s: float = 30.0
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = field(
        default_factory=lambda: lambda: datetime.now(timezone.utc)
    )

    @property
    def enabled(self) -> bool:
        """Whether this policy permits waiting at all."""
        return self.max_waits > 0


@dataclass
class RateLimitWaiter:
    """Per-call bookkeeping for :class:`RateLimitPolicy`.

    :param policy: The bounds to enforce.
    :type policy: RateLimitPolicy

    One instance per ``prompt`` call: the budget is per call, not per process, so a
    long-running driver does not accumulate a debt that starves a later call.
    """

    policy: RateLimitPolicy
    waits: int = 0
    total_waited_s: float = 0.0

    def seconds_until(self, reset_time: Optional[datetime]) -> float:
        """Padded seconds to wait for *reset_time*, or ``0.0`` when it has passed.

        :param reset_time: The provider's reset timestamp. A naive datetime is read as
            UTC, since that is what chia's parsers produce.
        :type reset_time: Optional[datetime]
        :rtype: float

        ``None`` yields ``0.0``: with no timestamp there is nothing to wait *for*, and
        guessing a duration would be inventing the one fact this primitive depends on.
        """
        if reset_time is None:
            return 0.0
        if reset_time.tzinfo is None:
            reset_time = reset_time.replace(tzinfo=timezone.utc)
        delta = (reset_time - self.policy.now()).total_seconds()
        if delta <= 0:
            # Already reset — retrying immediately is correct, and sleeping would idle
            # a worker for no reason.
            return 0.0
        return delta + self.policy.pad_s

    def wait(self, exc: BaseException, reset_time: Optional[datetime]) -> float:
        """Wait out one limit, or raise when the budget is spent.

        :param exc: The provider error, re-raised as ``__cause__`` on failure.
        :param reset_time: When the window resets.
        :type exc: BaseException
        :type reset_time: Optional[datetime]
        :returns: Seconds actually slept.
        :rtype: float
        :raises RateLimitWaitExhausted: Waiting is disabled, the attempt or time budget
            is spent, the single wait is too long, or there is no reset timestamp to
            wait for.
        """
        policy = self.policy
        if not policy.enabled:
            raise RateLimitWaitExhausted(
                "rate-limit waiting is disabled (max_waits=0); the limit propagates. "
                "Pass a RateLimitPolicy(max_waits=N) to wait for the window instead."
            ) from exc

        if reset_time is None:
            raise RateLimitWaitExhausted(
                "the provider gave no reset time, so there is nothing to wait for. "
                "Guessing a duration would invent the one fact this depends on."
            ) from exc

        if self.waits >= policy.max_waits:
            raise RateLimitWaitExhausted(
                f"already waited out {self.waits} limit(s), the configured maximum. "
                f"Raise max_waits deliberately if a run is expected to span more than "
                f"that many windows."
            ) from exc

        seconds = self.seconds_until(reset_time)
        if seconds > policy.max_single_wait_s:
            raise RateLimitWaitExhausted(
                f"the reset is {seconds / 3600:.1f}h away, over the "
                f"{policy.max_single_wait_s / 3600:.1f}h single-wait ceiling. A reset "
                f"that far out is more likely a clock skew or a parse error than a real "
                f"window, and parking a worker on it would be worse than failing."
            ) from exc
        if self.total_waited_s + seconds > policy.max_total_wait_s:
            raise RateLimitWaitExhausted(
                f"waiting {seconds / 3600:.1f}h would bring this call's total wait to "
                f"{(self.total_waited_s + seconds) / 3600:.1f}h, over the "
                f"{policy.max_total_wait_s / 3600:.1f}h ceiling. An account out of "
                f"quota for the month must not park a worker indefinitely."
            ) from exc

        self.waits += 1
        self.total_waited_s += seconds
        if seconds:
            logger.warning(
                "usage limit hit; waiting %.1f min for the window to reset "
                "(wait %d/%d, %.1f min total)",
                seconds / 60, self.waits, policy.max_waits,
                self.total_waited_s / 60,
            )
            policy.sleep(seconds)
        else:
            logger.warning("usage limit hit but the window has already reset; "
                           "retrying immediately (wait %d/%d)",
                           self.waits, policy.max_waits)
        return seconds
