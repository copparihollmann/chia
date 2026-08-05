"""Canonical token-usage accounting shared by every LLM backend.

:class:`~chia.base.llm_call.QueryResult` carries a :class:`TokenUsage` so callers
read a run's token and cost accounting off the public result instead of reaching
into a backend's private ``_last_metadata`` dict. This module is the single home
for that vocabulary because the type sits on ``QueryResult`` (in ``chia.base``)
while the backends that populate it live in ``chia.models`` — putting it any
lower would make ``chia.base`` depend on ``chia.models``.

Three accounting rules are enforced here rather than left to each caller, because
each one was a source of wrong numbers in practice:

**Input tokens come in three separately-priced classes.** ``input_tokens``,
``cache_read_input_tokens`` and ``cache_creation_input_tokens`` are billed at
different rates (cache reads at roughly a tenth of fresh input, cache writes at a
premium). Collapsing them into one "input" figure and pricing it at the fresh-input
rate misreports cost by whatever fraction of the prompt was cached — which for a
long-context agent loop is most of it.

**An unknown price is ``None``, never ``0.0``.** A fabricated zero is
indistinguishable from a genuinely free call once it has been summed, so
:func:`estimate_cost_usd` returns ``None`` and :attr:`TokenUsage.cost_source`
records ``"unavailable"``. Downstream tooling can then report the run as unpriced
instead of silently deflating a total.

**Subscription quota is not money.** A seat-authenticated CLI reports what the
call *would* have cost on the metered API. Adding that to real per-token charges
produces a number that is neither spend nor quota, so :attr:`TokenUsage.billing_mode`
records which one it is and :meth:`TokenUsage.__add__` refuses to combine the two.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal, Optional

CostSource = Literal["billed", "estimated", "unavailable"]
BillingMode = Literal["per_token", "subscription"]


#: Where :attr:`TokenUsage.cost_usd` came from.
#:
#: ``billed`` — the provider reported the figure itself (the Claude CLI's
#: ``total_cost_usd``, opencode's per-message ``cost``). ``estimated`` — chia
#: derived it from token counts and a price table. ``unavailable`` — the price is
#: not known, and ``cost_usd`` is ``None``.
COST_SOURCES = ("billed", "estimated", "unavailable")

#: How the call is paid for. ``per_token`` is metered spend that may be summed
#: into a bill; ``subscription`` is quota consumption reported in dollar-equivalent
#: terms, which may not.
BILLING_MODES = ("per_token", "subscription")

#: The canonical usage vocabulary. Kept as a tuple so callers can iterate it.
CANONICAL_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
    "cost_usd",
    "num_turns",
)

#: The subset of :data:`CANONICAL_USAGE_KEYS` that are token counts, i.e. the
#: keys that are summed when two usages are combined.
TOKEN_COUNT_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
)

# Non-canonical alias -> canonical key. Every provider spells the same four
# counters differently; each alias here is one a chia backend has actually been
# observed to emit (opencode's short cache keys, the OpenAI completions naming,
# codex's ``cached_input_tokens``).
_USAGE_KEY_ALIASES = {
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_input_tokens",
    "cached_input_tokens": "cache_read_input_tokens",
    "prompt_tokens": "input_tokens",
    "completion_tokens": "output_tokens",
}


def normalize_usage_keys(meta: dict) -> dict:
    """Return *meta* with any aliased usage keys renamed to the canonical set.

    :param meta: A backend's raw usage metadata. Non-usage keys (``model``,
        ``tools``, ``duration_s``, ...) pass through untouched.
    :type meta: dict
    :returns: A new dict using only canonical key names.
    :rtype: dict

    When both an alias and its canonical target are present their numeric values
    are summed, so no count is dropped by the rename.
    """
    out: dict = {}
    for key, value in meta.items():
        dest = _USAGE_KEY_ALIASES.get(key, key)
        if dest in out and isinstance(out[dest], (int, float)) and isinstance(value, (int, float)):
            out[dest] = out[dest] + value
        else:
            out[dest] = value
    return out


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    model: str = "",
) -> Optional[float]:
    """Best-effort USD cost estimate honoring the four input classes.

    :param input_tokens: Fresh (uncached) input tokens.
    :param output_tokens: Generated tokens.
    :param cache_read_input_tokens: Prompt tokens served from the prompt cache.
    :param cache_creation_input_tokens: Prompt tokens written into the cache.
    :param model: Provider model id, used to look the rates up.
    :type input_tokens: int
    :type output_tokens: int
    :type cache_read_input_tokens: int
    :type cache_creation_input_tokens: int
    :type model: str
    :returns: The estimate, or ``None`` when the price is unknown.
    :rtype: Optional[float]

    Pricing is delegated to aet's ``PriceTable`` when aet is importable. chia
    stays standalone, so when it is not installed the estimate is simply
    unavailable — the caller must omit the cost, never substitute ``0``.
    """
    try:
        from aet.trajectory.pricing import PriceTable
    except Exception:
        return None
    try:
        return PriceTable.from_env().estimate_usd(
            input_tokens,
            output_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
            model=model,
        )
    except Exception:
        return None


@dataclass(frozen=True)
class TokenUsage:
    """One call's (or one run's) token and cost accounting.

    :param input_tokens: Fresh input tokens, excluding anything served from or
        written to the prompt cache.
    :param output_tokens: Generated tokens.
    :param cache_read_input_tokens: Prompt tokens served from the cache.
    :param cache_creation_input_tokens: Prompt tokens written into the cache.
    :param reasoning_tokens: Thinking/reasoning tokens, where the provider
        reports them separately from ``output_tokens``.
    :param num_turns: Assistant turns in the agent loop.
    :param cost_usd: Cost in USD, or ``None`` when unknown.
    :param cost_source: One of :data:`COST_SOURCES`.
    :param billing_mode: One of :data:`BILLING_MODES`.
    :param model: The provider model id the counts belong to.
    :type input_tokens: int
    :type output_tokens: int
    :type cache_read_input_tokens: int
    :type cache_creation_input_tokens: int
    :type reasoning_tokens: int
    :type num_turns: int
    :type cost_usd: Optional[float]
    :type cost_source: str
    :type billing_mode: str
    :type model: str

    Frozen, so a usage handed to a caller cannot be mutated behind its back;
    combine two with ``+`` (see :meth:`__add__`) to accumulate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    num_turns: int = 0
    cost_usd: Optional[float] = None
    cost_source: CostSource = "unavailable"
    billing_mode: BillingMode = "per_token"
    model: str = ""

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def billed_input_tokens(self) -> int:
        """Every input token the provider charged for, across all three classes.

        Use this for "how big was the prompt", never for costing — the three
        classes carry different rates, which is the whole reason they are kept
        apart.
        """
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        """All billed input plus all output. ``reasoning_tokens`` is excluded
        because providers that report it count it inside ``output_tokens``."""
        return self.billed_input_tokens + self.output_tokens

    @property
    def cache_hit_ratio(self) -> Optional[float]:
        """Share of billed input served from the prompt cache, or ``None`` when
        there was no input to divide by."""
        billed = self.billed_input_tokens
        if not billed:
            return None
        return self.cache_read_input_tokens / billed

    @property
    def is_metered(self) -> bool:
        """Whether :attr:`cost_usd` is real money that may be summed into a bill."""
        return self.billing_mode == "per_token"

    @property
    def is_priced(self) -> bool:
        """Whether a cost is known at all (``cost_source != "unavailable"``)."""
        return self.cost_usd is not None and self.cost_source != "unavailable"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_metadata(
        cls,
        meta: Optional[dict],
        *,
        model: str = "",
        billing_mode: BillingMode = "per_token",
    ) -> "TokenUsage":
        """Build a usage from a backend's ``_last_metadata`` dict.

        :param meta: Raw per-call metadata; aliased keys are normalized first, and
            ``None`` or a non-dict yields an all-zero usage.
        :param model: Model id to record and to price against. Falls back to
            ``meta["model"]``.
        :param billing_mode: One of :data:`BILLING_MODES`; see the module
            docstring for why this is not inferred from the numbers.
        :type meta: Optional[dict]
        :type model: str
        :type billing_mode: str
        :rtype: TokenUsage

        ``cost_source`` is resolved in one place, here: a provider-reported
        ``cost_usd`` in *meta* is ``"billed"``; otherwise a price-table estimate is
        attempted and recorded as ``"estimated"``; if that is unavailable the cost
        stays ``None`` and the source is ``"unavailable"``.
        """
        if not isinstance(meta, dict):
            meta = {}
        meta = normalize_usage_keys(meta)

        def _count(key: str) -> int:
            value = meta.get(key, 0)
            return int(value) if isinstance(value, (int, float)) else 0

        input_tokens = _count("input_tokens")
        output_tokens = _count("output_tokens")
        cache_read = _count("cache_read_input_tokens")
        cache_creation = _count("cache_creation_input_tokens")
        resolved_model = model or str(meta.get("model", "") or "")

        reported = meta.get("cost_usd")
        if isinstance(reported, (int, float)) and not isinstance(reported, bool):
            cost: Optional[float] = float(reported)
            cost_source: CostSource = "billed"
        else:
            cost = estimate_cost_usd(
                input_tokens,
                output_tokens,
                cache_read,
                cache_creation,
                model=resolved_model,
            )
            cost_source = "estimated" if cost is not None else "unavailable"

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            reasoning_tokens=_count("reasoning_tokens"),
            num_turns=_count("num_turns"),
            cost_usd=cost,
            cost_source=cost_source,
            billing_mode=billing_mode,
            model=resolved_model,
        )

    def as_metadata(self) -> dict:
        """Render back to a canonical metadata dict, dropping zeros and ``None``.

        The inverse of :meth:`from_metadata` for the keys it consumes, so a usage
        can be handed to anything that still speaks the flat dict vocabulary (the
        profiler's ``extra``, the aet sink).

        :rtype: dict
        """
        out: dict = {}
        for key in TOKEN_COUNT_KEYS + ("num_turns",):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.cost_usd is not None:
            out["cost_usd"] = self.cost_usd
        out["cost_source"] = self.cost_source
        out["billing_mode"] = self.billing_mode
        if self.model:
            out["model"] = self.model
        return out

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Sum two usages — token counts, turns, and cost.

        :raises TypeError: *other* is not a :class:`TokenUsage`.
        :raises ValueError: The two carry different :attr:`billing_mode` values.
            Subscription quota and metered spend are different units, and a sum of
            the two is neither; the caller must keep them in separate buckets.

        Cost addition is deliberately conservative: an unknown cost poisons the
        sum (the result is ``None`` with source ``"unavailable"``) rather than being
        treated as zero, because a total that silently omits an unpriced call reads
        as complete when it is not. Two known costs of differing sources combine to
        the weaker one — ``billed + estimated`` is ``"estimated"``.
        """
        if not isinstance(other, TokenUsage):
            return NotImplemented
        if self.billing_mode != other.billing_mode:
            raise ValueError(
                f"refusing to add usage across billing modes "
                f"({self.billing_mode!r} + {other.billing_mode!r}): subscription "
                f"quota is reported in dollar-equivalent terms and is not metered "
                f"spend, so the sum would be neither. Accumulate each mode "
                f"separately."
            )

        if self.cost_usd is None or other.cost_usd is None:
            cost: Optional[float] = None
            cost_source: CostSource = "unavailable"
        else:
            cost = self.cost_usd + other.cost_usd
            cost_source = (
                "billed"
                if self.cost_source == other.cost_source == "billed"
                else "estimated"
            )

        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            num_turns=self.num_turns + other.num_turns,
            cost_usd=cost,
            cost_source=cost_source,
            billing_mode=self.billing_mode,
            model=_merge_model(self.model, other.model),
        )

    def __radd__(self, other):
        """Support ``sum(usages)``, whose start value is the int ``0``."""
        if other == 0:
            return self
        return self.__add__(other)

    def replace(self, **changes) -> "TokenUsage":
        """Return a copy with *changes* applied (``dataclasses.replace``)."""
        return replace(self, **changes)


def _merge_model(left: str, right: str) -> str:
    """The model id two summed usages agree on, or ``""`` when they disagree.

    An empty id is "no opinion" rather than a disagreement, so accumulating a
    usage that never learned its model does not erase the one that did.
    """
    if left == right:
        return left
    if not left:
        return right
    if not right:
        return left
    return ""


def sum_usages(usages: Iterable[TokenUsage]) -> TokenUsage:
    """Accumulate *usages* into one, or an all-zero usage when empty.

    :param usages: Usages to combine; all must share a ``billing_mode``.
    :type usages: Iterable[TokenUsage]
    :rtype: TokenUsage
    :raises ValueError: Any two carry different ``billing_mode`` values (see
        :meth:`TokenUsage.__add__`).

    The empty case returns ``TokenUsage()``, whose ``cost_source`` is
    ``"unavailable"`` — "nothing was recorded", not "it was free".
    """
    total: Optional[TokenUsage] = None
    for usage in usages:
        total = usage if total is None else total + usage
    return total if total is not None else TokenUsage()
