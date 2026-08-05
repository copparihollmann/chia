"""Token pricing, and the two ways of doing it that this study compares.

The claim under test is that pricing a prompt-cached agent loop with a *two-class*
model (input, output) rather than a *four-class* one (input, output, cache read,
cache write) is not a rounding error. Both are implemented here as pure functions
over the same counts so the difference is a number, not an argument.

Rates are USD per million tokens, keyed by a substring of the model id, resolved
longest-key-wins so a versioned id (``us.anthropic.claude-opus-4-6-v1``) matches the
version-specific entry rather than the coarse family one. They are duplicated here
rather than imported from aet on purpose: a figure has to be reproducible from this
directory alone, and a rate table that can change under the plot is not evidence.
The provenance of each entry is in the comment beside it.

An unknown model is **cost-unavailable**, not cheap: :func:`price` returns ``None``,
matching chia's own rule (see :mod:`chia.base.usage`) that a fabricated ``0.0`` is
indistinguishable from a genuinely free call once summed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: ``model-key -> (input, output, cache_read, cache_creation)`` in USD per Mtok.
#:
#: The Claude entries follow Anthropic's published multipliers (output = 5x input,
#: cache read = 0.1x, cache write = 1.25x). The version-specific 4-6 families are
#: billed at a different tier from the classic list price and must resolve ahead of
#: the coarse family keys, which is what longest-key-wins gives.
RATES_USD_PER_MTOK: Dict[str, Tuple[float, float, float, float]] = {
    "opus-4-6":   (5.0, 25.0, 0.5, 6.25),
    "sonnet-4-6": (3.0, 15.0, 0.3, 3.75),
    "haiku-4-5":  (1.0, 5.0, 0.1, 1.25),
    # Classic Claude list price, kept as the fallback for versions with no entry.
    "opus":       (15.0, 75.0, 1.5, 18.75),
    "sonnet":     (3.0, 15.0, 0.3, 3.75),
    "haiku":      (0.80, 4.0, 0.08, 1.0),
    # Amazon Nova on-demand list price. Nova does not publish separate cache rates,
    # so cache_read is approximated at 0.25x input and a cache write at one input
    # pass — stated here rather than hidden, since it affects any Nova figure.
    "nova-micro": (0.035, 0.14, 0.035 * 0.25, 0.035),
    "nova-lite":  (0.06, 0.24, 0.06 * 0.25, 0.06),
    "nova-pro":   (0.80, 3.20, 0.80 * 0.25, 0.80),
}


def rate_for(model: str) -> Optional[Tuple[float, float, float, float]]:
    """The rate tuple for *model*, or ``None`` when no key matches.

    :param model: A provider model id, in any of the vendors' spellings.
    :type model: str
    :rtype: Optional[Tuple[float, float, float, float]]

    Resolution is longest-key-wins, with ties broken on the key string so the result
    does not depend on dict ordering.
    """
    lowered = (model or "").lower()
    best: Optional[str] = None
    for key in RATES_USD_PER_MTOK:
        if key in lowered and (best is None or (len(key), key) > (len(best), best)):
            best = key
    return RATES_USD_PER_MTOK[best] if best is not None else None


@dataclass(frozen=True)
class Counts:
    """The four token classes of one call or run.

    :param input_tokens: Fresh input, excluding anything cached.
    :param output_tokens: Generated tokens.
    :param cache_read_tokens: Prompt tokens served from the cache.
    :param cache_creation_tokens: Prompt tokens written into the cache.
    :type input_tokens: int
    :type output_tokens: int
    :type cache_read_tokens: int
    :type cache_creation_tokens: int
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def billed_input(self) -> int:
        """All three input classes together."""
        return (self.input_tokens + self.cache_read_tokens
                + self.cache_creation_tokens)

    @property
    def cache_share(self) -> Optional[float]:
        """Cache tokens as a fraction of billed input, or ``None`` with no input."""
        if not self.billed_input:
            return None
        return (self.cache_read_tokens + self.cache_creation_tokens) / self.billed_input


def price_four_class(counts: Counts, model: str) -> Optional[float]:
    """Cost with each input class at its own rate — the correct calculation.

    :rtype: Optional[float]
    """
    rate = rate_for(model)
    if rate is None:
        return None
    r_in, r_out, r_read, r_write = rate
    return (
        counts.input_tokens * r_in
        + counts.output_tokens * r_out
        + counts.cache_read_tokens * r_read
        + counts.cache_creation_tokens * r_write
    ) / 1e6


def price_two_class(counts: Counts, model: str) -> Optional[float]:
    """Cost with every input token priced at the fresh-input rate.

    :rtype: Optional[float]

    This is what a backend that records only ``input_tokens`` / ``output_tokens``
    produces once its counts are handed to a pricing step: whatever the cache did is
    invisible, so all of it is billed as fresh input. Cache reads cost a tenth of
    fresh input and cache writes a quarter more, so the error has no fixed sign —
    which is exactly why it cannot be corrected after the fact with a fudge factor.
    """
    rate = rate_for(model)
    if rate is None:
        return None
    r_in, r_out, _, _ = rate
    return (counts.billed_input * r_in + counts.output_tokens * r_out) / 1e6


def error_ratio(counts: Counts, model: str) -> Optional[float]:
    """``price_two_class / price_four_class``, or ``None`` when unpriceable.

    :rtype: Optional[float]

    Above 1.0 means the two-class calculation *overstates* cost (a cache-read-heavy
    run, since reads are cheap); below 1.0 means it understates (a cache-write-heavy
    run, since writes carry a premium).
    """
    correct = price_four_class(counts, model)
    naive = price_two_class(counts, model)
    if correct is None or naive is None or correct == 0:
        return None
    return naive / correct
