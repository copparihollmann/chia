"""Canonical usage-metadata keys shared across the model backends.

Every backend accumulates per-call token/cost metadata into ``_last_metadata``.
Historically the key names drifted between clients (e.g. opencode used
``cache_read`` / ``cache_write`` where the others used
``cache_read_input_tokens`` / ``cache_creation_input_tokens``). This module
pins the canonical set and offers a tiny normalizer so downstream consumers
(the profiler, the aet sink) see one vocabulary regardless of backend.

The canonical keys are::

    input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens,
    reasoning_tokens, cost_usd, num_turns

Cost estimation is delegated to aet's :class:`~aet.trajectory.pricing.PriceTable`
when aet is importable; when it is not (chia stays standalone), the estimate is
simply unavailable and callers must omit cost rather than fabricate ``0``.
"""

from __future__ import annotations

from typing import Optional


# The canonical usage vocabulary. Kept as a tuple so callers can iterate it.
CANONICAL_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
    "cost_usd",
    "num_turns",
)

# Non-canonical alias -> canonical key. Only the aliases that a backend actually
# emits need listing (opencode's short cache keys today).
_USAGE_KEY_ALIASES = {
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_input_tokens",
}


def normalize_usage_keys(meta: dict) -> dict:
    """Return *meta* with any aliased usage keys renamed to the canonical set.

    Non-usage keys (``model``, ``tools``, ...) pass through untouched. When both
    an alias and its canonical target are present, their (numeric) values are
    summed so no count is dropped; otherwise the value is moved verbatim.
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
    """Best-effort USD cost estimate via aet's ``PriceTable``.

    Returns ``None`` when the price is unknown *or* when aet is not installed —
    in both cases the caller must treat the cost as unknown and omit it, never
    substitute ``0``.
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
