"""Tests for the shared usage-metadata normalizer (:mod:`chia.models.usage`)."""

from __future__ import annotations

from chia.models.usage import (
    CANONICAL_USAGE_KEYS,
    estimate_cost_usd,
    normalize_usage_keys,
)


def test_normalize_maps_opencode_cache_keys():
    # opencode emits cache_read / cache_write; they must become the canonical
    # cache_read_input_tokens / cache_creation_input_tokens.
    meta = {
        "input_tokens": 10,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "cache_read": 100,
        "cache_write": 40,
        "cost_usd": 0.01,
        "num_turns": 3,
    }
    out = normalize_usage_keys(meta)
    assert out["cache_read_input_tokens"] == 100
    assert out["cache_creation_input_tokens"] == 40
    assert "cache_read" not in out
    assert "cache_write" not in out
    # Everything else is preserved verbatim.
    assert out["input_tokens"] == 10
    assert out["output_tokens"] == 5
    assert out["reasoning_tokens"] == 2
    assert out["cost_usd"] == 0.01
    assert out["num_turns"] == 3
    # Result speaks only the canonical vocabulary.
    assert set(out).issubset(set(CANONICAL_USAGE_KEYS))


def test_normalize_passes_through_non_usage_keys():
    meta = {"model": "claude", "tools": [{"name": "x"}], "cache_read": 7}
    out = normalize_usage_keys(meta)
    assert out["model"] == "claude"
    assert out["tools"] == [{"name": "x"}]
    assert out["cache_read_input_tokens"] == 7


def test_normalize_sums_alias_and_canonical_collision():
    # If both the alias and its canonical target are present, values sum so no
    # count is dropped.
    meta = {"cache_read": 3, "cache_read_input_tokens": 4}
    out = normalize_usage_keys(meta)
    assert out["cache_read_input_tokens"] == 7


def test_estimate_cost_unknown_model_is_none_not_zero():
    # A model with no known price must yield None (cost unknown), never 0.
    cost = estimate_cost_usd(100, 50, model="totally-unknown-model-xyz")
    assert cost is None
