"""Tests for the pricing the figures rest on.

The headline claim of this study is a ratio between two cost calculations, so the two
calculations are worth testing directly: a bug in either would move every figure and
nothing else would catch it.

Run with::

    pytest examples/harness_study/test_pricing.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pricing import (  # noqa: E402  (path setup must precede the import)
    Counts,
    error_ratio,
    price_four_class,
    price_two_class,
    rate_for,
)


# claude-sonnet-4-6: (input, output, cache_read, cache_creation) USD/Mtok.
SONNET = "claude-sonnet-4-6"


def test_rate_resolution_is_longest_key_wins():
    """A versioned id must match its version-specific tier, not the coarse family
    key — classic Opus is 15/75 while opus-4-6 is 5/25, a 3x difference."""
    assert rate_for("us.anthropic.claude-opus-4-6-v1") == (5.0, 25.0, 0.5, 6.25)
    assert rate_for("anthropic.claude-opus-4-1") == (15.0, 75.0, 1.5, 18.75)


def test_an_unknown_model_is_unpriceable_not_cheap():
    """The policy that matters: no rate yields None, never 0.0. A model priced at zero
    looks free, and free is the one answer that is never true."""
    assert rate_for("some.model-nobody-has-priced") is None
    unknown = "some.model-nobody-has-priced"
    assert price_four_class(Counts(input_tokens=1000), unknown) is None
    assert price_two_class(Counts(input_tokens=1000), unknown) is None
    assert error_ratio(Counts(input_tokens=1000), unknown) is None


def test_an_estimated_rate_is_flagged_as_one():
    """GLM's tuple is unconfirmed and Nova's cache rates are approximated, so a figure
    that prices either has to be able to say so rather than presenting every number with
    the same authority."""
    from pricing import rate_is_estimated

    assert rate_is_estimated("zai.glm-5") is True
    assert rate_is_estimated("us.amazon.nova-lite-v1:0") is True
    assert rate_is_estimated("us.anthropic.claude-sonnet-4-6") is False


def test_glm_prices_from_the_one_table_that_defines_it():
    """Sourced from oscar-merlin's bedrock_prices.yaml — the same table aet reads via
    AET_PRICE_TABLE — so a GLM cost here and there cannot disagree."""
    assert rate_for("zai.glm-5") == (0.60, 2.20, 0.06, 0.75)


def test_four_class_prices_each_class_at_its_own_rate():
    counts = Counts(input_tokens=1_000_000, output_tokens=1_000_000,
                    cache_read_tokens=1_000_000, cache_creation_tokens=1_000_000)

    # 3.00 + 15.00 + 0.30 + 3.75
    assert price_four_class(counts, SONNET) == pytest.approx(22.05)


def test_two_class_prices_all_input_at_the_fresh_rate():
    counts = Counts(input_tokens=1_000_000, output_tokens=1_000_000,
                    cache_read_tokens=1_000_000, cache_creation_tokens=1_000_000)

    # (1M + 1M + 1M) * 3.00 + 1M * 15.00
    assert price_two_class(counts, SONNET) == pytest.approx(24.0)


def test_the_two_agree_when_nothing_was_cached():
    """With no cache traffic there is nothing to get wrong, so the ratio is exactly
    1.0 — the figures' baseline."""
    counts = Counts(input_tokens=500_000, output_tokens=100_000)

    assert error_ratio(counts, SONNET) == pytest.approx(1.0)


def test_a_cache_read_heavy_run_is_overstated():
    """Reads cost a tenth of fresh input, so billing them as fresh inflates the
    figure — the ratio goes above 1."""
    counts = Counts(input_tokens=10, output_tokens=10, cache_read_tokens=1_000_000)

    ratio = error_ratio(counts, SONNET)

    assert ratio > 5.0


def test_a_cache_write_heavy_run_is_understated():
    """Writes carry a 25% premium over fresh input, so the ratio drops below 1 —
    which is why the error has no fixed sign and cannot be corrected with a
    constant."""
    counts = Counts(input_tokens=10, output_tokens=10,
                    cache_creation_tokens=1_000_000)

    ratio = error_ratio(counts, SONNET)

    assert ratio == pytest.approx(3.0 / 3.75, rel=1e-3)
    assert ratio < 1.0


def test_billed_input_covers_all_three_classes():
    counts = Counts(input_tokens=100, cache_read_tokens=660,
                    cache_creation_tokens=78)

    assert counts.billed_input == 838
    assert counts.cache_share == pytest.approx(738 / 838)


def test_cache_share_is_none_without_input():
    """No denominator means no ratio, not a zero — a call with no prompt at all did
    not have a cache miss."""
    assert Counts().cache_share is None


def test_a_zero_cost_run_has_no_ratio():
    """Dividing by a zero correct cost would be an infinity in a figure; the ratio is
    undefined instead."""
    assert error_ratio(Counts(), SONNET) is None
