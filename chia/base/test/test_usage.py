"""Tests for the canonical token-usage accounting.

Offline and dependency-free: nothing here spawns a subprocess, contacts a
provider, or requires ``aet`` (the one test that needs a known price stubs the
estimator, so the suite behaves identically with and without aet installed).

Covers the three accounting rules :mod:`chia.base.usage` exists to enforce:

  1. The three input classes stay separate (``test_four_input_classes_*``).
  2. An unknown price is ``None``, never ``0.0`` (``test_unknown_price_*``).
  3. Subscription quota is never summed with metered spend (``test_add_refuses_*``).

Plus the plumbing that makes them reachable from a caller: alias normalization,
``QueryResult.usage`` defaults, and ``LLMCallBase.attach_usage``.
"""
import pytest

from chia.base.llm_call import LLMCallBase, QueryResult
from chia.base.usage import (
    BILLING_MODES,
    COST_SOURCES,
    TokenUsage,
    normalize_usage_keys,
    sum_usages,
)


# A model id no price table can know, so estimate_cost_usd returns None whether
# or not aet is installed. Using a deliberately absurd id (rather than a plausible
# one that a future price table might learn) keeps the unknown-price tests stable.
UNPRICED_MODEL = "chia-test-model-with-no-price-9e3f"


class _FakeLLM(LLMCallBase):
    """Minimal concrete LLMCallBase, to exercise attach_usage without a provider."""

    def __init__(self, model="", billing_mode=None):
        super().__init__(system_message="")
        self.model = model
        self._last_metadata = {}
        if billing_mode is not None:
            self.billing_mode = billing_mode

    def prompt(self, user_message, tools=None):  # pragma: no cover - never called
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rule 1 — the three input classes stay separate
# ---------------------------------------------------------------------------


def test_four_input_classes_survive_from_metadata():
    usage = TokenUsage.from_metadata({
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 15_345,
        "cache_creation_input_tokens": 900,
        "reasoning_tokens": 7,
        "num_turns": 3,
    }, model=UNPRICED_MODEL)

    assert usage.input_tokens == 100
    assert usage.cache_read_input_tokens == 15_345
    assert usage.cache_creation_input_tokens == 900
    assert usage.reasoning_tokens == 7
    assert usage.num_turns == 3


def test_four_input_classes_sum_into_billed_input():
    """billed_input_tokens is the only place the classes are collapsed, and it is
    documented as a size, not a basis for costing."""
    usage = TokenUsage(
        input_tokens=100,
        cache_read_input_tokens=660,
        cache_creation_input_tokens=78,
        output_tokens=40,
    )

    assert usage.billed_input_tokens == 838
    assert usage.total_tokens == 878
    assert usage.cache_hit_ratio == pytest.approx(660 / 838)


def test_cache_hit_ratio_is_none_without_input():
    """No denominator means no ratio — not a zero, which would read as "no cache
    hits" on a call that had no prompt at all."""
    assert TokenUsage().cache_hit_ratio is None


# ---------------------------------------------------------------------------
# Rule 2 — an unknown price is None, never 0.0
# ---------------------------------------------------------------------------


def test_unknown_price_yields_none_not_zero():
    usage = TokenUsage.from_metadata(
        {"input_tokens": 1000, "output_tokens": 500}, model=UNPRICED_MODEL,
    )

    assert usage.cost_usd is None
    assert usage.cost_source == "unavailable"
    assert usage.is_priced is False


def test_unknown_price_is_distinguishable_from_a_free_call():
    """The regression this rule exists for: an unpriced call and a genuinely
    zero-cost one must not compare equal once summed."""
    unpriced = TokenUsage.from_metadata(
        {"input_tokens": 10}, model=UNPRICED_MODEL,
    )
    free = TokenUsage.from_metadata({"input_tokens": 10, "cost_usd": 0.0})

    assert unpriced.cost_usd is None
    assert free.cost_usd == 0.0
    assert free.cost_source == "billed"


def test_provider_reported_cost_is_billed():
    usage = TokenUsage.from_metadata({
        "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0431,
    }, model=UNPRICED_MODEL)

    assert usage.cost_usd == pytest.approx(0.0431)
    assert usage.cost_source == "billed"


def test_derived_cost_is_estimated(monkeypatch):
    """With no provider-reported cost, a price-table hit is recorded as an
    estimate — never conflated with a billed figure."""
    monkeypatch.setattr("chia.base.usage.estimate_cost_usd", lambda *a, **k: 0.018)

    usage = TokenUsage.from_metadata(
        {"input_tokens": 1000, "output_tokens": 1000}, model="some-priced-model",
    )

    assert usage.cost_usd == pytest.approx(0.018)
    assert usage.cost_source == "estimated"
    assert usage.is_priced is True


def test_cost_source_is_always_one_of_the_declared_values():
    for meta in ({}, {"cost_usd": 1.0}, {"input_tokens": 5}):
        usage = TokenUsage.from_metadata(meta, model=UNPRICED_MODEL)
        assert usage.cost_source in COST_SOURCES
        assert usage.billing_mode in BILLING_MODES


def test_a_bool_cost_is_not_mistaken_for_a_price():
    """bool is a subclass of int, so a stray truthy flag would otherwise be read
    as $1.00 of billed spend."""
    usage = TokenUsage.from_metadata(
        {"input_tokens": 5, "cost_usd": True}, model=UNPRICED_MODEL,
    )

    assert usage.cost_usd is None
    assert usage.cost_source == "unavailable"


# ---------------------------------------------------------------------------
# Rule 3 — subscription quota is never summed with metered spend
# ---------------------------------------------------------------------------


def test_add_refuses_across_billing_modes():
    metered = TokenUsage(input_tokens=10, cost_usd=0.01, cost_source="billed")
    quota = TokenUsage(input_tokens=10, cost_usd=0.02, cost_source="billed",
                       billing_mode="subscription")

    with pytest.raises(ValueError) as exc:
        metered + quota

    assert "billing mode" in str(exc.value)


def test_add_within_a_billing_mode_sums_tokens_and_cost():
    a = TokenUsage(input_tokens=10, output_tokens=1, cache_read_input_tokens=100,
                   num_turns=1, cost_usd=0.01, cost_source="billed", model="m")
    b = TokenUsage(input_tokens=20, output_tokens=2, cache_read_input_tokens=200,
                   num_turns=2, cost_usd=0.02, cost_source="billed", model="m")

    total = a + b

    assert (total.input_tokens, total.output_tokens) == (30, 3)
    assert total.cache_read_input_tokens == 300
    assert total.num_turns == 3
    assert total.cost_usd == pytest.approx(0.03)
    assert total.cost_source == "billed"
    assert total.model == "m"


def test_add_degrades_cost_source_to_the_weaker_of_the_two():
    billed = TokenUsage(cost_usd=1.0, cost_source="billed")
    estimated = TokenUsage(cost_usd=2.0, cost_source="estimated")

    assert (billed + estimated).cost_source == "estimated"


def test_add_lets_an_unknown_cost_poison_the_total():
    """A sum that silently skipped the unpriced call would read as complete."""
    known = TokenUsage(input_tokens=10, cost_usd=1.0, cost_source="billed")
    unknown = TokenUsage(input_tokens=10)

    total = known + unknown

    assert total.input_tokens == 20          # token counts still accumulate
    assert total.cost_usd is None
    assert total.cost_source == "unavailable"


def test_add_forgets_the_model_only_when_the_two_disagree():
    same = TokenUsage(model="a") + TokenUsage(model="a")
    unknown_side = TokenUsage(model="a") + TokenUsage(model="")
    conflict = TokenUsage(model="a") + TokenUsage(model="b")

    assert same.model == "a"
    assert unknown_side.model == "a"
    assert conflict.model == ""


def test_sum_usages_over_an_empty_iterable_is_unpriced_not_free():
    total = sum_usages([])

    assert total.total_tokens == 0
    assert total.cost_usd is None
    assert total.cost_source == "unavailable"


def test_sum_usages_matches_repeated_addition():
    parts = [TokenUsage(input_tokens=i, cost_usd=float(i), cost_source="billed")
             for i in range(1, 5)]

    assert sum_usages(parts) == parts[0] + parts[1] + parts[2] + parts[3]


def test_builtin_sum_works_despite_its_integer_start_value():
    parts = [TokenUsage(input_tokens=1), TokenUsage(input_tokens=2)]

    assert sum(parts).input_tokens == 3


# ---------------------------------------------------------------------------
# Alias normalization — one vocabulary across backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias,canonical", [
    ("cache_read", "cache_read_input_tokens"),          # opencode
    ("cache_write", "cache_creation_input_tokens"),      # opencode
    ("cached_input_tokens", "cache_read_input_tokens"),  # codex
    ("prompt_tokens", "input_tokens"),                   # OpenAI completions
    ("completion_tokens", "output_tokens"),              # OpenAI completions
])
def test_normalize_usage_keys_renames_every_known_alias(alias, canonical):
    assert normalize_usage_keys({alias: 42}) == {canonical: 42}


def test_normalize_usage_keys_sums_rather_than_drops_a_collision():
    """A backend emitting both spellings must not lose one of them."""
    out = normalize_usage_keys({"cache_read": 10, "cache_read_input_tokens": 5})

    assert out == {"cache_read_input_tokens": 15}


def test_normalize_usage_keys_passes_non_usage_keys_through():
    out = normalize_usage_keys({"model": "m", "tools": [], "duration_s": 1.5})

    assert out == {"model": "m", "tools": [], "duration_s": 1.5}


def test_from_metadata_normalizes_before_reading():
    """The opencode path end to end: short cache keys reach the canonical fields."""
    usage = TokenUsage.from_metadata({
        "input_tokens": 1, "cache_read": 100, "cache_write": 10,
    }, model=UNPRICED_MODEL)

    assert usage.cache_read_input_tokens == 100
    assert usage.cache_creation_input_tokens == 10


def test_from_metadata_tolerates_none_and_junk():
    """Error paths hand over no metadata at all; the result must still be a
    readable all-zero usage rather than an exception or a None."""
    for meta in (None, {}, {"input_tokens": "not a number"}, ["not a dict"]):
        usage = TokenUsage.from_metadata(meta)
        assert usage.input_tokens == 0
        assert usage.cost_usd is None


def test_from_metadata_falls_back_to_the_model_in_the_metadata():
    usage = TokenUsage.from_metadata({"input_tokens": 1, "model": "from-meta"})

    assert usage.model == "from-meta"


# ---------------------------------------------------------------------------
# as_metadata — the round trip back to the flat dict vocabulary
# ---------------------------------------------------------------------------


def test_as_metadata_drops_zeros_but_keeps_the_annotations():
    usage = TokenUsage(input_tokens=5, cost_usd=0.5, cost_source="billed",
                       billing_mode="subscription", model="m")

    out = usage.as_metadata()

    assert out == {
        "input_tokens": 5, "cost_usd": 0.5,
        "cost_source": "billed", "billing_mode": "subscription", "model": "m",
    }
    assert "output_tokens" not in out


def test_as_metadata_omits_an_unknown_cost_entirely():
    out = TokenUsage(input_tokens=5).as_metadata()

    assert "cost_usd" not in out
    assert out["cost_source"] == "unavailable"


def test_metadata_round_trip_preserves_the_counts():
    usage = TokenUsage(input_tokens=1, output_tokens=2, cache_read_input_tokens=3,
                       cache_creation_input_tokens=4, reasoning_tokens=5,
                       num_turns=6, cost_usd=0.7, cost_source="billed", model="m")

    again = TokenUsage.from_metadata(usage.as_metadata())

    assert again == usage


# ---------------------------------------------------------------------------
# The public surface: QueryResult.usage and LLMCallBase.attach_usage
# ---------------------------------------------------------------------------


def test_query_result_usage_defaults_to_an_all_zero_usage():
    """Never None, so `result.usage.input_tokens` is always safe to read."""
    result = QueryResult(result="", returncode=0, stderr="", stream_result="")

    assert isinstance(result.usage, TokenUsage)
    assert result.usage.total_tokens == 0
    assert result.usage.cost_source == "unavailable"


def test_query_result_usage_is_not_shared_between_instances():
    """A mutable default would alias every result to one object."""
    a = QueryResult(result="", returncode=0, stderr="", stream_result="")
    b = QueryResult(result="", returncode=0, stderr="", stream_result="")

    assert a.usage is not b.usage


def test_attach_usage_reads_the_instances_last_metadata_by_default():
    llm = _FakeLLM(model=UNPRICED_MODEL)
    llm._last_metadata = {"input_tokens": 7, "output_tokens": 3}
    result = QueryResult(result="", returncode=0, stderr="", stream_result="")

    llm.attach_usage(result)

    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 3
    assert result.usage.model == UNPRICED_MODEL


def test_attach_usage_stamps_the_backends_billing_mode():
    """The reason attach_usage is on the base class: the mode comes from the
    backend, so no caller has to know which backends are seat-authenticated."""
    llm = _FakeLLM(model=UNPRICED_MODEL, billing_mode="subscription")
    llm._last_metadata = {"input_tokens": 1, "cost_usd": 0.25}
    result = QueryResult(result="", returncode=0, stderr="", stream_result="")

    llm.attach_usage(result)

    assert result.usage.billing_mode == "subscription"
    assert result.usage.is_metered is False


def test_default_billing_mode_is_metered():
    """per_token is the safe default: it is the only mode whose cost may be
    summed, so a backend that forgets to declare one is not silently excluded."""
    assert _FakeLLM().billing_mode == "per_token"


def test_attach_usage_returns_the_result_for_chaining():
    llm = _FakeLLM()
    result = QueryResult(result="", returncode=0, stderr="", stream_result="")

    assert llm.attach_usage(result, {}) is result
