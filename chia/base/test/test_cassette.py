"""Tests for :mod:`chia.base.cassette` and the ``recorded_prompt`` seam.

Three things are worth pinning, and only one of them is "a hit returns the recording":

* the key separates calls that must not share an entry — different model, different backend,
  different system prompt, different repetition;
* ``replay_only`` never falls through to the provider;
* nothing reaches disk with a credential in it.

The repetition case is the one that comes from a real bug rather than from symmetry: without a
discriminator, rep 2 of a grid cell is a cache hit on rep 1, and an experiment that looks like
two samples per cell has one.
"""

from __future__ import annotations

import json

import pytest

from chia.base.cassette import (
    CASSETTE_VERSION,
    Cassette,
    CassetteEntry,
    CassetteMiss,
    cassette_key,
)
from chia.base.llm_call import LLMCallBase, QueryResult
from chia.base.usage import TokenUsage


TOKEN = "bedrock-abcdef0123456789abcdef"


class FakeLLM(LLMCallBase):
    """A backend that counts its calls, so a hit is provable by the provider not running."""

    def __init__(self, *, model="fake-model-1", system_message="", reply="the answer",
                 stderr="", usage=None):
        super().__init__(system_message=system_message)
        self.model = model
        self.reply = reply
        self.stderr = stderr
        self.calls = 0
        self._usage = usage or TokenUsage(input_tokens=10, output_tokens=3, cost_usd=0.01,
                                         cost_source="billed", model=model)

    def prompt(self, user_message, tools=None):
        self.calls += 1
        return QueryResult(result=f"{self.reply} #{self.calls}", returncode=0,
                           stderr=self.stderr, stream_result=f"[turn] {self.reply}",
                           success=True, usage=self._usage)


def _entry(key="k" * 32, *, prompt="hi", result=None, **kw):
    return CassetteEntry(
        key=key, prompt=prompt, provider_id="FakeLLM",
        result=result or QueryResult(result="ok", returncode=0, stderr="", stream_result="",
                                     success=True, usage=TokenUsage(input_tokens=5)),
        **kw)


# --------------------------------------------------------------------------- the key
def test_the_key_is_stable_and_version_prefixed():
    first = cassette_key("hi", "FakeLLM", "m1")
    assert first == cassette_key("hi", "FakeLLM", "m1")
    assert len(first) == 32
    assert CASSETTE_VERSION.startswith("chia.cassette.")


@pytest.mark.parametrize("changed", [
    {"prompt": "different"},
    {"provider_id": "OtherLLM"},
    {"model_version": "m2"},
    {"system_message": "you are terse"},
    {"variant": "1"},
])
def test_every_part_of_the_key_separates_calls(changed):
    base = dict(prompt="hi", provider_id="FakeLLM", model_version="m1",
                system_message="", variant="")
    assert cassette_key(**base) != cassette_key(**{**base, **changed})


def test_two_repetitions_are_two_entries():
    """The bug this discriminator exists for.

    Repetition 2 of a condition sends the identical prompt as repetition 1. Without a variant
    it is a cache hit, and an experiment reporting two samples per cell has one sample twice.
    """
    assert cassette_key("hi", "FakeLLM", "m1", variant="0") != \
        cassette_key("hi", "FakeLLM", "m1", variant="1")


def test_a_null_model_version_hashes_as_empty():
    assert cassette_key("hi", "FakeLLM", None) == cassette_key("hi", "FakeLLM", "")


# --------------------------------------------------------------------------- the store
def test_a_stored_entry_round_trips_including_usage(tmp_path):
    """The chia-specific part: a replay re-derives the same cost with nothing spent."""
    cassette = Cassette(tmp_path)
    usage = TokenUsage(input_tokens=100, output_tokens=20, cache_read_input_tokens=4000,
                       cache_creation_input_tokens=500, cost_usd=0.0123,
                       cost_source="billed", model="m1", num_turns=2)
    cassette.put(_entry(result=QueryResult(result="ok", returncode=0, stderr="e",
                                           stream_result="s", success=True, usage=usage)))

    back = cassette.get("k" * 32)
    assert back is not None
    assert back.result.result == "ok"
    assert back.result.usage.input_tokens == 100
    assert back.result.usage.cache_read_input_tokens == 4000
    assert back.result.usage.cache_creation_input_tokens == 500
    assert back.result.usage.cost_usd == pytest.approx(0.0123)
    # cost_source survives: re-deriving it from the flat dict would silently downgrade a
    # provider-reported figure to "estimated"
    assert back.result.usage.cost_source == "billed"
    assert back.result.usage.num_turns == 2


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    """A truncated file from an interrupted run must not be able to abort a whole grid."""
    cassette = Cassette(tmp_path)
    path = cassette.path_for("k" * 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"key": "kkk", "result": {tru')

    assert cassette.get("k" * 32) is None


def test_no_partial_file_is_left_behind(tmp_path):
    """Write-then-rename: a half-written cassette read as the model's actual output would be a
    fabricated observation."""
    cassette = Cassette(tmp_path)
    cassette.put(_entry())
    assert list(tmp_path.rglob("*.partial")) == []
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_an_unknown_mode_is_refused():
    """Not treated as ``off``: a typo that silently disables recording is how a run ends up
    unreproducible while reporting that it was recorded."""
    with pytest.raises(ValueError, match="unknown cassette mode"):
        Cassette("/tmp/nope", mode="replay")


def test_off_reads_and_writes_nothing(tmp_path):
    cassette = Cassette(tmp_path, mode="off")
    assert cassette.put(_entry()) is None
    assert cassette.get("k" * 32) is None
    assert list(tmp_path.rglob("*.json")) == []


def test_entries_are_sharded(tmp_path):
    cassette = Cassette(tmp_path)
    key = cassette_key("hi", "FakeLLM", "m1")
    assert cassette.path_for(key).parent.name == key[:2]


# --------------------------------------------------------------------------- replay_only
def test_replay_only_raises_rather_than_calling_the_provider(tmp_path):
    """THE asymmetry. A run claiming to be a replay must not quietly become fresh model calls,
    because then its numbers are not the numbers being replayed."""
    llm = FakeLLM()
    llm.cassette = Cassette(tmp_path, mode="replay_only")

    with pytest.raises(CassetteMiss):
        llm.recorded_prompt("hi")
    assert llm.calls == 0


def test_replay_only_never_writes(tmp_path):
    cassette = Cassette(tmp_path, mode="replay_only")
    assert cassette.put(_entry()) is None
    assert list(tmp_path.rglob("*.json")) == []


# --------------------------------------------------------------------------- the seam
def test_no_cassette_is_a_plain_call(tmp_path):
    llm = FakeLLM()
    assert llm.recorded_prompt("hi").result == "the answer #1"
    assert llm.calls == 1


def test_a_second_identical_call_is_served_from_the_store(tmp_path):
    """Proven by the provider not running, not by comparing strings: FakeLLM's reply changes
    with its call count, so a live second call would be distinguishable."""
    llm = FakeLLM()
    llm.cassette = Cassette(tmp_path)

    first = llm.recorded_prompt("hi")
    second = llm.recorded_prompt("hi")

    assert llm.calls == 1
    assert second.result == first.result == "the answer #1"
    assert llm.cassette.stats.hits == 1
    assert llm.cassette.stats.misses == 1
    assert llm.cassette.stats.writes == 1


def test_a_different_variant_calls_the_provider_again(tmp_path):
    llm = FakeLLM()
    llm.cassette = Cassette(tmp_path)

    llm.recorded_prompt("hi", variant="0")
    llm.recorded_prompt("hi", variant="1")

    assert llm.calls == 2


def test_a_different_system_message_calls_the_provider_again(tmp_path):
    """chia keeps the system prompt on the instance, not in the prompt text, so two calls can
    send identical user messages under different system prompts."""
    store = Cassette(tmp_path)
    terse, verbose = FakeLLM(system_message="be terse"), FakeLLM(system_message="be verbose")
    terse.cassette = verbose.cassette = store

    terse.recorded_prompt("hi")
    verbose.recorded_prompt("hi")

    assert terse.calls == 1 and verbose.calls == 1
    assert store.stats.writes == 2


def test_the_provider_id_defaults_to_the_class_name():
    assert FakeLLM().provider_id == "FakeLLM"


def test_summary_reports_the_hit_rate(tmp_path):
    llm = FakeLLM()
    llm.cassette = Cassette(tmp_path)
    llm.recorded_prompt("a")
    llm.recorded_prompt("a")
    llm.recorded_prompt("b")

    summary = llm.cassette.summary()
    assert summary["mode"] == "record"
    assert (summary["hits"], summary["misses"]) == (1, 2)
    assert summary["hit_rate"] == pytest.approx(1 / 3)


def test_an_unused_cassette_has_no_hit_rate(tmp_path):
    """``None``, not ``0.0``: a run that made no calls did not have a 0% hit rate."""
    assert Cassette(tmp_path).summary()["hit_rate"] is None


# --------------------------------------------------------------------------- redaction
def test_a_credential_never_reaches_the_stored_file(tmp_path):
    """A cassette is written once and read for years, and it is the artifact most likely to be
    committed. A recorded stderr tail from an auth failure is exactly where a live bearer token
    would otherwise land on disk.
    """
    env = {"AWS_BEARER_TOKEN_BEDROCK": TOKEN}
    llm = FakeLLM(reply=f"leaked {TOKEN}", stderr=f"AccessDenied: token {TOKEN} expired")
    llm.cassette = Cassette(tmp_path, redact_env=env)

    llm.recorded_prompt(f"my key is {TOKEN}")

    on_disk = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert TOKEN not in on_disk
    # still diagnosable: the variable is named, and the surrounding text survives
    assert "AWS_BEARER_TOKEN_BEDROCK" in on_disk
    assert "AccessDenied" in on_disk
    # and the stored JSON is still valid, because the mask has no quote or backslash
    for path in tmp_path.rglob("*.json"):
        json.loads(path.read_text())


def test_the_masked_response_is_what_a_replay_returns(tmp_path):
    """The replay must not resurrect the secret the recording removed."""
    env = {"ANTHROPIC_API_KEY": TOKEN}
    llm = FakeLLM(reply=f"leaked {TOKEN}")
    llm.cassette = Cassette(tmp_path, redact_env=env)

    llm.recorded_prompt("hi")                      # records, masked
    replayed = FakeLLM(reply="unused")
    replayed.cassette = Cassette(tmp_path, mode="replay_only", redact_env=env)
    out = replayed.recorded_prompt("hi")

    assert TOKEN not in out.result
    assert replayed.calls == 0
