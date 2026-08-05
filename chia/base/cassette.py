"""A content-addressed store of prompt/response pairs, so a published number stays re-derivable.

An LLM is not a function. The same prompt gives a different answer tomorrow, and the model
behind ``claude -p`` will be replaced. chia ships examples that publish numbers —
``examples/harness_study`` publishes a cost table, ``examples/aet_trajectory`` publishes a
spend curve — and without a recording those are claims about a system that no longer exists.
With one, anybody can re-derive every metric from the exact bytes the model produced, with the
provider disabled and nothing spent.

What is in the key
------------------

``sha256`` over ``CASSETTE_VERSION``, ``provider_id``, ``model_version``, ``system_message``,
``variant`` and the prompt. Each part earns its place:

* **prompt** — it is the input.
* **model_version** — a response from a different model is a different observation, and
  replaying one under another model's name is the quietest kind of dishonesty.
* **provider_id** — the same prompt through a subprocess CLI and through an API is not the
  same call. In chia this is the backend class name, so ``ClaudeCodeLLM`` and ``BedrockLLM``
  never share an entry even on an identical model id.
* **system_message** — chia keeps this on the *instance* rather than in the prompt text, so two
  calls can send identical ``user_message`` under different system prompts. Leaving it out of
  the key would collapse them.
* **variant** — the repetition index. **This is in the key because of a real bug, not for
  symmetry.** Repetition 2 of a condition sends the identical prompt as repetition 1, so
  without a discriminator it is a cache hit. In the reference implementation's first live grid
  that produced byte-identical 63,230-byte responses for reps 1 and 2: what looked like a
  24-call experiment with two samples per cell was 12 calls with no replication at all. A
  cassette records *one call*, and two repetitions are two calls.
* **version prefix** — so a change to the key recipe invalidates every cassette at once, in
  one visible diff, rather than leaving a matching subset while the rest miss.

Note what ``variant`` does *not* do: it does not make the provider stochastic. If a backend
returns the same text for the same prompt regardless, two repetitions are two calls that happen
to agree, and reporting an interval over them would claim precision that does not exist.

Two modes matter, and they are not symmetric
--------------------------------------------

``record``
    A miss calls the provider and stores the result.
``replay_only``
    A miss is a hard :class:`CassetteMiss`, **never** a silent live call. A run claiming to be a
    replay must not quietly become a fresh set of model calls, because then the numbers it
    produces are not the numbers being replayed.
``off``
    Nothing is read or written. Named, so "not recording" is a recorded choice.

Usage is stored, which is the part specific to chia
---------------------------------------------------

An entry carries the whole :class:`~chia.base.llm_call.QueryResult`, including its
:class:`~chia.base.usage.TokenUsage`. So a replay re-derives the same cost table without
spending anything — which is the whole point for a published figure.

**A replayed usage is the original call's usage.** It must not be added to a spend ledger as
new money. :meth:`Cassette.summary` exists so a run manifest can record how much of a run was
replayed; a rollup that ignores it will double-count.

Credentials are masked before anything is written
-------------------------------------------------

A cassette is the most durable artifact in the system — it is written once and read for years,
and it is the thing most likely to be committed. Every stored text goes through
:func:`chia.base.redact.redact` first. A recorded stderr tail from an auth failure is exactly
where a live bearer token would otherwise end up on disk.

Pure ``hashlib`` + ``json``. No network, no provider imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from chia.base.llm_call import QueryResult
from chia.base.redact import redact
from chia.base.usage import TokenUsage

PathLike = Union[str, os.PathLike]

#: Bump when the key recipe changes. Every cassette then misses together, visibly, rather than
#: leaving a matching subset behind.
CASSETTE_VERSION = "chia.cassette.v1"

#: The modes. ``off`` is named rather than implied, so a run that is not recording says so.
MODES = ("record", "replay_only", "off")

_JSON = {"sort_keys": True, "indent": 1, "ensure_ascii": False}


class CassetteMiss(LookupError):
    """A ``replay_only`` lookup found no recording. Never falls through to the provider."""


def cassette_key(prompt: str, provider_id: str, model_version: Optional[str] = None, *,
                 system_message: str = "", variant: str = "") -> str:
    """The content address of one call.

    :param prompt: The user message.
    :param provider_id: Backend identity — the class name, in chia.
    :param model_version: The model id. ``None`` hashes as the empty string.
    :param system_message: The instance's system prompt, which chia keeps outside the prompt.
    :param variant: Discriminator for two calls that send the same bytes; in practice the
        repetition index. See the module docstring for the bug that made this necessary.
    :type prompt: str
    :type provider_id: str
    :type model_version: Optional[str]
    :type system_message: str
    :type variant: str
    :rtype: str

    Parts are NUL-separated, so there is no delimiter ambiguity between a provider id ending in
    ``x`` and a model version beginning with ``y``.
    """
    digest = hashlib.sha256()
    for part in (CASSETTE_VERSION, provider_id, model_version or "", system_message,
                 variant, prompt):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class CassetteEntry:
    """One recorded call, response and accounting together.

    :param key: The content address.
    :param prompt: The user message that produced it.
    :param provider_id: Backend identity.
    :param result: The recorded ``QueryResult``, usage included.
    :param model_version: The model id.
    :param system_message: The system prompt in force.
    :param variant: The discriminator this entry was recorded under.
    :param original_wall_s: Wall time of the *original* call. A replay did not spend it, so a
        replayed row must not report it as its own — that is how a timing table starts
        double-counting.
    :type key: str
    :type prompt: str
    :type provider_id: str
    :type result: chia.base.llm_call.QueryResult
    :type model_version: Optional[str]
    :type system_message: str
    :type variant: str
    :type original_wall_s: Optional[float]
    """

    key: str
    prompt: str
    provider_id: str
    result: QueryResult
    model_version: Optional[str] = None
    system_message: str = ""
    variant: str = ""
    original_wall_s: Optional[float] = None

    def to_dict(self) -> dict:
        payload = {
            "cassette_version": CASSETTE_VERSION,
            "key": self.key,
            "provider_id": self.provider_id,
            "model_version": self.model_version,
            "prompt": self.prompt,
            "result": {
                "result": self.result.result,
                "returncode": self.result.returncode,
                "stderr": self.result.stderr,
                "stream_result": self.result.stream_result,
                "success": self.result.success,
                "usage": self.result.usage.as_metadata(),
            },
        }
        if self.system_message:
            payload["system_message"] = self.system_message
        if self.variant:
            payload["variant"] = self.variant
        if self.original_wall_s is not None:
            payload["original_wall_s"] = self.original_wall_s
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "CassetteEntry":
        raw = payload.get("result") or {}
        usage_meta = raw.get("usage") or {}
        result = QueryResult(
            result=str(raw.get("result", "")),
            returncode=int(raw.get("returncode", 0) or 0),
            stderr=str(raw.get("stderr", "")),
            stream_result=str(raw.get("stream_result", "")),
            success=bool(raw.get("success", False)),
            usage=TokenUsage.from_metadata(
                usage_meta,
                model=str(usage_meta.get("model", "") or payload.get("model_version") or ""),
                billing_mode=str(usage_meta.get("billing_mode", "per_token")),
            ),
        )
        # A cost_source recorded at capture time is authoritative and must survive the round
        # trip: re-deriving it from the flat dict would silently downgrade a provider-reported
        # "billed" figure to "estimated", which is the exact distinction the four-class
        # accounting exists to preserve.
        source = usage_meta.get("cost_source")
        if source in ("billed", "estimated", "unavailable"):
            result.usage = result.usage.replace(cost_source=source)
        return cls(
            key=str(payload["key"]),
            prompt=str(payload.get("prompt", "")),
            provider_id=str(payload.get("provider_id", "")),
            result=result,
            model_version=payload.get("model_version"),
            system_message=str(payload.get("system_message", "")),
            variant=str(payload.get("variant", "")),
            original_wall_s=payload.get("original_wall_s"),
        )


@dataclass
class CassetteStats:
    """What a run's cassette did. Recorded in a manifest so a replay is identifiable as one.

    :param hits: Calls served from the store.
    :param misses: Calls the store did not have.
    :param writes: Entries written.
    :type hits: int
    :type misses: int
    :type writes: int
    """

    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> Optional[float]:
        """Fraction served from the store, or ``None`` when nothing was looked up.

        ``None`` rather than ``0.0``: a run that made no calls did not have a 0% hit rate.
        """
        return self.hits / self.lookups if self.lookups else None

    def to_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes,
                "lookups": self.lookups, "hit_rate": self.hit_rate}


class Cassette:
    """A directory of recorded calls.

    :param root: Where entries live.
    :param mode: One of :data:`MODES`.
    :param redact_env: Environment consulted for credential values to mask before writing.
        Defaults to the real environment; pass a dict in tests.
    :type root: PathLike
    :type mode: str
    :type redact_env: Optional[dict]
    :raises ValueError: *mode* is not one of :data:`MODES`. Refused up front rather than
        treated as ``off``, because a typo that silently disables recording is how a run ends
        up unreproducible while reporting that it was recorded.
    """

    def __init__(self, root: PathLike, mode: str = "record",
                 redact_env: Optional[dict] = None):
        if mode not in MODES:
            raise ValueError(f"unknown cassette mode {mode!r}; expected one of {MODES}")
        self.root = Path(root)
        self.mode = mode
        self.stats = CassetteStats()
        self._redact_env = redact_env
        # chia fans provider calls out across Ray workers, and a driver-side cassette is shared
        # by every thread that joins them. Unsynchronized `+= 1` would under-report the hit
        # rate -- a statistic about reproducibility that is itself unreproducible.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths

    def path_for(self, key: str) -> Path:
        """Sharded by the first two hex characters, so a grid of thousands stays navigable."""
        return self.root / key[:2] / f"{key}.json"

    # ------------------------------------------------------------------ access

    def get(self, key: str) -> Optional[CassetteEntry]:
        """The entry for *key*, or ``None``. Never raises.

        A corrupt or truncated file is a miss rather than a crash: in ``record`` mode the call
        is simply remade, and in ``replay_only`` it surfaces as a :class:`CassetteMiss`, which
        is the honest outcome. A half-written file from an interrupted run must not be able to
        abort a whole grid.
        """
        if self.mode == "off":
            return None
        path = self.path_for(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            return CassetteEntry.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, entry: CassetteEntry) -> Optional[Path]:
        """Store *entry*, masking credentials first. A no-op outside ``record`` mode."""
        if self.mode != "record":
            return None
        path = self.path_for(entry.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._redacted(entry).to_dict()
        # Write-then-rename, so an interrupted run cannot leave a half-written cassette that a
        # later replay would read as the model's actual output.
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(payload, **_JSON) + "\n", encoding="utf-8")
        partial.replace(path)
        with self._lock:
            self.stats.writes += 1
        return path

    def lookup(self, prompt: str, provider_id: str, model_version: Optional[str] = None, *,
               system_message: str = "", variant: str = "",
               ) -> Tuple[str, Optional[CassetteEntry]]:
        """``(key, entry_or_None)``, updating the counters.

        :raises CassetteMiss: In ``replay_only`` mode. Falling through to the provider would
            make a "replay" produce fresh numbers under a replay's name.
        """
        key = cassette_key(prompt, provider_id, model_version,
                           system_message=system_message, variant=variant)
        entry = self.get(key)
        if entry is None:
            with self._lock:
                self.stats.misses += 1
            if self.mode == "replay_only":
                raise CassetteMiss(
                    f"no recorded response for key {key} (provider {provider_id!r}, "
                    f"model {model_version!r}, variant {variant!r}); run in 'record' mode "
                    f"to record it")
            return key, None
        with self._lock:
            self.stats.hits += 1
        return key, entry

    def summary(self) -> dict:
        """The run's cassette statistics, for a manifest.

        Worth writing down: a spend rollup that treats a replayed run's recorded usage as new
        money double-counts, and ``hit_rate`` is what tells it not to.
        """
        return {"mode": self.mode, "root": str(self.root), **self.stats.to_dict()}

    # ------------------------------------------------------------------ redaction

    def _redacted(self, entry: CassetteEntry) -> CassetteEntry:
        """*entry* with every stored text masked.

        Applied at the write boundary rather than at each call site, so there is one place to
        check. A cassette is written once and read for years, and it is the artifact most
        likely to end up committed.
        """
        def mask(text: str) -> str:
            return redact(text, env=self._redact_env) or ""

        result = entry.result
        return CassetteEntry(
            key=entry.key,
            prompt=mask(entry.prompt),
            provider_id=entry.provider_id,
            result=QueryResult(
                result=mask(result.result),
                returncode=result.returncode,
                stderr=mask(result.stderr),
                stream_result=mask(result.stream_result),
                success=result.success,
                usage=result.usage,
            ),
            model_version=entry.model_version,
            system_message=mask(entry.system_message),
            variant=entry.variant,
            original_wall_s=entry.original_wall_s,
        )
