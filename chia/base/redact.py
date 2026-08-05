"""Mask credential values before they enter anything durable.

chia runs agent CLIs as subprocesses and hands their ``stderr`` straight back to the
caller: :attr:`chia.base.llm_call.QueryResult.stderr` carries it, every typed CLI error
carries a 300-character ``raw_message`` slice of it, and the CLI backends log a 500-character
slice at ``WARNING``.  On the authentication-failure path that text is exactly where a
provider echoes the credential it rejected — so a run that fails to authenticate is a run
that can write a live bearer token into a results file, a log, or a Ray exception
travelling between nodes.

This is not hypothetical downstream: ``mvp-lhwir/spec`` redacts ``raw``, ``stdout`` *and*
``stderr`` before recording anything, because its ``results.jsonl`` is committed to git.
It had to solve the problem outside chia, one layer above the code that produced the text.

Two rules make this safe to apply everywhere:

**Redact before truncating.** A secret straddling a ``[:300]`` boundary survives truncation
as a fragment, and a fragment of a token is still a disclosure.  :func:`truncate` exists so
that ordering is spelled out at the call site rather than remembered.

**A minimum length, which the downstream version does not need.** spec reads a curated
``.env`` file; chia inherits whatever environment the caller had. Masking every value of
every ``*_KEY``-shaped variable without a length floor turns a two-character value into a
substring that matches half the output, and the redacted log becomes useless. The floor
(:data:`MIN_SECRET_LEN`) is the one place this must not be a literal port.

False positives are cheap here and false negatives are a disclosure, so the matching is
deliberately generous: a name that *looks* like a credential is treated as one. Values that
cannot be credentials — absolute paths, anything containing whitespace — are skipped, since
those are the ones whose masking would obscure a log without protecting anything.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MASK",
    "MIN_SECRET_LEN",
    "SECRET_ENV_NAMES",
    "SECRET_NAME_RE",
    "is_secret_name",
    "redact",
    "secret_values",
    "truncate",
]


#: Format of a masked value. Keeps the variable's *name* so a reader can tell which
#: credential was involved without learning it.
MASK = "<redacted:{name}>"

#: Values shorter than this are never masked. See the module docstring: without a floor,
#: a short value of a credential-shaped variable matches everywhere and destroys the log.
MIN_SECRET_LEN = 8

#: Credential-carrying variables chia itself knows about, by name. The pattern below
#: catches the rest; this list exists so the common cases are documented and so a name
#: that does *not* match the pattern (none today, but providers keep inventing them) has
#: somewhere to go.
SECRET_ENV_NAMES = frozenset({
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
})

#: Name shapes treated as credential-carrying. Generous on purpose — see the module
#: docstring on the asymmetry between a masked path and a leaked token.
SECRET_NAME_RE = re.compile(
    r"(^|_)(API_?KEY|SECRET|SECRET_?KEY|TOKEN|PASSWORD|PASSWD|CREDENTIALS?)(_|$)",
    re.IGNORECASE,
)


def is_secret_name(name: str) -> bool:
    """Whether *name* names a variable whose value should be masked.

    :param name: Environment-variable name.
    :type name: str
    :returns: ``True`` for a known credential name or a credential-shaped one.
    :rtype: bool
    """
    return name in SECRET_ENV_NAMES or bool(SECRET_NAME_RE.search(name))


def _maskable(value: str) -> bool:
    """Whether *value* is long and opaque enough to be worth masking.

    A credential is a single opaque run of characters. An absolute path or anything with
    whitespace is something else — masking it would obscure output without protecting a
    secret, which is the failure mode that makes a redacted log unreadable.
    """
    if len(value) < MIN_SECRET_LEN:
        return False
    if value.startswith("/") or value.startswith("~"):
        return False
    return not any(ch.isspace() for ch in value)


def secret_values(
    env: Optional[Mapping[str, str]] = None,
    extra: Iterable[str] = (),
) -> Tuple[Tuple[str, str], ...]:
    """The ``(name, value)`` pairs worth masking, longest value first.

    :param env: Environment to scan. Defaults to :data:`os.environ`.
    :param extra: Additional literal secrets not present in the environment — an
        ``api_key`` passed to a backend constructor, for instance. Reported under the
        name ``"secret"``.
    :type env: Optional[Mapping[str, str]]
    :type extra: Iterable[str]
    :returns: Pairs sorted by descending value length.
    :rtype: Tuple[Tuple[str, str], ...]

    The ordering is load-bearing. When one secret contains another (an access key id
    embedded in a longer composite token, say), masking the shorter one first would leave
    the remainder of the longer one in the text. Longest-first makes that impossible.
    """
    source = os.environ if env is None else env
    found: Dict[str, str] = {}
    for name, value in source.items():
        if value and is_secret_name(name) and _maskable(value):
            found[value] = name
    for value in extra:
        if value and _maskable(value):
            found.setdefault(value, "secret")
    return tuple(sorted(
        ((name, value) for value, name in found.items()),
        key=lambda pair: len(pair[1]),
        reverse=True,
    ))


def redact(
    text: Optional[str],
    *,
    values: Optional[Sequence[Tuple[str, str]]] = None,
    env: Optional[Mapping[str, str]] = None,
    extra: Iterable[str] = (),
) -> Optional[str]:
    """Replace every known secret in *text* with a named mask.

    :param text: Text to scrub. ``None`` and ``""`` pass through unchanged.
    :param values: Pre-computed ``(name, value)`` pairs from :func:`secret_values`. Pass
        this when scrubbing many lines from one subprocess — it avoids rescanning the
        environment per line.
    :param env: Environment to scan when *values* is not given.
    :param extra: Additional literal secrets, as for :func:`secret_values`.
    :type text: Optional[str]
    :type values: Optional[Sequence[Tuple[str, str]]]
    :type env: Optional[Mapping[str, str]]
    :type extra: Iterable[str]
    :returns: *text* with each secret replaced by ``<redacted:NAME>``.
    :rtype: Optional[str]

    Idempotent: the mask contains no secret, so re-redacting redacted text is a no-op.
    """
    if not text:
        return text
    pairs = secret_values(env, extra) if values is None else values
    for name, value in pairs:
        if value in text:
            text = text.replace(value, MASK.format(name=name))
    return text


def truncate(
    text: Optional[str],
    limit: int,
    *,
    values: Optional[Sequence[Tuple[str, str]]] = None,
    env: Optional[Mapping[str, str]] = None,
    extra: Iterable[str] = (),
) -> str:
    """Redact *text*, then cut it to *limit* characters — in that order.

    :param text: Text to scrub and shorten.
    :param limit: Maximum length of the result.
    :param values: As for :func:`redact`.
    :param env: As for :func:`redact`.
    :param extra: As for :func:`redact`.
    :type text: Optional[str]
    :type limit: int
    :type values: Optional[Sequence[Tuple[str, str]]]
    :type env: Optional[Mapping[str, str]]
    :type extra: Iterable[str]
    :returns: The redacted prefix, never longer than *limit*.
    :rtype: str

    Truncating first would slice a secret in half and emit the first half. This function
    exists so that ordering is enforced by the call rather than by the caller remembering
    it, since every ``raw_message=...[:300]`` site in the CLI backends had it backwards.
    """
    scrubbed = redact(text, values=values, env=env, extra=extra) or ""
    return scrubbed[:limit]
