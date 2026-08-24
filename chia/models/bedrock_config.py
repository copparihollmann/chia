"""Reusable, model-agnostic Bedrock multi-model configuration for Claude Code.

Claude Code has **no** automatic task-complexity routing: a single run always
drives one primary/orchestrator model, and "delegation down" happens only when
that model spawns **Task-tool subagents** (which run on a separate,
configurable model) or does background chores like titling/summarising (a third,
"small/fast" model). Amazon Bedrock is wired in purely through environment
variables. This module turns a tier description ("Opus orchestrates, delegates
to Sonnet, background chores on Haiku") into the exact env-var dict Claude Code
expects, so **any** project driving Claude Code through chia can reuse it for
**any** tier mix and **any** models.

The levers (all accept Bedrock inference-profile ids such as
``us.anthropic.claude-opus-4-6-v1`` or the short aliases ``opus`` / ``sonnet`` /
``haiku``):

- ``CLAUDE_CODE_USE_BEDROCK=1`` + ``AWS_REGION`` (+ ``AWS_DEFAULT_REGION``) +
  ``AWS_BEARER_TOKEN_BEDROCK`` — route Claude Code to Bedrock with bearer-token
  auth.
- ``ANTHROPIC_MODEL`` — the PRIMARY / orchestrator model (the "Opus" tier). This
  mirrors the CLI ``--model`` flag.
- ``CLAUDE_CODE_SUBAGENT_MODEL`` — the model **all** Task-tool subagents run on
  (the "Sonnet" tier). This is how an Opus orchestrator "delegates down".
- ``ANTHROPIC_SMALL_FAST_MODEL`` (and the newer ``ANTHROPIC_DEFAULT_HAIKU_MODEL``)
  — background chores such as titles/summaries (the "Haiku" tier).
- ``ANTHROPIC_DEFAULT_OPUS_MODEL`` / ``ANTHROPIC_DEFAULT_SONNET_MODEL`` /
  ``ANTHROPIC_DEFAULT_HAIKU_MODEL`` — pin each tier's Bedrock profile so an alias
  (``opus``/``sonnet``/``haiku``) resolves to a concrete, invocable profile.

The defaults below are the currently invocable set on the account in use; they
are just sensible DEFAULTS — every id is overridable, so this helper is not tied
to any particular model or generation.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, MutableMapping, Optional

# Sensible DEFAULT tier pins — the invocable inference profiles on the account
# in use. These are NOT hardcoded as the only option: every one is overridable
# via the ``opus`` / ``sonnet`` / ``haiku`` arguments. (Note: ``opus-4-8`` /
# ``sonnet-5`` / ``opus-5`` are listed on the account but NOT invocable, which
# is why they are not used as defaults here.)
DEFAULT_OPUS_MODEL = "us.anthropic.claude-opus-4-6-v1"
DEFAULT_SONNET_MODEL = "us.anthropic.claude-sonnet-4-6"
DEFAULT_HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

DEFAULT_REGION = "us-east-1"


# --------------------------------------------------------------------------- #
# Verified Bedrock model registry (provider-agnostic).
#
# Claude Code the CLI only speaks the Anthropic Messages API on the wire, so a model's
# `via` records how the CLI reaches it:
#   * `cli_native` (Anthropic models) -> driven directly by :func:`bedrock_model_env`
#     (``CLAUDE_CODE_USE_BEDROCK`` + ``ANTHROPIC_MODEL`` ...), giving the native
#     orchestrator / Task-subagent / background tiering.
#   * `proxy` (everything else) -> reached through :mod:`chia.models.proxy.server`,
#     which translates Anthropic Messages to Converse. Verified end to end against
#     claude-cli 2.1.222. Before that proxy existed these models were CLI-unreachable
#     and this registry said so; they are not any more, which is the whole point of
#     mixed-provider tiers.
#
# Either way chia's own ``BedrockLLM`` (Converse) can talk to all of them directly —
# that is the second harness in the study, not a fallback.
#
# ``supports_tools`` = whether the model accepts a Converse ``toolConfig`` and
# actually emits ``toolUse`` (i.e. can drive an AGENTIC, tool-using loop).
# Probed on the account in use (us-east-1, bearer-token auth) 2026-07-30.
# --------------------------------------------------------------------------- #
from dataclasses import dataclass


#: How the Claude Code CLI reaches a model.
#:
#: ``cli_native`` — the CLI's Bedrock transport speaks this model's schema directly
#: (Anthropic Messages), so no translation is involved.
#: ``proxy`` — the model speaks Converse, so the CLI can only reach it through
#: :mod:`chia.models.proxy.server`. Verified working end to end against claude-cli
#: 2.1.222; before that proxy existed these models were CLI-unreachable, which is what
#: this field records.
VIA_CHOICES = ("cli_native", "proxy")


@dataclass(frozen=True)
class BedrockModel:
    """A Bedrock model: its inference-profile id, provider, how the CLI reaches it,
    and whether it can drive an agentic (tool-using) loop via the Converse API.

    :param id: Bedrock model or inference-profile id.
    :param provider: Vendor name, for reporting.
    :param supports_tools: Whether Converse accepts a ``toolConfig`` for it *and* it
        emits ``toolUse``. ``False`` means it cannot drive an agentic loop at all —
        not merely that it is worse at it.
    :param via: One of :data:`VIA_CHOICES`.
    :param notes: Anything a caller needs to know before picking it.
    :type id: str
    :type provider: str
    :type supports_tools: bool
    :type via: str
    :type notes: str
    """

    id: str
    provider: str
    supports_tools: bool
    via: str = "proxy"
    notes: str = ""


MODELS: Dict[str, "BedrockModel"] = {
    # Anthropic (Claude Code CLI via bedrock_model_env, or chia BedrockLLM).
    "opus": BedrockModel(DEFAULT_OPUS_MODEL, "Anthropic", True, "cli_native"),
    "sonnet": BedrockModel(DEFAULT_SONNET_MODEL, "Anthropic", True, "cli_native"),
    "haiku": BedrockModel(DEFAULT_HAIKU_MODEL, "Anthropic", True, "cli_native"),
    # Non-Anthropic: Converse-only on the wire, so the Claude Code CLI reaches them
    # only through chia.models.proxy.server. chia's own BedrockLLM talks to them
    # directly either way.
    "glm5": BedrockModel("zai.glm-5", "Z.AI", True),
    "glm4.7": BedrockModel("zai.glm-4.7", "Z.AI", True),
    "nemotron": BedrockModel("nvidia.nemotron-super-3-120b", "NVIDIA", True),
    "kimi": BedrockModel("moonshotai.kimi-k2.5", "Moonshot AI", True),
    "deepseek": BedrockModel("deepseek.v3.2", "DeepSeek", True),
    "deepseek-r1": BedrockModel(
        "deepseek.r1-v1:0", "DeepSeek", False, "proxy",
        "reasoning model; Converse rejects toolConfig -> NO agentic tool use",
    ),
    "qwen-coder": BedrockModel("qwen.qwen3-coder-next", "Qwen", True),
    "nova-pro": BedrockModel("us.amazon.nova-pro-v1:0", "Amazon", True),
    "nova-lite": BedrockModel("us.amazon.nova-lite-v1:0", "Amazon", True),
}


class TierError(ValueError):
    """A requested tier mix cannot be driven as configured.

    Raised at configuration time rather than left to surface as a provider error
    mid-run. A grid that discovers its model choice was impossible on row 200 has
    already spent 199 rows' worth of budget finding out.
    """


def resolve_model(name: str) -> str:
    """Resolve a short alias (``"glm5"``) or a raw Bedrock id to the concrete
    inference-profile id. Unknown names pass through unchanged, so a caller may
    always hand a full id."""
    m = MODELS.get(name)
    return m.id if m is not None else name


def model_via(name: str) -> str:
    """How the Claude Code CLI reaches *name*: one of :data:`VIA_CHOICES`.

    :param name: A registry alias or a raw Bedrock id.
    :type name: str
    :rtype: str

    An id absent from the registry is classified by its vendor prefix, so a model this
    registry has never heard of still routes correctly rather than defaulting to the
    wrong transport.
    """
    known = MODELS.get(name)
    if known is not None:
        return known.via
    from chia.models.proxy.translate import is_anthropic_model

    return "cli_native" if is_anthropic_model(name) else "proxy"


def requires_proxy(name: str) -> bool:
    """Whether *name* is **known** to need the Converse translator.

    :param name: A registry alias or a raw Bedrock id.
    :type name: str
    :rtype: bool

    Deliberately narrower than ``model_via(name) == "proxy"``: only a model this
    registry lists as Converse-only counts. An unknown raw id is *not* refused, for the
    same reason :meth:`ModelTier.validate` lets one through — the registry cannot
    enumerate every Bedrock profile, and a newly-released Anthropic id that this table
    has not learned yet would otherwise be rejected for a reason that is not true.
    Refusal needs certainty; classification only needs a best guess.
    """
    known = MODELS.get(name)
    return known is not None and known.via == "proxy"


def tier_transports(tier: "ModelTier") -> Dict[str, str]:
    """``{tier_name: via}`` for each populated tier of *tier*.

    :param tier: The mix to classify.
    :type tier: ModelTier
    :rtype: Dict[str, str]
    """
    return {
        name: model_via(value)
        for name, value in (("primary", tier.primary),
                            ("subagent", tier.subagent),
                            ("background", tier.background))
        if value
    }


@dataclass(frozen=True)
class ModelTier:
    """A provider-agnostic orchestrator/delegate/background tier mix, by alias or
    id. For Anthropic tiers, feed the resolved ids to :func:`bedrock_model_env`
    (Claude Code CLI). For non-Anthropic tiers, hand each resolved id to chia's
    ``BedrockLLM`` — the CLI cannot drive them. ``background``/``subagent`` may be
    ``None`` to leave that tier on the primary."""

    primary: str
    subagent: Optional[str] = None
    background: Optional[str] = None

    def resolved(self) -> Dict[str, Optional[str]]:
        """The tier as concrete Bedrock ids (``None`` tiers stay ``None``)."""
        return {
            "primary": resolve_model(self.primary),
            "subagent": resolve_model(self.subagent) if self.subagent else None,
            "background": resolve_model(self.background) if self.background else None,
        }

    def needs_proxy(self) -> bool:
        """Whether any populated tier can only be reached through the proxy.

        :rtype: bool

        A *mixed* mix needs it too: the CLI has one ``ANTHROPIC_BEDROCK_BASE_URL``, so
        as soon as one tier is Converse-only every tier goes through the proxy — which
        is fine, because the proxy forwards Anthropic requests verbatim.
        """
        return "proxy" in tier_transports(self).values()

    def validate(self, *, require_tools: bool = True) -> None:
        """Raise :class:`TierError` if this mix cannot work as configured.

        :param require_tools: Refuse a model that cannot drive a tool-using loop.
            Leave ``True`` for anything agentic; set ``False`` only for a
            single-turn text task.
        :type require_tools: bool
        :raises TierError: The mix is impossible, with the reason and the offending
            tier named.

        Two failures are caught here rather than at call time, because both are
        knowable from the registry alone:

        * A tier whose model has ``supports_tools=False`` cannot run an agent loop.
          Assigned to ``subagent``, every delegated task fails; assigned to
          ``primary``, the whole run does.
        * A registry alias that resolves to nothing usable.
        """
        for name, value in (("primary", self.primary), ("subagent", self.subagent),
                            ("background", self.background)):
            if not value:
                continue
            known = MODELS.get(value)
            if known is None:
                # A raw id is allowed through — the registry cannot enumerate every
                # Bedrock model, and refusing an unlisted id would be worse than
                # letting the provider judge it.
                continue
            if require_tools and not known.supports_tools:
                raise TierError(
                    f"tier {name!r} is {value!r} ({known.provider}), which cannot "
                    f"drive a tool-using loop"
                    + (f": {known.notes}" if known.notes else "")
                    + ". Pass require_tools=False only if this task needs no tools."
                )


# The canonical Anthropic tier (Opus orchestrates -> Sonnet subagents -> Haiku
# background) and a sensible non-Anthropic default (GLM-5 orchestrates, delegates
# to Qwen-coder, background on Nova-lite) — both just presets; every id overrides.
ANTHROPIC_TIER = ModelTier(primary="opus", subagent="sonnet", background="haiku")
NON_ANTHROPIC_TIER = ModelTier(primary="glm5", subagent="qwen-coder", background="nova-lite")


def bedrock_model_env(
    *,
    primary: str,
    subagent: Optional[str] = None,
    background: Optional[str] = None,
    region: str = DEFAULT_REGION,
    bearer_token: Optional[str] = None,
    opus: Optional[str] = None,
    sonnet: Optional[str] = None,
    haiku: Optional[str] = None,
    base_env: Optional[Mapping[str, str]] = None,
    proxy_url: Optional[str] = None,
    require_tools: bool = True,
) -> Dict[str, str]:
    """Build the env-var dict that points Claude Code at a Bedrock tier mix.

    This is a **pure** function: it never mutates ``os.environ`` (use
    :func:`apply_to_env` for that). It returns a plain ``dict`` of the variables
    to set, so a caller can merge them into whatever environment it hands to a
    ``claude`` subprocess.

    The canonical "Opus orchestrates, delegates to Sonnet, background Haiku"
    setup is just one call::

        env = bedrock_model_env(
            primary="opus",      # orchestrator            -> ANTHROPIC_MODEL
            subagent="sonnet",   # Task-tool subagents     -> CLAUDE_CODE_SUBAGENT_MODEL
            background="haiku",  # titles/summaries        -> ANTHROPIC_SMALL_FAST_MODEL
        )

    But the helper is model-agnostic: pass any Bedrock profile ids or aliases for
    any tier, or omit a tier to leave that lever unset.

    Args:
        primary: The orchestrator model. Sets ``ANTHROPIC_MODEL`` (the CLI
            ``--model``). Required.
        subagent: Model for **all** Task-tool subagents. When given, sets
            ``CLAUDE_CODE_SUBAGENT_MODEL`` — this is how the orchestrator
            delegates work down a tier. Omit to leave subagents on the default.
        background: Model for background chores (titles/summaries). When given,
            sets both ``ANTHROPIC_SMALL_FAST_MODEL`` and the newer
            ``ANTHROPIC_DEFAULT_HAIKU_MODEL`` (the background chore tier is the
            Haiku tier), which also overrides the ``haiku`` pin below.
        region: AWS region. Always sets ``AWS_REGION`` and ``AWS_DEFAULT_REGION``.
        bearer_token: Bedrock bearer token. Only sets
            ``AWS_BEARER_TOKEN_BEDROCK`` when provided; otherwise auth is left to
            the ambient AWS environment (env var, shared profile, or IAM role).
        opus: Pin for ``ANTHROPIC_DEFAULT_OPUS_MODEL``. Defaults to
            :data:`DEFAULT_OPUS_MODEL`.
        sonnet: Pin for ``ANTHROPIC_DEFAULT_SONNET_MODEL``. Defaults to
            :data:`DEFAULT_SONNET_MODEL`.
        haiku: Pin for ``ANTHROPIC_DEFAULT_HAIKU_MODEL``. Defaults to
            :data:`DEFAULT_HAIKU_MODEL`. Overridden by ``background`` when that is
            given (both name the Haiku tier).
        base_env: Optional environment to start from. When provided, the returned
            dict is a **copy** of ``base_env`` with the Bedrock variables overlaid
            on top; the passed mapping is never mutated. Handy for building a
            subprocess environment in one call.

        proxy_url: Base URL of a running :mod:`chia.models.proxy.server`. Required
            whenever any tier is a Converse-only model, since the CLI cannot speak
            that schema itself. When given, also sets
            ``CLAUDE_CODE_SKIP_BEDROCK_AUTH`` (the proxy verifies no signature).

            It deliberately does **not** set
            ``CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD``. That flag was needed
            while the proxy replied in plain SSE, and setting it hid a real defect:
            the CLI could not parse the SSE stream and silently retried every turn
            non-streaming, so each turn reached Bedrock twice. The proxy now answers
            in AWS event-stream framing and the flag must stay unset.
        require_tools: Forwarded to :meth:`ModelTier.validate`. Leave ``True`` for
            anything agentic.

    Returns:
        A ``dict[str, str]`` of environment variables to set.

    Raises:
        TierError: The mix cannot work as configured — a tool-incapable model in an
            agentic tier, or a Converse-only model with no ``proxy_url``. Refused here
            rather than at call time: a grid that discovers its model choice was
            impossible on row 200 has already spent 199 rows finding out.
    """
    tier = ModelTier(primary=primary, subagent=subagent, background=background)
    tier.validate(require_tools=require_tools)

    proxied = [name for name, value in (("primary", primary), ("subagent", subagent),
                                        ("background", background))
               if value and requires_proxy(value)]
    if proxied and not proxy_url:
        raise TierError(
            f"tier(s) {', '.join(sorted(proxied))} need a Converse translator "
            f"(models: {', '.join(sorted(set(str(v) for v in (primary, subagent, background) if v)))}), "
            f"which the Claude Code CLI cannot speak directly. Start "
            f"chia.models.proxy.server and pass proxy_url=..., or choose Anthropic "
            f"models for every tier."
        )

    env: Dict[str, str] = dict(base_env) if base_env is not None else {}

    # Always: route to Bedrock + region.
    env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    env["AWS_REGION"] = region
    env["AWS_DEFAULT_REGION"] = region

    if proxy_url:
        # One base URL for every tier, which is exactly why a mixed mix routes all of
        # them through the proxy: it forwards Anthropic requests verbatim, so the
        # cli_native tiers are unaffected by passing through it.
        env["ANTHROPIC_BEDROCK_BASE_URL"] = proxy_url
        env["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] = "1"

    # Bearer-token auth only when explicitly supplied; otherwise leave auth to
    # the ambient AWS credential chain.
    if bearer_token is not None:
        env["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

    # Per-tier default pins so aliases resolve to concrete invocable profiles.
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = opus if opus is not None else DEFAULT_OPUS_MODEL
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = (
        sonnet if sonnet is not None else DEFAULT_SONNET_MODEL
    )
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = (
        haiku if haiku is not None else DEFAULT_HAIKU_MODEL
    )

    def _wire(name: str) -> str:
        """The value to put in an env var for tier model *name*.

        An Anthropic alias (``opus``/``sonnet``/``haiku``) is left as the alias, because
        the ``ANTHROPIC_DEFAULT_*_MODEL`` pins above are what resolve it — that is the
        CLI's own mechanism. A proxy-routed alias has no such pin, so it must be
        resolved here or the CLI would send the literal string ``"glm5"`` as a model id
        and Bedrock would reject it.
        """
        return resolve_model(name) if model_via(name) == "proxy" else name

    # Primary / orchestrator (the "Opus" role).
    env["ANTHROPIC_MODEL"] = _wire(primary)

    # Delegate tier: all Task-tool subagents (the "Sonnet" role).
    if subagent is not None:
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = _wire(subagent)

    # Background chores (the "Haiku" role). Sets the small/fast lever and, since
    # background chores are the Haiku tier, overrides the Haiku pin too.
    if background is not None:
        env["ANTHROPIC_SMALL_FAST_MODEL"] = _wire(background)
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = _wire(background)

    return env


def apply_to_env(
    env: Optional[MutableMapping[str, str]] = None,
    **kwargs,
) -> MutableMapping[str, str]:
    """Merge :func:`bedrock_model_env` output into a mutable environment.

    Convenience wrapper that DOES mutate: it computes the Bedrock env dict from
    ``**kwargs`` (same signature as :func:`bedrock_model_env`, minus
    ``base_env``) and writes each key into ``env`` in place.

    Args:
        env: The mapping to update. Defaults to ``os.environ`` so a caller can
            configure the current process with one call.
        **kwargs: Forwarded to :func:`bedrock_model_env` (``primary`` is
            required). Passing ``base_env`` here is redundant — ``env`` already
            plays that role — so it is ignored if present.

    Returns:
        The same ``env`` mapping, for chaining.
    """
    if env is None:
        env = os.environ
    kwargs.pop("base_env", None)
    for key, value in bedrock_model_env(**kwargs).items():
        env[key] = value
    return env
