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
# Two integration paths, because Claude Code the CLI only speaks the Anthropic
# Messages API:
#   * Anthropic models  -> drive Claude Code via :func:`bedrock_model_env`
#     (``CLAUDE_CODE_USE_BEDROCK`` + ``ANTHROPIC_MODEL`` ...), which gives the
#     native orchestrator / Task-subagent / background tiering.
#   * Non-Anthropic models -> the CLI CANNOT drive them; run them through chia's
#     own ``BedrockLLM`` (the Converse API), which normalises tool use across
#     providers. The orchestrator/delegate/background pattern is then expressed
#     by picking a model per tier from this registry (see :class:`ModelTier`).
#
# ``supports_tools`` = whether the model accepts a Converse ``toolConfig`` and
# actually emits ``toolUse`` (i.e. can drive an AGENTIC, tool-using loop).
# Probed on the account in use (us-east-1, bearer-token auth) 2026-07-30.
# --------------------------------------------------------------------------- #
from dataclasses import dataclass


@dataclass(frozen=True)
class BedrockModel:
    """A Bedrock model: its inference-profile id, provider, and whether it can
    drive an agentic (tool-using) loop via the Converse API."""

    id: str
    provider: str
    supports_tools: bool
    notes: str = ""


MODELS: Dict[str, "BedrockModel"] = {
    # Anthropic (Claude Code CLI via bedrock_model_env, or chia BedrockLLM).
    "opus": BedrockModel(DEFAULT_OPUS_MODEL, "Anthropic", True),
    "sonnet": BedrockModel(DEFAULT_SONNET_MODEL, "Anthropic", True),
    "haiku": BedrockModel(DEFAULT_HAIKU_MODEL, "Anthropic", True),
    # Non-Anthropic (chia BedrockLLM / Converse ONLY — not the Claude Code CLI).
    "glm5": BedrockModel("zai.glm-5", "Z.AI", True),
    "glm4.7": BedrockModel("zai.glm-4.7", "Z.AI", True),
    "nemotron": BedrockModel("nvidia.nemotron-super-3-120b", "NVIDIA", True),
    "kimi": BedrockModel("moonshotai.kimi-k2.5", "Moonshot AI", True),
    "deepseek": BedrockModel("deepseek.v3.2", "DeepSeek", True),
    "deepseek-r1": BedrockModel(
        "deepseek.r1-v1:0", "DeepSeek", False,
        "reasoning model; Converse rejects toolConfig -> NO agentic tool use",
    ),
    "qwen-coder": BedrockModel("qwen.qwen3-coder-next", "Qwen", True),
    "nova-pro": BedrockModel("us.amazon.nova-pro-v1:0", "Amazon", True),
    "nova-lite": BedrockModel("us.amazon.nova-lite-v1:0", "Amazon", True),
}


def resolve_model(name: str) -> str:
    """Resolve a short alias (``"glm5"``) or a raw Bedrock id to the concrete
    inference-profile id. Unknown names pass through unchanged, so a caller may
    always hand a full id."""
    m = MODELS.get(name)
    return m.id if m is not None else name


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

    Returns:
        A ``dict[str, str]`` of environment variables to set.
    """
    env: Dict[str, str] = dict(base_env) if base_env is not None else {}

    # Always: route to Bedrock + region.
    env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    env["AWS_REGION"] = region
    env["AWS_DEFAULT_REGION"] = region

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

    # Primary / orchestrator (the "Opus" role).
    env["ANTHROPIC_MODEL"] = primary

    # Delegate tier: all Task-tool subagents (the "Sonnet" role).
    if subagent is not None:
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = subagent

    # Background chores (the "Haiku" role). Sets the small/fast lever and, since
    # background chores are the Haiku tier, overrides the Haiku pin too.
    if background is not None:
        env["ANTHROPIC_SMALL_FAST_MODEL"] = background
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = background

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
