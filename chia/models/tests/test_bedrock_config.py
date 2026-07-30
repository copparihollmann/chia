"""Unit tests for the reusable Bedrock multi-model config helper.

Covers :func:`chia.models.bedrock_config.bedrock_model_env` /
:func:`apply_to_env` and their integration into
:class:`chia.models.claude.ClaudeCodeLLM` (``use_bedrock=True``). All offline —
no network, no AWS, no ``claude`` subprocess.
"""

from __future__ import annotations

import os

from chia.models.bedrock_config import (
    DEFAULT_HAIKU_MODEL,
    DEFAULT_OPUS_MODEL,
    DEFAULT_REGION,
    DEFAULT_SONNET_MODEL,
    apply_to_env,
    bedrock_model_env,
)
from chia.models.claude import ClaudeCodeLLM


# ---------------------------------------------------------------------------
# bedrock_model_env
# ---------------------------------------------------------------------------


def test_full_orchestrator_delegate_background_config():
    """The canonical Opus/Sonnet/Haiku mix sets every expected lever."""
    env = bedrock_model_env(
        primary="opus",
        subagent="sonnet",
        background="haiku",
    )
    # Always-on routing.
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == DEFAULT_REGION
    assert env["AWS_DEFAULT_REGION"] == DEFAULT_REGION
    # Tiers.
    assert env["ANTHROPIC_MODEL"] == "opus"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sonnet"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "haiku"
    # background overrides the haiku pin (same tier).
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "haiku"
    # Default pins for the other tiers.
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == DEFAULT_OPUS_MODEL
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == DEFAULT_SONNET_MODEL
    # No bearer token unless asked.
    assert "AWS_BEARER_TOKEN_BEDROCK" not in env


def test_default_pins_are_invocable_profiles():
    """With no tier overrides, the default pins are the invocable profiles."""
    env = bedrock_model_env(primary="opus")
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "us.anthropic.claude-opus-4-6-v1"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "us.anthropic.claude-sonnet-4-6"
    assert (
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"]
        == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )


def test_omitting_subagent_and_background_omits_keys():
    env = bedrock_model_env(primary="opus")
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in env
    # But the Haiku pin still defaults (it is always set).
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == DEFAULT_HAIKU_MODEL
    # And ANTHROPIC_MODEL is always present.
    assert env["ANTHROPIC_MODEL"] == "opus"


def test_omitting_only_background_keeps_haiku_pin_default():
    env = bedrock_model_env(primary="opus", subagent="sonnet")
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sonnet"
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in env
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == DEFAULT_HAIKU_MODEL


def test_custom_model_ids_pass_through():
    """Arbitrary Bedrock profile ids are threaded verbatim — not tied to defaults."""
    env = bedrock_model_env(
        primary="us.anthropic.claude-opus-4-8-v1",
        subagent="us.anthropic.claude-sonnet-5-v1",
        background="us.anthropic.claude-haiku-9-v1",
        region="eu-west-1",
        opus="pin.opus.custom",
        sonnet="pin.sonnet.custom",
        haiku="pin.haiku.custom",
    )
    assert env["ANTHROPIC_MODEL"] == "us.anthropic.claude-opus-4-8-v1"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "us.anthropic.claude-sonnet-5-v1"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "us.anthropic.claude-haiku-9-v1"
    assert env["AWS_REGION"] == "eu-west-1"
    assert env["AWS_DEFAULT_REGION"] == "eu-west-1"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "pin.opus.custom"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "pin.sonnet.custom"
    # background overrides the explicit haiku pin.
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "us.anthropic.claude-haiku-9-v1"


def test_haiku_pin_used_when_no_background():
    env = bedrock_model_env(primary="opus", haiku="pin.haiku.only")
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "pin.haiku.only"
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in env


def test_bearer_token_set_only_when_given():
    without = bedrock_model_env(primary="opus")
    assert "AWS_BEARER_TOKEN_BEDROCK" not in without

    with_token = bedrock_model_env(primary="opus", bearer_token="secret-abc")
    assert with_token["AWS_BEARER_TOKEN_BEDROCK"] == "secret-abc"


def test_base_env_is_copied_not_mutated():
    base = {"PATH": "/usr/bin", "EXISTING": "keep"}
    env = bedrock_model_env(primary="opus", base_env=base)
    # Overlaid on top of a copy.
    assert env["PATH"] == "/usr/bin"
    assert env["EXISTING"] == "keep"
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    # Original untouched.
    assert "CLAUDE_CODE_USE_BEDROCK" not in base
    assert base == {"PATH": "/usr/bin", "EXISTING": "keep"}


def test_does_not_mutate_os_environ():
    before = dict(os.environ)
    bedrock_model_env(primary="opus", subagent="sonnet", background="haiku")
    assert dict(os.environ) == before


# ---------------------------------------------------------------------------
# apply_to_env
# ---------------------------------------------------------------------------


def test_apply_to_env_merges_into_given_mapping():
    target = {"PATH": "/bin"}
    returned = apply_to_env(target, primary="opus", subagent="sonnet")
    assert returned is target
    assert target["PATH"] == "/bin"
    assert target["ANTHROPIC_MODEL"] == "opus"
    assert target["CLAUDE_CODE_SUBAGENT_MODEL"] == "sonnet"
    assert target["CLAUDE_CODE_USE_BEDROCK"] == "1"


def test_apply_to_env_ignores_redundant_base_env():
    target = {}
    apply_to_env(target, primary="opus", base_env={"SHOULD": "ignore"})
    assert "SHOULD" not in target
    assert target["ANTHROPIC_MODEL"] == "opus"


# ---------------------------------------------------------------------------
# ClaudeCodeLLM integration
# ---------------------------------------------------------------------------


def test_llm_without_bedrock_leaves_env_untouched(monkeypatch):
    """Default behavior: the subprocess env is os.environ minus CLAUDECODE."""
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    monkeypatch.setenv("CLAUDECODE", "1")
    llm = ClaudeCodeLLM(model="claude-sonnet-4-6")
    env = llm._subprocess_env()
    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert "ANTHROPIC_MODEL" not in env
    # Exactly the passthrough env.
    expected = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    assert env == expected


def test_llm_with_bedrock_threads_vars_into_subprocess_env(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("SOME_HOST_VAR", "keep-me")
    llm = ClaudeCodeLLM(
        model="opus",
        use_bedrock=True,
        subagent_model="sonnet",
        background_model="haiku",
        region="us-west-2",
        bearer_token="tok-123",
    )
    env = llm._subprocess_env()
    # Bedrock levers present.
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert env["AWS_REGION"] == "us-west-2"
    assert env["AWS_DEFAULT_REGION"] == "us-west-2"
    assert env["AWS_BEARER_TOKEN_BEDROCK"] == "tok-123"
    # Primary is self.model (kept consistent with --model).
    assert env["ANTHROPIC_MODEL"] == "opus"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sonnet"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "haiku"
    # Host env carried through, CLAUDECODE still stripped.
    assert env["SOME_HOST_VAR"] == "keep-me"
    assert "CLAUDECODE" not in env


def test_llm_bedrock_tier_overrides_thread_through():
    llm = ClaudeCodeLLM(
        model="my.primary",
        use_bedrock=True,
        opus_model="my.opus",
        sonnet_model="my.sonnet",
        haiku_model="my.haiku",
    )
    env = llm._subprocess_env()
    assert env["ANTHROPIC_MODEL"] == "my.primary"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "my.opus"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "my.sonnet"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "my.haiku"
    # No subagent/background provided -> those keys omitted.
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env
    assert "ANTHROPIC_SMALL_FAST_MODEL" not in env


def test_llm_bedrock_defaults_region_when_unset():
    llm = ClaudeCodeLLM(model="opus", use_bedrock=True)
    env = llm._subprocess_env()
    assert env["AWS_REGION"] == DEFAULT_REGION
    assert "AWS_BEARER_TOKEN_BEDROCK" not in env
