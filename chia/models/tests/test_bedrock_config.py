"""Unit tests for the reusable Bedrock multi-model config helper.

Covers :func:`chia.models.bedrock_config.bedrock_model_env` /
:func:`apply_to_env` and their integration into
:class:`chia.models.claude.ClaudeCodeLLM` (``use_bedrock=True``). All offline —
no network, no AWS, no ``claude`` subprocess.
"""

from __future__ import annotations

import os

import pytest

from chia.models import bedrock_config as bc
from chia.models.bedrock_config import (
    DEFAULT_HAIKU_MODEL,
    DEFAULT_OPUS_MODEL,
    DEFAULT_REGION,
    DEFAULT_SONNET_MODEL,
    MODELS,
    ModelTier,
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


# ---------------------------------------------------------------------------
# Mixed-provider tiers
#
# `via` records how the Claude Code CLI reaches a model. Before
# chia.models.proxy.server existed, non-Anthropic Bedrock models were
# CLI-unreachable and this registry's comments said so. They are reachable now, and
# these tests pin the two things that makes newly possible — and the two impossible
# combinations that must be refused before a grid spends money discovering them.
# ---------------------------------------------------------------------------


def test_anthropic_models_are_cli_native():
    for alias in ("opus", "sonnet", "haiku"):
        assert MODELS[alias].via == "cli_native"
        assert bc.model_via(alias) == "cli_native"


def test_non_anthropic_models_route_through_the_proxy():
    for alias in ("glm5", "nova-pro", "qwen-coder", "kimi", "deepseek"):
        assert MODELS[alias].via == "proxy"
        assert bc.model_via(alias) == "proxy"


def test_every_registry_entry_declares_a_valid_transport():
    for alias, model in MODELS.items():
        assert model.via in bc.VIA_CHOICES, alias


def test_an_unlisted_id_is_classified_by_its_vendor_prefix():
    """A model the registry has never heard of still routes to the right transport
    rather than defaulting to the wrong one."""
    assert bc.model_via("us.anthropic.claude-opus-5-v1:0") == "cli_native"
    assert bc.model_via("some.new-vendor-model") == "proxy"


def test_an_unlisted_id_is_classified_but_never_refused():
    """Refusal needs certainty, classification only a best guess: a newly-released
    Anthropic profile this table has not learned yet must not be rejected for a reason
    that is not true."""
    assert bc.model_via("some.new-vendor-model") == "proxy"
    assert bc.requires_proxy("some.new-vendor-model") is False
    assert bc.requires_proxy("glm5") is True


def test_a_mixed_tier_reports_its_transports():
    tier = ModelTier(primary="glm5", subagent="sonnet", background="nova-lite")

    assert bc.tier_transports(tier) == {
        "primary": "proxy", "subagent": "cli_native", "background": "proxy",
    }
    assert tier.needs_proxy() is True


def test_an_all_anthropic_tier_needs_no_proxy():
    assert ModelTier(primary="opus", subagent="sonnet",
                     background="haiku").needs_proxy() is False


def test_a_converse_only_tier_is_refused_without_a_proxy_url():
    """The plan's requirement: refuse an impossible combination up front instead of
    failing at call time. A grid that finds out on row 200 has already spent 199 rows."""
    with pytest.raises(bc.TierError) as exc:
        bedrock_model_env(primary="glm5", subagent="nova-pro")

    message = str(exc.value)
    assert "primary" in message and "subagent" in message
    assert "proxy_url" in message


def test_a_proxy_url_sets_the_two_flags_the_proxy_needs():
    """Without both of these the CLI either signs a request nobody verifies or rejects
    the proxy's SSE reply by content type. Setting them here means a caller cannot get
    a working tier mix and a broken transport at the same time."""
    env = bedrock_model_env(primary="glm5", background="nova-lite",
                            proxy_url="http://127.0.0.1:8123")

    assert env["ANTHROPIC_BEDROCK_BASE_URL"] == "http://127.0.0.1:8123"
    assert env["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] == "1"
    assert env["CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD"] == "1"


def test_a_proxy_routed_alias_is_resolved_to_a_concrete_id():
    """opus/sonnet/haiku are resolved by the CLI's own ANTHROPIC_DEFAULT_*_MODEL pins;
    a proxy-routed alias has no such pin, so leaving it unresolved would send Bedrock
    the literal string "glm5"."""
    env = bedrock_model_env(primary="glm5", subagent="nova-pro",
                            background="nova-lite",
                            proxy_url="http://127.0.0.1:8123")

    assert env["ANTHROPIC_MODEL"] == "zai.glm-5"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "us.amazon.nova-pro-v1:0"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "us.amazon.nova-lite-v1:0"


def test_anthropic_aliases_stay_aliases():
    """The CLI resolves them through the pins; rewriting them here would bypass the
    mechanism the pins exist for."""
    env = bedrock_model_env(primary="opus", subagent="sonnet", background="haiku")

    assert env["ANTHROPIC_MODEL"] == "opus"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sonnet"


def test_a_mixed_tier_routes_every_tier_through_the_proxy():
    """The CLI has one ANTHROPIC_BEDROCK_BASE_URL, so one Converse-only tier sends all
    of them through — which is harmless, because the proxy forwards Anthropic requests
    verbatim."""
    env = bedrock_model_env(primary="glm5", subagent="sonnet",
                            proxy_url="http://127.0.0.1:8123")

    assert env["ANTHROPIC_BEDROCK_BASE_URL"] == "http://127.0.0.1:8123"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "sonnet"


def test_a_tool_incapable_model_is_refused_in_an_agentic_tier():
    """deepseek-r1's Converse endpoint rejects toolConfig outright, so as a subagent
    every delegated task fails and as primary the whole run does."""
    with pytest.raises(bc.TierError) as exc:
        bedrock_model_env(primary="deepseek-r1", proxy_url="http://127.0.0.1:8123")

    assert "tool-using loop" in str(exc.value)
    assert "deepseek-r1" in str(exc.value)


def test_a_tool_incapable_model_is_allowed_when_tools_are_not_required():
    """A single-turn text task genuinely does not need them, so the check is opt-out
    rather than absolute."""
    env = bedrock_model_env(primary="deepseek-r1", require_tools=False,
                            proxy_url="http://127.0.0.1:8123")

    assert env["ANTHROPIC_MODEL"] == "deepseek.r1-v1:0"


def test_validate_names_the_offending_tier():
    with pytest.raises(bc.TierError) as exc:
        ModelTier(primary="glm5", subagent="deepseek-r1").validate()

    assert "'subagent'" in str(exc.value)


def test_the_non_anthropic_preset_is_now_actually_drivable():
    """NON_ANTHROPIC_TIER used to be documentation: the CLI could not drive any of it.
    With the proxy it is a working configuration, which is what makes ModelTier more
    than a comment."""
    env = bedrock_model_env(
        primary=bc.NON_ANTHROPIC_TIER.primary,
        subagent=bc.NON_ANTHROPIC_TIER.subagent,
        background=bc.NON_ANTHROPIC_TIER.background,
        proxy_url="http://127.0.0.1:8123",
    )

    assert env["ANTHROPIC_MODEL"] == "zai.glm-5"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "qwen.qwen3-coder-next"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "us.amazon.nova-lite-v1:0"
