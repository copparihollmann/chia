"""Offline contracts shared by the Claude Code and OpenCode harnesses."""

from __future__ import annotations

import json
import pickle

import pytest

from chia.models.agents import AgentDefinition, ModelRef, ProviderSpec
from chia.models.claude import ClaudeCodeLLM
from chia.models.opencode import AdditionalModelProvider, OpenCodeLLM


def _agents():
    return [
        AgentDefinition(
            name="orchestrator", description="Own the final answer", prompt="Delegate research.",
            role="primary", model=ModelRef("gateway", "strong"), tools=("Task",), effort="high",
        ),
        AgentDefinition(
            name="explorer", description="Read the repository", prompt="Return evidence only.",
            model=ModelRef("gateway", "weak"), tools=("Read", "Glob", "Grep"), effort="low",
        ),
    ]


def test_shared_records_are_pickleable_and_model_refs_are_opaque():
    restored = pickle.loads(pickle.dumps(_agents()))
    assert restored == _agents()
    assert str(restored[1].model) == "gateway/weak"
    assert ModelRef.parse("gateway/org/model") == ModelRef("gateway", "org/model")


def test_duplicate_and_invalid_primary_agents_fail_before_launch():
    agent = _agents()[1]
    with pytest.raises(ValueError, match="duplicate agent"):
        ClaudeCodeLLM(agents=[agent, agent])
    with pytest.raises(ValueError, match="role='primary'"):
        OpenCodeLLM(agents=[agent], primary_agent="explorer")


def test_claude_renders_deterministic_agents_and_primary_selection():
    llm = ClaudeCodeLLM(
        agents=list(reversed(_agents())), primary_agent="orchestrator",
        claude_bin="/opt/claude-2.1.233", log_stream=False,
    )
    cmd = llm._build_cmd([])
    assert cmd[0] == "/opt/claude-2.1.233"
    rendered = cmd[cmd.index("--agents") + 1]
    assert rendered == json.dumps(json.loads(rendered), sort_keys=True, separators=(",", ":"))
    assert list(json.loads(rendered)) == ["explorer", "orchestrator"]
    assert json.loads(rendered)["explorer"]["model"] == "gateway/weak"
    assert cmd[cmd.index("--agent") + 1] == "orchestrator"


def test_opencode_renders_the_same_agents_and_provider_without_secret(monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKEN", "do-not-write-me")
    provider = ProviderSpec(
        id="gateway", protocol="openai-compatible", models=("strong", "weak"),
        base_url="http://127.0.0.1:8124/v1", credential_env="GATEWAY_TOKEN",
    )
    llm = OpenCodeLLM(
        agents=_agents(), primary_agent="orchestrator", providers=[provider],
        extra_env={"EXPERIMENT_ARM": "delegation"},
    )
    cfg = llm._build_config([])
    assert cfg["agent"]["explorer"]["tools"] == {"Read": True, "Glob": True, "Grep": True}
    assert cfg["agent"]["orchestrator"]["mode"] == "primary"
    assert cfg["provider"]["gateway"]["options"]["apiKey"] == "{env:GATEWAY_TOKEN}"
    assert "do-not-write-me" not in json.dumps(cfg)
    assert llm._build_run_cmd("hello")[4:6] == ["--agent", "orchestrator"]


def test_legacy_additional_provider_adapts_to_shared_shape():
    legacy = AdditionalModelProvider(
        id="local", models=["tiny"], base_url="http://localhost:8000/v1",
        api_key="{env:LOCAL_TOKEN}",
    )
    spec = legacy.to_provider_spec()
    assert spec == ProviderSpec(
        id="local", protocol="openai-compatible", models=("tiny",),
        base_url="http://localhost:8000/v1", credential_env="LOCAL_TOKEN",
    )


def test_duplicate_provider_ids_are_rejected():
    provider = ProviderSpec("gateway", "openai-compatible", ("weak",))
    with pytest.raises(ValueError, match="duplicate provider"):
        OpenCodeLLM(providers=[provider, provider])._build_config([])


def test_agentic_role_rejects_provider_without_tool_support():
    provider = ProviderSpec(
        "gateway", "openai-compatible", ("weak",), options={"supports_tools": False}
    )
    with pytest.raises(ValueError, match="requires tools"):
        OpenCodeLLM(agents=[_agents()[1]], providers=[provider])
    assert "supports_tools" not in json.dumps(provider.to_opencode())


def test_claude_feature_detects_agent_flags(monkeypatch):
    from chia.models import claude as claude_module

    claude_module._CLI_AGENT_CAPABILITY.clear()
    monkeypatch.setattr(claude_module.subprocess, "run", lambda *args, **kwargs: type(
        "Probe", (), {"returncode": 0, "stdout": "  --agents <json>\n  --agent <name>\n", "stderr": ""}
    )())
    ClaudeCodeLLM(agents=[_agents()[1]])._ensure_cli_supports_agents()

    claude_module._CLI_AGENT_CAPABILITY.clear()
    monkeypatch.setattr(claude_module.subprocess, "run", lambda *args, **kwargs: type(
        "Probe", (), {"returncode": 0, "stdout": "old cli", "stderr": ""}
    )())
    with pytest.raises(RuntimeError, match="2.1.233"):
        ClaudeCodeLLM(agents=[_agents()[1]])._ensure_cli_supports_agents()

    # ``--agents`` must not accidentally satisfy the singular flag check.
    claude_module._CLI_AGENT_CAPABILITY.clear()
    monkeypatch.setattr(claude_module.subprocess, "run", lambda *args, **kwargs: type(
        "Probe", (), {"returncode": 0, "stdout": "  --agents <json>\n", "stderr": ""}
    )())
    with pytest.raises(RuntimeError, match="2.1.233"):
        ClaudeCodeLLM(agents=[_agents()[1]])._ensure_cli_supports_agents()
