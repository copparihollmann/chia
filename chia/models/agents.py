"""Provider-neutral agent and model configuration.

The command-line harnesses supported by Chia use different configuration
shapes, but the useful concepts are the same: a named agent, the model it
uses, the tools it may call, and the provider which serves that model.  The
small immutable records in this module are intentionally safe to pickle and
send through Ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union


_AGENT_ROLES = frozenset(("primary", "subagent", "background"))
_PROTOCOLS = frozenset(("anthropic-messages", "bedrock-converse", "openai-compatible"))


@dataclass(frozen=True)
class ModelRef:
    """An opaque model identifier qualified by its provider."""

    provider: str
    model: str

    def __post_init__(self) -> None:
        if not self.provider or "/" in self.provider:
            raise ValueError("provider must be a non-empty id without '/'")
        if not self.model:
            raise ValueError("model must be non-empty")

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"

    @classmethod
    def parse(cls, value: Union["ModelRef", str]) -> "ModelRef":
        if isinstance(value, cls):
            return value
        provider, separator, model = value.partition("/")
        if not separator:
            raise ValueError("model reference must be 'provider/model'")
        return cls(provider=provider, model=model)


@dataclass(frozen=True)
class AgentDefinition:
    """A named agent which can be rendered for Claude Code or OpenCode.

    ``tools`` contains harness tool identifiers.  An empty tuple means the
    agent receives no tools; ``None`` means inherit the harness defaults.
    """

    name: str
    description: str
    prompt: str
    role: str = "subagent"
    model: Optional[Union[ModelRef, str]] = None
    tools: Optional[Sequence[str]] = None
    effort: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise ValueError("agent name must be non-empty and contain no whitespace")
        if self.role not in _AGENT_ROLES:
            raise ValueError(f"agent role must be one of {sorted(_AGENT_ROLES)}")
        if not self.description:
            raise ValueError(f"agent {self.name!r} requires a description")
        if not self.prompt:
            raise ValueError(f"agent {self.name!r} requires a prompt")
        if self.tools is not None:
            object.__setattr__(self, "tools", tuple(self.tools))

    @property
    def model_id(self) -> Optional[str]:
        return str(self.model) if self.model is not None else None

    def to_claude(self) -> dict:
        """Render one entry in Claude Code's ``--agents`` JSON object."""
        result: Dict[str, Any] = {
            "description": self.description,
            "prompt": self.prompt,
        }
        if self.model_id is not None:
            result["model"] = self.model_id
        if self.tools is not None:
            result["tools"] = list(self.tools)
        if self.effort is not None:
            result["effort"] = self.effort
        return result

    def to_opencode(self) -> dict:
        """Render one entry in OpenCode's ``agent`` configuration object."""
        result: Dict[str, Any] = {
            "description": self.description,
            "prompt": self.prompt,
            "mode": "primary" if self.role == "primary" else "subagent",
        }
        if self.model_id is not None:
            result["model"] = self.model_id
        if self.tools is not None:
            # OpenCode uses a name -> boolean map, unlike Claude's list.
            result["tools"] = {name: True for name in self.tools}
        if self.effort is not None:
            result["effort"] = self.effort
        return result


@dataclass(frozen=True)
class ProviderSpec:
    """Connection information for a model provider.

    Secrets are referenced by environment-variable *name*.  They are never
    copied into this record, generated configuration, or log output.
    """

    id: str
    protocol: str
    models: Union[Sequence[str], Mapping[str, Mapping[str, Any]]]
    base_url: Optional[str] = None
    credential_env: Optional[str] = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or "/" in self.id:
            raise ValueError("provider id must be non-empty and contain no '/'")
        if self.protocol not in _PROTOCOLS:
            raise ValueError(f"protocol must be one of {sorted(_PROTOCOLS)}")
        models = self.models
        if isinstance(models, Mapping):
            object.__setattr__(self, "models", dict(models))
        else:
            object.__setattr__(self, "models", tuple(models))
        if not self.models:
            raise ValueError(f"provider {self.id!r} must declare at least one model")
        object.__setattr__(self, "options", dict(self.options))

    def has_model(self, model: str) -> bool:
        return model in self.models

    def to_opencode(self) -> dict:
        """Render an OpenCode custom-provider config without materializing secrets."""
        npm = {
            "openai-compatible": "@ai-sdk/openai-compatible",
            "anthropic-messages": "@ai-sdk/anthropic",
            "bedrock-converse": "@ai-sdk/amazon-bedrock",
        }[self.protocol]
        models = (dict(self.models) if isinstance(self.models, Mapping)
                  else {model: {} for model in self.models})
        options: Dict[str, Any] = {}
        if self.base_url is not None:
            options["baseURL"] = self.base_url
        if self.credential_env is not None:
            options["apiKey"] = f"{{env:{self.credential_env}}}"
        # Capability declarations belong to Chia's preflight validation, not
        # to the provider SDK's options object.
        options.update({key: value for key, value in self.options.items()
                        if key not in {"supports_tools"}})
        result: Dict[str, Any] = {"npm": npm, "name": self.id, "models": models}
        if options:
            result["options"] = options
        return result


def validate_agents(agents: Sequence[AgentDefinition], primary_agent: Optional[str]) -> None:
    """Reject ambiguous agent sets before a subprocess is launched."""
    names = [agent.name for agent in agents]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate agent name(s): {', '.join(duplicates)}")
    if primary_agent is not None and primary_agent not in names:
        raise ValueError(f"primary_agent {primary_agent!r} is not present in agents")
    if primary_agent is not None:
        selected = next(agent for agent in agents if agent.name == primary_agent)
        if selected.role != "primary":
            raise ValueError(f"primary_agent {primary_agent!r} must have role='primary'")
