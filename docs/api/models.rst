Model Backends
==============

**NOTE:** We provide a single function to interface with each of these providers even though some are agents, some are LLM providers, and some are on-premises LLM servers. This interface is an agentic one, and for the non-agents, we turn the model into a (very primitive) agent, by placing it into a query->tool call->query loop. 

We plan to in the near future expose an interface to the non-agent models (providers and servers), which is, instead of a primitive agent, the interface you would use to build your own agents.

For most serious tasks that don't require on-premises LLM serving, we expect you will get better results using the agents (Claude Code, Codex, Copilot, Antigravity, or OpenCode), with your preferred provider for credentials for the agent, as opposed to using the specific node for your provider (e.g. Claude Code with Bedrock credentials instead of the bedrock node).

API reference for :mod:`chia.models`. These pages are generated from the docstrings in the source, so they stay in sync with the code.

Provider-neutral agents
-----------------------

``AgentDefinition`` is the shared description accepted by both
``ClaudeCodeLLM(agents=..., primary_agent=...)`` and
``OpenCodeLLM(agents=..., primary_agent=...)``. ``ModelRef`` keeps provider and
model identities explicit, while ``ProviderSpec`` describes transport without
copying a credential into configuration::

   from chia.models.agents import AgentDefinition, ModelRef, ProviderSpec

   providers = [ProviderSpec(
       id="local", protocol="openai-compatible", models=("strong", "weak"),
       base_url="http://127.0.0.1:8124/v1", credential_env="LOCAL_API_KEY",
   )]
   agents = [
       AgentDefinition("lead", "Own the answer", "Delegate repository research.",
                       role="primary", model=ModelRef("local", "strong")),
       AgentDefinition("explorer", "Read the tree", "Return evidence only.",
                       model=ModelRef("local", "weak"),
                       tools=("Read", "Glob", "Grep")),
   ]

For Claude Code, run the loopback Messages gateway and set
``ANTHROPIC_BASE_URL``, ``ANTHROPIC_AUTH_TOKEN`` and
``CLAUDE_CODE_USE_GATEWAY=1`` in the Claude subprocess environment. For
OpenCode, pass the same providers and agents directly. The older
``AdditionalModelProvider`` remains supported. Offline fixtures pin Claude Code
2.1.233 and OpenCode 1.18.10; Claude feature-detects the required flags before
launching a delegated run.

.. automodule:: chia.models.agents

Claude
------

.. automodule:: chia.models.claude

Bedrock
-------

.. automodule:: chia.models.bedrock

Vertex
------

.. automodule:: chia.models.vertex

Antigravity
-----------

.. automodule:: chia.models.antigravity

Codex
-----

.. automodule:: chia.models.codex

Copilot
-------

.. automodule:: chia.models.copilot

Opencode
--------

.. automodule:: chia.models.opencode

Openai Compat
-------------

.. automodule:: chia.models.openai_compat

Openai Providers
----------------

.. automodule:: chia.models.openai_providers

Ollama
------

.. automodule:: chia.models.ollama

vLLM
----

.. automodule:: chia.models.vllm

Bedrock Converse proxy
----------------------

A local Bedrock-shaped endpoint that lets the Claude Code CLI drive **any** Bedrock
model, not only the Anthropic ones. Requests for Anthropic models are forwarded
verbatim; everything else is translated to Converse and streamed back using AWS
event-stream framing.
Because ``ANTHROPIC_MODEL``, ``CLAUDE_CODE_SUBAGENT_MODEL`` and
``ANTHROPIC_SMALL_FAST_MODEL`` are separate levers, one proxy can route each tier to a
different provider.

.. automodule:: chia.models.proxy.server

.. automodule:: chia.models.proxy.translate

Provider-agnostic Messages gateway
----------------------------------

The gateway routes opaque ``provider/model`` identifiers to a backend protocol.
It includes Bedrock Converse and OpenAI-compatible adapters, binds loopback by
default, and references credentials by environment-variable name. It is suitable
for Claude Code's ``ANTHROPIC_BASE_URL`` transport and for deterministic local
test endpoints.

.. automodule:: chia.models.proxy.gateway
