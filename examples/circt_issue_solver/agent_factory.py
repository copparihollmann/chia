"""Agent construction seam for CIRCT experiments.

Set ``CFG["agent_factory"]`` to ``"module:function"`` to compare a different
harness without editing the worker pipeline. The function receives keyword
arguments ``cfg``, ``phase`` and ``default_kwargs`` and must return an object
with Chia's ``prompt`` interface.
"""

from __future__ import annotations

import importlib


def make_agent(cfg: dict, phase: str):
    from chia.models.claude import ClaudeCodeLLM

    kwargs = {
        "model": cfg["model"],
        "system_message": cfg["system_prompt"],
        "timeout_seconds": cfg["timeouts"][phase],
        "extra_cli_args": ["--effort", "max"],
        "resume_session": True,
        "projects_cwd": None,
    }
    kwargs.update(cfg.get("agent_kwargs") or {})
    factory_ref = cfg.get("agent_factory")
    if not factory_ref:
        return ClaudeCodeLLM(**kwargs)
    if not isinstance(factory_ref, str) or ":" not in factory_ref:
        raise ValueError("agent_factory must be an importable 'module:function' string")
    module_name, function_name = factory_ref.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory(cfg=cfg, phase=phase, default_kwargs=kwargs)
