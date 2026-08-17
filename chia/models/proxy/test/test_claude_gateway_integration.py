"""Opt-in real Claude Code CLI test against a deterministic local gateway.

This test never contacts Anthropic or another paid provider. It launches the
installed CLI with a fake API key and a loopback ``ANTHROPIC_BASE_URL``. Enable
it explicitly with ``CHIA_RUN_LOCAL_CLAUDE_FIXTURE=1``.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from urllib import request as urllib_request

import pytest

from chia.models.agents import AgentDefinition, ModelRef
from chia.models.claude import ClaudeCodeLLM
from chia.models.proxy.gateway import ProviderRouter, build_gateway_app


pytestmark = pytest.mark.skipif(
    os.environ.get("CHIA_RUN_LOCAL_CLAUDE_FIXTURE") != "1" or shutil.which("claude") is None,
    reason="requires opt-in installed Claude CLI; endpoint remains local and free",
)


class _DelegatingFixture:
    def __init__(self):
        self.requests = []

    def complete(self, model, body):
        self.requests.append((model, body))
        if model == "weak":
            return self._text(model, "EVIDENCE: src/worker.py owns dispatch")
        tool_results = [block for message in body.get("messages", [])
                        for block in (message.get("content", [])
                                      if isinstance(message.get("content"), list) else [])
                        if isinstance(block, dict) and block.get("type") == "tool_result"]
        if tool_results:
            return self._text(model, "FINAL: delegated evidence received")
        tools = body.get("tools") or []
        delegate = next((tool for tool in tools
                         if tool.get("name") in ("Agent", "Task")), None)
        if delegate is None:
            raise RuntimeError("Claude did not advertise its delegation tool")
        schema = delegate.get("input_schema") or {}
        inputs = {}
        for key in schema.get("required") or []:
            if key == "description":
                inputs[key] = "Inspect dispatch"
            elif key in ("prompt", "task"):
                inputs[key] = "Find the file that owns worker dispatch and return one line."
            elif key in ("subagent_type", "agent", "agent_name", "name"):
                inputs[key] = "explorer"
            else:
                prop_type = (schema.get("properties", {}).get(key) or {}).get("type")
                inputs[key] = [] if prop_type == "array" else False if prop_type == "boolean" else "explorer"
        if "run_in_background" in (schema.get("properties") or {}):
            inputs["run_in_background"] = False
        if "subagent_type" in (schema.get("properties") or {}):
            inputs["subagent_type"] = "explorer"
        return {
            "id": "msg_delegate", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "tool_use", "id": "toolu_delegate",
                         "name": delegate["name"], "input": inputs}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def stream(self, model, body):
        from chia.models.proxy.gateway import _sse_from_message
        return _sse_from_message(self.complete(model, body))

    @staticmethod
    def _text(model, text):
        return {"id": f"msg_{model}", "type": "message", "role": "assistant",
                "model": model, "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}}


def test_real_cli_delegates_to_the_configured_weaker_model(tmp_path):
    import uvicorn

    fixture = _DelegatingFixture()
    app = build_gateway_app(ProviderRouter({"fixture": fixture}, default_provider="fixture"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            try:
                urllib_request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.2).read()
                break
            except Exception:
                time.sleep(0.02)
        agents = [
            AgentDefinition("lead", "Own the answer", "Always delegate to explorer first.",
                            role="primary", model="sonnet"),
            AgentDefinition("explorer", "Inspect files", "Return concise evidence.",
                            model=ModelRef("fixture", "weak"), tools=("Read", "Glob", "Grep")),
        ]
        llm = ClaudeCodeLLM(
            model="sonnet", agents=agents, primary_agent="lead",
            log_stream=False, timeout_seconds=15,
            extra_cli_args=["--debug", "api", "--debug-file", str(tmp_path / "claude.log")],
            extra_env={
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                "ANTHROPIC_AUTH_TOKEN": "fixture-not-a-secret",
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_USE_GATEWAY": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
        )
        try:
            result = llm._run_claude("Delegate one repository lookup, then report its evidence.", [])
        except subprocess.TimeoutExpired as exc:
            summary = [(model, len(body.get("messages") or []))
                       for model, body in fixture.requests]
            debug = (tmp_path / "claude.log").read_text()[-4000:]
            raise AssertionError(
                f"Claude CLI timed out; gateway requests={summary!r}; debug={debug}"
            ) from exc
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()

    assert result.returncode == 0, result.stderr
    assert "delegated evidence received" in result.result
    models = [model for model, _ in fixture.requests]
    assert "sonnet" in models[0]
    assert "weak" in models
    assert "sonnet" in models[-1]
