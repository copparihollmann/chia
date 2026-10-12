"""Opt-in OpenCode CLI delegation test against a local OpenAI-compatible fixture.

No paid provider is contacted. XDG state and the agent workspace live under
pytest's temporary directory. Enable with ``CHIA_RUN_LOCAL_OPENCODE_FIXTURE=1``.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chia.models.agents import AgentDefinition, ModelRef, ProviderSpec
from chia.models.opencode import OpenCodeLLM


pytestmark = pytest.mark.skipif(
    os.environ.get("CHIA_RUN_LOCAL_OPENCODE_FIXTURE") != "1"
    or shutil.which("opencode") is None,
    reason="requires opt-in installed OpenCode CLI; endpoint remains local and free",
)


def test_real_opencode_cli_delegates_to_the_configured_weaker_model(tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            model = body.get("model", "")
            tools = [item.get("function", {}).get("name")
                     for item in body.get("tools") or []]
            roles = [message.get("role") for message in body.get("messages") or []]
            requests.append((model, tools, roles))

            if model == "weak":
                delta = {"role": "assistant", "content": "EVIDENCE: worker.py owns dispatch"}
                finish = "stop"
            elif "task" not in tools:  # session-title request
                delta = {"role": "assistant", "content": "Inspect dispatch"}
                finish = "stop"
            elif "tool" in roles:
                delta = {"role": "assistant", "content": "FINAL: delegated evidence received"}
                finish = "stop"
            else:
                delta = {"tool_calls": [{
                    "index": 0, "id": "call_delegate", "type": "function",
                    "function": {
                        "name": "task",
                        "arguments": json.dumps({
                            "description": "Inspect dispatch",
                            "prompt": "Find the file that owns worker dispatch and return one line.",
                            "subagent_type": "explorer",
                        }),
                    },
                }]}
                finish = "tool_calls"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"id": "chatcmpl-fixture", "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"id": "chatcmpl-fixture", "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": model,
                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                           "total_tokens": 12}},
            ):
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = ProviderSpec(
            "fixture", "openai-compatible",
            {name: {"limit": {"context": 8192, "output": 1024}}
             for name in ("strong", "weak")},
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            credential_env="FIXTURE_TOKEN",
        )
        agents = [
            AgentDefinition("lead", "Own the answer",
                            "Always delegate to explorer before answering.",
                            role="primary", model=ModelRef("fixture", "strong")),
            AgentDefinition("explorer", "Inspect files", "Return concise evidence.",
                            model=ModelRef("fixture", "weak"),
                            tools=("read", "glob", "grep")),
        ]
        llm = OpenCodeLLM(
            model="fixture/strong", agents=agents, primary_agent="lead",
            providers=[provider], work_dir=str(tmp_path), retries=1,
            timeout_seconds=25,
            extra_env={
                "FIXTURE_TOKEN": "fixture-not-a-secret",
                "XDG_DATA_HOME": str(tmp_path / "data"),
                "XDG_CONFIG_HOME": str(tmp_path / "config"),
                "XDG_CACHE_HOME": str(tmp_path / "cache"),
            },
        )
        result = llm._run_opencode(
            "Delegate one repository lookup, then report its evidence.", [])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert "delegated evidence received" in result.result
    main = [(model, roles) for model, tools, roles in requests if "task" in tools]
    assert main[0][0] == "strong"
    assert any(model == "weak" for model, _, _ in requests)
    assert main[-1][0] == "strong" and "tool" in main[-1][1]
    assert result.provider_usage["input_tokens"] == 30
    assert result.provider_usage["output_tokens"] == 6
    assert result.provider_usage["subagent_usage_complete"] is True
    assert result.provider_usage["usage_scope"] == "exported_sessions"
    agents_by_name = {item["agent"]: item for item in result.provider_usage["agent_usage"]}
    assert agents_by_name["lead"]["input_tokens"] == 20
    assert agents_by_name["explorer"]["input_tokens"] == 10
    assert agents_by_name["explorer"]["parent_session_id"] == result.session_id
