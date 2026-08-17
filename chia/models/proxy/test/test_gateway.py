"""Offline tests for the provider-agnostic Messages gateway."""

from __future__ import annotations

import json

import pytest

from chia.models.agents import ProviderSpec
from chia.models.proxy.gateway import (
    OpenAICompatibleBackend,
    ProviderRouter,
    VERIFIED_CLAUDE_CODE_VERSION,
    build_gateway_app,
)

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def complete(self, model, body):
        self.calls.append(("complete", model, body))
        return {"id": "msg_fake", "type": "message", "role": "assistant",
                "model": model, "content": [{"type": "text", "text": "PONG"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 2,
                                                        "output_tokens": 1}}

    def stream(self, model, body):
        self.calls.append(("stream", model, body))
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def test_gateway_routes_qualified_models_and_streams():
    fake = _FakeBackend()
    spec = ProviderSpec("weak", "openai-compatible", ("coder",))
    client = TestClient(build_gateway_app(ProviderRouter({"weak": fake}, [spec])))

    health = client.get("/healthz")
    assert health.json()["verified_claude_code_version"] == VERIFIED_CLAUDE_CODE_VERSION
    response = client.post("/v1/messages", json={
        "model": "weak/coder", "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8, "stream": True,
    })
    assert response.status_code == 200
    assert "message_stop" in response.text
    assert fake.calls[0][0:2] == ("stream", "coder")


def test_gateway_rejects_unknown_or_undeclared_models_without_calling_backend():
    fake = _FakeBackend()
    spec = ProviderSpec("weak", "openai-compatible", ("coder",))
    client = TestClient(build_gateway_app(ProviderRouter({"weak": fake}, [spec])))
    response = client.post("/v1/messages", json={"model": "weak/other", "messages": []})
    assert response.status_code == 400
    assert "does not declare" in response.json()["error"]["message"]
    assert fake.calls == []


def test_openai_adapter_round_trips_tools_usage_and_credential_by_env(monkeypatch):
    seen = {}

    def post(url, payload, headers):
        seen.update(url=url, payload=payload, headers=headers)
        return type("Response", (), {"status": 200, "body": json.dumps({
            "id": "chatcmpl_1", "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }],
            }}], "usage": {"prompt_tokens": 20, "completion_tokens": 5,
                              "prompt_tokens_details": {"cached_tokens": 12}},
        }).encode()})()

    monkeypatch.setenv("LOCAL_TOKEN", "secret-value")
    backend = OpenAICompatibleBackend(
        "http://127.0.0.1:9000/v1", "LOCAL_TOKEN", http_post=post
    )
    result = backend.complete("coder", {
        "system": "Be precise.", "max_tokens": 64,
        "messages": [{"role": "user", "content": "inspect"}],
        "tools": [{"name": "read_file", "description": "read",
                   "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
    })
    assert seen["url"] == "http://127.0.0.1:9000/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer secret-value"
    assert "secret-value" not in json.dumps(seen["payload"])
    assert seen["payload"]["tools"][0]["function"]["name"] == "read_file"
    assert result["stop_reason"] == "tool_use"
    assert result["content"][0]["input"] == {"path": "README.md"}
    assert result["usage"] == {"input_tokens": 8, "output_tokens": 5,
                               "cache_read_input_tokens": 12}

    stream = b"".join(backend.stream("coder", {"messages": []})).decode()
    payloads = [json.loads(line.removeprefix("data: "))
                for line in stream.splitlines() if line.startswith("data: ")]
    start = next(payload for payload in payloads if payload["type"] == "message_start")
    delta = next(payload for payload in payloads if payload["type"] == "message_delta")
    assert start["message"]["usage"]["input_tokens"] == 0
    assert delta["usage"]["input_tokens"] == 8
    assert delta["usage"]["output_tokens"] == 5


def test_openai_cache_subset_is_clamped_to_total_prompt_tokens():
    def post(url, payload, headers):
        return type("Response", (), {"status": 200, "body": json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                      "prompt_tokens_details": {"cached_tokens": 9}},
        }).encode()})()

    result = OpenAICompatibleBackend("http://local/v1", http_post=post).complete(
        "coder", {"messages": []}
    )
    assert result["usage"]["input_tokens"] == 0
    assert result["usage"]["cache_read_input_tokens"] == 9
