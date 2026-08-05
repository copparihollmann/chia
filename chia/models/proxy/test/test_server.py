"""Tests for the proxy's HTTP surface.

No credentials, no network: a fake ``bedrock-runtime`` client is injected into
:func:`chia.models.proxy.server.build_app`, so every route is exercised end to end
through FastAPI's test client while boto3 is never constructed.

What is pinned here is the contract with the CLI, measured against ``claude-cli
2.1.222``: the routes it calls, the model id in the path, the discovery call it makes
first, and the ``text/event-stream`` content type that its
``CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD`` flag exists to accept.
"""

from __future__ import annotations

import json

import pytest

from chia.models.proxy.server import VERIFIED_CLI_VERSION, build_app

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="fastapi is a pinned chia dependency; install it"
)
TestClient = fastapi_testclient.TestClient

GLM = "zai.glm-5"
SONNET = "us.anthropic.claude-sonnet-4-6-v1:0"


class _FakeBedrock:
    """Records what it was asked for and replays canned Converse output."""

    def __init__(self, stream=None, converse_response=None, raise_on_call=None):
        self.calls = []
        self._stream = stream or []
        self._converse_response = converse_response or {
            "output": {"message": {"role": "assistant", "content": [{"text": "PONG"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }
        self._raise = raise_on_call

    def converse_stream(self, **kwargs):
        self.calls.append(("converse_stream", kwargs))
        if self._raise:
            raise self._raise
        return {"stream": iter(self._stream)}

    def converse(self, **kwargs):
        self.calls.append(("converse", kwargs))
        if self._raise:
            raise self._raise
        return self._converse_response

    def invoke_model_with_response_stream(self, **kwargs):
        self.calls.append(("invoke_model_with_response_stream", kwargs))
        frames = [
            {"type": "message_start", "message": {"id": "msg_1"}},
            {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": "PONG"}},
            {"type": "message_stop"},
        ]
        return {"body": [{"chunk": {"bytes": json.dumps(f).encode()}} for f in frames]}

    def invoke_model(self, **kwargs):
        self.calls.append(("invoke_model", kwargs))

        class _Body:
            @staticmethod
            def read():
                return json.dumps({"id": "msg_1", "content": [
                    {"type": "text", "text": "PONG"}]}).encode()

        return {"body": _Body()}


def _client(fake):
    return TestClient(build_app(client=fake))


def _sse_events(text):
    """Parse an SSE response body into ``[(event_type, payload), ...]``."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


# ---------------------------------------------------------------------------
# The routes the CLI was measured to call
# ---------------------------------------------------------------------------


def test_healthz_reports_the_verified_cli_version():
    """So a launcher can wait for readiness, and so the version this was verified
    against is discoverable at runtime rather than only in a docstring."""
    response = _client(_FakeBedrock()).get("/healthz")

    assert response.status_code == 200
    assert response.json()["verified_cli_version"] == VERIFIED_CLI_VERSION


def test_the_discovery_call_the_cli_makes_first_is_answered():
    """aws-sdk-js issues GET /inference-profiles?type=SYSTEM_DEFINED before the first
    invoke. An empty summary list is a valid answer and the CLI proceeds — verified
    against the recorded probe, not assumed. Inventing profiles would make it request
    ids that do not exist."""
    response = _client(_FakeBedrock()).get(
        "/inference-profiles", params={"type": "SYSTEM_DEFINED"})

    assert response.status_code == 200
    assert response.json() == {"inferenceProfileSummaries": []}


def test_the_model_id_comes_from_the_path():
    """The measured shape: POST /model/{modelId}/invoke-with-response-stream, with the
    id in the path and never in the body."""
    fake = _FakeBedrock(stream=[{"messageStop": {"stopReason": "end_turn"}}])

    _client(fake).post(f"/model/{GLM}/invoke-with-response-stream",
                       json={"messages": [{"role": "user", "content": "ping"}]})

    kind, kwargs = fake.calls[0]
    assert kind == "converse_stream"
    assert kwargs["modelId"] == GLM


def test_a_model_id_with_slashes_still_routes():
    """Bedrock ids can carry an ARN-ish shape; a non-greedy path parameter would 404
    mid-session on one."""
    fake = _FakeBedrock(stream=[{"messageStop": {"stopReason": "end_turn"}}])
    model_id = "arn:aws:bedrock:us-east-1:1234:inference-profile/zai.glm-5"

    response = _client(fake).post(
        f"/model/{model_id}/invoke-with-response-stream",
        json={"messages": [{"role": "user", "content": "ping"}]})

    assert response.status_code == 200
    assert fake.calls[0][1]["modelId"] == model_id


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_a_translated_stream_comes_back_as_anthropic_sse():
    fake = _FakeBedrock(stream=[
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "PONG"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 2,
                                "cacheReadInputTokens": 100}}},
    ])

    response = _client(fake).post(
        f"/model/{GLM}/invoke-with-response-stream",
        json={"messages": [{"role": "user", "content": "ping"}], "max_tokens": 64})

    assert response.status_code == 200
    # text/event-stream, which is exactly what the CLI's content-type guard rejects
    # unless CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD=1 is set. Asserted here so
    # the deviation is pinned by a test rather than only described in prose.
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _sse_events(response.text)
    types = [t for t, _ in events]
    assert types[0] == "message_start"
    assert types[-2:] == ["message_delta", "message_stop"]
    text = "".join(p["delta"]["text"] for t, p in events
                   if t == "content_block_delta")
    assert text == "PONG"
    delta = next(p for t, p in events if t == "message_delta")
    assert delta["usage"]["cache_read_input_tokens"] == 100


def test_an_anthropic_model_is_forwarded_untranslated():
    """Passthrough keeps the Anthropic arm of a harness comparison free of translation
    artefacts — the request body is handed to Bedrock exactly as the CLI wrote it."""
    fake = _FakeBedrock()
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 64,
            "messages": [{"role": "user", "content": "ping"}]}

    response = _client(fake).post(f"/model/{SONNET}/invoke-with-response-stream",
                                  json=body)

    assert response.status_code == 200
    kind, kwargs = fake.calls[0]
    assert kind == "invoke_model_with_response_stream"
    assert json.loads(kwargs["body"]) == body
    events = _sse_events(response.text)
    assert [t for t, _ in events][0] == "message_start"


def test_the_request_body_is_translated_not_forwarded_for_a_converse_model():
    fake = _FakeBedrock(stream=[{"messageStop": {"stopReason": "end_turn"}}])

    _client(fake).post(f"/model/{GLM}/invoke-with-response-stream", json={
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "ping"}],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    })

    _, kwargs = fake.calls[0]
    assert "anthropic_version" not in kwargs
    assert kwargs["inferenceConfig"]["maxTokens"] == 64
    assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "Bash"


# ---------------------------------------------------------------------------
# Non-streaming fallback
# ---------------------------------------------------------------------------


def test_the_non_streaming_path_is_implemented():
    """The probe caught the CLI falling back to /invoke, so leaving it to 404 would
    break a session mid-run."""
    fake = _FakeBedrock()

    response = _client(fake).post(f"/model/{GLM}/invoke", json={
        "messages": [{"role": "user", "content": "ping"}]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["role"] == "assistant"
    assert payload["content"][0] == {"type": "text", "text": "PONG"}
    assert fake.calls[0][0] == "converse"


def test_the_non_streaming_path_also_passes_anthropic_through():
    fake = _FakeBedrock()

    response = _client(fake).post(f"/model/{SONNET}/invoke", json={
        "messages": [{"role": "user", "content": "ping"}]})

    assert response.status_code == 200
    assert fake.calls[0][0] == "invoke_model"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_a_provider_failure_becomes_a_502_with_the_reason():
    """Not a 500: the failure is upstream, and the CLI's own error classification keys
    off the status. Swallowing the reason would make a throttle indistinguishable from a
    bad model id."""
    fake = _FakeBedrock(raise_on_call=RuntimeError("ThrottlingException: slow down"))

    response = _client(fake).post(
        f"/model/{GLM}/invoke-with-response-stream",
        json={"messages": [{"role": "user", "content": "ping"}]})

    assert response.status_code == 502
    assert "Throttling" in response.json()["message"]


def test_a_malformed_body_is_a_400():
    response = _client(_FakeBedrock()).post(
        f"/model/{GLM}/invoke-with-response-stream",
        content=b"not json", headers={"content-type": "application/json"})

    assert response.status_code == 400


def test_a_non_object_body_is_a_400():
    response = _client(_FakeBedrock()).post(
        f"/model/{GLM}/invoke-with-response-stream", json=["not", "an", "object"])

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Binding policy
# ---------------------------------------------------------------------------


def test_main_refuses_to_bind_a_non_loopback_host_without_the_opt_out():
    """The proxy authenticates nobody and holds live AWS credentials, so a
    network-reachable sidecar is a network-reachable way to spend them."""
    from chia.models.proxy.server import main

    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0", "--port", "8123"])
