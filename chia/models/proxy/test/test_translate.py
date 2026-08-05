"""Tests for the Anthropic ⇄ Converse translation.

Offline and dependency-light: the pure translation tests need nothing, and the server
tests use FastAPI's test client with an injected fake ``bedrock-runtime`` — so no AWS
credentials, no network, and no boto3 call is ever made.

The two fidelity questions the whole feature turns on get their own sections:

* **Does a tool call survive the round trip?** An agent harness is tool calls; a
  translation that loses one turns a working agent into a chatbot.
* **Does usage survive with its cache buckets intact?** Collapsing them here would
  silently reintroduce the mispricing ``chia.base.usage`` exists to prevent, on every
  call routed through the proxy.

``test_the_recorded_cli_request_still_translates`` replays a real captured request from
``claude-cli 2.1.222``. A CLI change that alters the schema fails that test rather than
failing a grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chia.models.proxy.translate import (
    is_anthropic_model,
    to_anthropic_message,
    to_anthropic_sse,
    to_converse,
    usage_to_anthropic,
)

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED_REQUEST = FIXTURES / "cli_2_1_222_invoke_stream.json"

GLM = "zai.glm-5"
NOVA = "us.amazon.nova-pro-v1:0"


def _frames(events, model=GLM):
    """Parse the SSE byte stream into ``[(event_type, payload), ...]``."""
    raw = b"".join(to_anthropic_sse(events, model=model)).decode()
    out = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", [
    "anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-6-v1:0",
    "claude-sonnet-4-6",
])
def test_anthropic_models_are_recognised_for_passthrough(model_id):
    """Passthrough is not just an optimisation: it keeps the Anthropic arm of a harness
    comparison free of any translation artefact."""
    assert is_anthropic_model(model_id) is True


@pytest.mark.parametrize("model_id", [
    GLM, NOVA, "qwen.qwen3-235b", "meta.llama3-1-70b-instruct-v1:0",
    "deepseek.r1-v1:0",
])
def test_non_anthropic_models_are_translated(model_id):
    assert is_anthropic_model(model_id) is False


# ---------------------------------------------------------------------------
# Anthropic -> Converse
# ---------------------------------------------------------------------------


def test_a_plain_turn_translates():
    kwargs = to_converse({"messages": [{"role": "user", "content": "ping"}],
                          "max_tokens": 1234}, GLM)

    assert kwargs["modelId"] == GLM
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "ping"}]}]
    assert kwargs["inferenceConfig"]["maxTokens"] == 1234


def test_content_block_lists_translate():
    kwargs = to_converse({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"},
                                     {"type": "text", "text": "b"}]},
    ]}, GLM)

    assert kwargs["messages"][0]["content"] == [{"text": "a"}, {"text": "b"}]


def test_system_blocks_become_converse_system():
    kwargs = to_converse({"messages": [], "system": [
        {"type": "text", "text": "be terse"},
    ]}, GLM)

    assert kwargs["system"] == [{"text": "be terse"}]


def test_a_cache_breakpoint_becomes_a_cache_point_where_supported():
    kwargs = to_converse({"messages": [], "system": [
        {"type": "text", "text": "big prompt", "cache_control": {"type": "ephemeral"}},
    ]}, NOVA)

    assert kwargs["system"] == [{"text": "big prompt"},
                               {"cachePoint": {"type": "default"}}]


def test_a_cache_breakpoint_is_dropped_where_unsupported():
    """Sending a cachePoint to a family without prompt caching is a request error;
    dropping it means the call succeeds and the usage honestly reports no cache
    tokens, because there were none."""
    kwargs = to_converse({"messages": [], "system": [
        {"type": "text", "text": "big prompt", "cache_control": {"type": "ephemeral"}},
    ]}, GLM)

    assert kwargs["system"] == [{"text": "big prompt"}]


def test_inference_parameters_translate():
    kwargs = to_converse({"messages": [], "max_tokens": 100, "temperature": 0.2,
                          "top_p": 0.9, "stop_sequences": ["STOP"]}, GLM)

    assert kwargs["inferenceConfig"] == {
        "maxTokens": 100, "temperature": 0.2, "topP": 0.9, "stopSequences": ["STOP"],
    }


def test_an_empty_message_is_skipped_rather_than_padded():
    """Converse rejects a message with empty content, and a blank turn carries no
    information — padding it with "" would send the model a turn that was never there."""
    kwargs = to_converse({"messages": [
        {"role": "user", "content": []},
        {"role": "user", "content": "real"},
    ]}, GLM)

    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "real"}]}]


def test_thinking_becomes_additional_model_fields_where_supported():
    kwargs = to_converse({"messages": [],
                          "thinking": {"type": "enabled", "budget_tokens": 31999}},
                         "us.deepseek.r1-v1:0")

    assert kwargs["additionalModelRequestFields"] == {
        "thinking": {"type": "enabled", "budget_tokens": 31999},
    }


def test_thinking_is_dropped_for_a_model_that_has_no_such_parameter():
    kwargs = to_converse({"messages": [],
                          "thinking": {"type": "enabled", "budget_tokens": 31999}},
                         NOVA)

    assert "additionalModelRequestFields" not in kwargs


def test_anthropic_only_fields_are_dropped():
    """anthropic_beta / metadata / anthropic_version have no Converse home. Smuggling
    them into additionalModelRequestFields would make every call a request error."""
    kwargs = to_converse({
        "messages": [], "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        "metadata": {"user_id": "x"},
    }, GLM)

    for key in ("anthropic_version", "anthropic_beta", "metadata"):
        assert key not in json.dumps(kwargs)


# ---------------------------------------------------------------------------
# Tool fidelity — the question that decides whether this is an agent harness
# ---------------------------------------------------------------------------


def test_tool_definitions_translate_to_a_tool_config():
    kwargs = to_converse({"messages": [], "tools": [
        {"name": "Bash", "description": "run a command",
         "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    ]}, GLM)

    spec = kwargs["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "Bash"
    assert spec["description"] == "run a command"
    assert spec["inputSchema"]["json"]["properties"]["command"]["type"] == "string"


def test_a_tool_without_a_schema_still_gets_a_valid_one():
    """Converse rejects a toolSpec with no inputSchema, and some tools genuinely take
    no arguments."""
    kwargs = to_converse({"messages": [], "tools": [{"name": "NoArgs"}]}, GLM)

    assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"] == {
        "json": {"type": "object", "properties": {}},
    }


@pytest.mark.parametrize("anthropic,converse", [
    ({"type": "auto"}, {"auto": {}}),
    ({"type": "any"}, {"any": {}}),
    ({"type": "tool", "name": "Bash"}, {"tool": {"name": "Bash"}}),
])
def test_tool_choice_translates(anthropic, converse):
    kwargs = to_converse({"messages": [], "tools": [{"name": "Bash"}],
                          "tool_choice": anthropic}, GLM)

    assert kwargs["toolConfig"]["toolChoice"] == converse


def test_tool_choice_none_omits_the_field_rather_than_forcing_use():
    """Converse has no "none"; forcing tool use would be the opposite of the request,
    and omitting toolChoice means auto, which is the closest honest reading."""
    kwargs = to_converse({"messages": [], "tools": [{"name": "Bash"}],
                          "tool_choice": {"type": "none"}}, GLM)

    assert "toolChoice" not in kwargs["toolConfig"]


def test_a_full_tool_round_trip_survives_translation():
    """The end-to-end fidelity case: assistant asks for a tool, the harness replies
    with a result, and the next request must carry both."""
    kwargs = to_converse({"messages": [
        {"role": "user", "content": "what is 21 doubled?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "let me compute"},
            {"type": "tool_use", "id": "tu_1", "name": "calc", "input": {"x": 21}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "42"},
        ]},
    ]}, GLM)

    assistant = kwargs["messages"][1]["content"]
    assert assistant[0] == {"text": "let me compute"}
    assert assistant[1]["toolUse"] == {"toolUseId": "tu_1", "name": "calc",
                                       "input": {"x": 21}}
    result = kwargs["messages"][2]["content"][0]["toolResult"]
    assert result == {"toolUseId": "tu_1", "content": [{"text": "42"}],
                      "status": "success"}


def test_a_failed_tool_result_keeps_its_error_status():
    """A harness that reports every tool failure as a success teaches the model that
    its broken commands worked."""
    kwargs = to_converse({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "boom",
         "is_error": True},
    ]}]}, GLM)

    assert kwargs["messages"][0]["content"][0]["toolResult"]["status"] == "error"


def test_a_structured_tool_result_is_serialised_not_dropped():
    kwargs = to_converse({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": {"rows": 3}},
    ]}]}, GLM)

    text = kwargs["messages"][0]["content"][0]["toolResult"]["content"][0]["text"]
    assert json.loads(text) == {"rows": 3}


def test_a_thinking_block_round_trips_as_reasoning_content():
    kwargs = to_converse({"messages": [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
    ]}]}, "us.deepseek.r1-v1:0")

    reasoning = kwargs["messages"][0]["content"][0]["reasoningContent"]
    assert reasoning["reasoningText"] == {"text": "hmm", "signature": "sig"}


# ---------------------------------------------------------------------------
# Usage fidelity — the four buckets must survive
# ---------------------------------------------------------------------------


def test_usage_keeps_the_cache_buckets_apart():
    """Both sides count cache tokens outside the plain input figure, so the mapping is
    direct — but it has to be done. Collapsing here would reintroduce the mispricing on
    every proxied call."""
    out = usage_to_anthropic({
        "inputTokens": 910, "outputTokens": 5,
        "cacheReadInputTokens": 15_345, "cacheWriteInputTokens": 900,
        "totalTokens": 17_160,
    })

    assert out == {"input_tokens": 910, "output_tokens": 5,
                   "cache_read_input_tokens": 15_345,
                   "cache_creation_input_tokens": 900}


def test_absent_usage_reads_as_zeros_not_as_missing_keys():
    """The CLI's parser requires the keys; omitting them is a malformed stream."""
    assert usage_to_anthropic(None) == {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }


def test_usage_reaches_the_message_delta_frame():
    frames = _frames([
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 2,
                                "cacheReadInputTokens": 100,
                                "cacheWriteInputTokens": 7}}},
    ])

    delta = next(p for t, p in frames if t == "message_delta")
    assert delta["usage"]["cache_read_input_tokens"] == 100
    assert delta["usage"]["cache_creation_input_tokens"] == 7


# ---------------------------------------------------------------------------
# Converse stream -> Anthropic SSE
# ---------------------------------------------------------------------------


def test_the_stream_has_the_frames_the_cli_parser_requires():
    frames = _frames([
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "PONG"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
    ])

    types = [t for t, _ in frames]
    assert types[0] == "message_start"
    assert types[-2:] == ["message_delta", "message_stop"]
    assert "content_block_delta" in types


def test_a_text_block_start_is_synthesised_when_converse_omits_it():
    """Converse sends no contentBlockStart for a plain text block, but Anthropic's
    parser requires one — without this the CLI reports a malformed stream."""
    frames = _frames([
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ])

    starts = [p for t, p in frames if t == "content_block_start"]
    assert starts and starts[0]["content_block"] == {"type": "text", "text": ""}


def test_a_tool_use_stream_becomes_input_json_deltas():
    frames = _frames([
        {"contentBlockStart": {"contentBlockIndex": 0,
                               "start": {"toolUse": {"toolUseId": "tu_1",
                                                     "name": "Bash"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"toolUse": {"input": '{"command"'}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"toolUse": {"input": ': "ls"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
    ])

    start = next(p for t, p in frames if t == "content_block_start")
    assert start["content_block"] == {"type": "tool_use", "id": "tu_1",
                                      "name": "Bash", "input": {}}
    partials = [p["delta"]["partial_json"] for t, p in frames
                if t == "content_block_delta"]
    assert json.loads("".join(partials)) == {"command": "ls"}
    assert next(p for t, p in frames
                if t == "message_delta")["delta"]["stop_reason"] == "tool_use"


def test_reasoning_deltas_become_thinking_deltas():
    frames = _frames([
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"reasoningContent": {"text": "hmm"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"reasoningContent": {"signature": "sig"}}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ])

    kinds = [p["delta"]["type"] for t, p in frames if t == "content_block_delta"]
    assert kinds == ["thinking_delta", "signature_delta"]


def test_an_unclosed_block_is_closed_before_the_message_ends():
    """An unbalanced block reads to the CLI as a malformed stream rather than as a
    provider error, which sends a debugging session in the wrong direction."""
    frames = _frames([
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ])

    types = [t for t, _ in frames]
    assert types.index("content_block_stop") < types.index("message_delta")


@pytest.mark.parametrize("converse,anthropic", [
    ("end_turn", "end_turn"),
    ("tool_use", "tool_use"),
    ("max_tokens", "max_tokens"),
    ("stop_sequence", "stop_sequence"),
    ("guardrail_intervened", "stop_sequence"),
    ("content_filtered", "stop_sequence"),
])
def test_stop_reasons_map(converse, anthropic):
    frames = _frames([{"messageStop": {"stopReason": converse}}])

    assert next(p for t, p in frames
                if t == "message_delta")["delta"]["stop_reason"] == anthropic


def test_a_guardrail_stop_is_preserved_alongside_the_mapped_one():
    """Anthropic has no guardrail stop reason, so the mapping loses information — and a
    study that cannot tell a refusal from a normal end is missing its interesting rows.
    The original is carried through rather than discarded."""
    frames = _frames([{"messageStop": {"stopReason": "content_filtered"}}])

    delta = next(p for t, p in frames if t == "message_delta")
    assert delta["delta"]["stop_reason"] == "stop_sequence"
    assert delta["chia_stop_reason"] == "content_filtered"


def test_the_stream_is_lazy():
    """A generator, so the CLI receives tokens as Bedrock produces them. Buffering the
    turn would make the proxy look slower than the native path for reasons that have
    nothing to do with translation."""
    consumed = []

    def events():
        for index in range(3):
            consumed.append(index)
            yield {"contentBlockDelta": {"contentBlockIndex": 0,
                                         "delta": {"text": str(index)}}}

    stream = to_anthropic_sse(events(), model=GLM)
    next(stream)  # message_start only

    assert consumed == []


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------


def test_a_non_streaming_converse_response_becomes_an_anthropic_message():
    out = to_anthropic_message({
        "output": {"message": {"role": "assistant", "content": [
            {"text": "PONG"},
            {"toolUse": {"toolUseId": "tu_1", "name": "Bash", "input": {"c": "ls"}}},
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 3},
    }, model=GLM)

    assert out["role"] == "assistant"
    assert out["content"][0] == {"type": "text", "text": "PONG"}
    assert out["content"][1]["type"] == "tool_use"
    assert out["stop_reason"] == "tool_use"
    assert out["usage"]["cache_read_input_tokens"] == 3


# ---------------------------------------------------------------------------
# The recorded CLI request
# ---------------------------------------------------------------------------


def test_the_recorded_cli_request_still_translates():
    """Replays a real capture from claude-cli 2.1.222. A CLI change that alters the
    schema fails here instead of failing a grid."""
    recorded = json.loads(RECORDED_REQUEST.read_text())
    assert "2.1.222" in recorded["captured_from"]
    body = recorded["body"]

    kwargs = to_converse(body, GLM)

    assert kwargs["modelId"] == GLM
    assert kwargs["messages"], "the user turn was lost"
    assert kwargs["system"], "the system prompt was lost"
    assert kwargs["inferenceConfig"]["maxTokens"] == body["max_tokens"]
    # All ten tools the CLI advertises in a bare session must survive, or the harness
    # silently degrades to a chatbot.
    assert len(kwargs["toolConfig"]["tools"]) == len(body["tools"])
    assert {t["toolSpec"]["name"] for t in kwargs["toolConfig"]["tools"]} == \
        {t["name"] for t in body["tools"]}
    for spec in kwargs["toolConfig"]["tools"]:
        assert "json" in spec["toolSpec"]["inputSchema"]


def test_the_recorded_request_pins_the_measured_shape():
    """The facts the whole design rests on, asserted against the capture rather than
    quoted from a note: model id in the path, Anthropic Messages body, not Converse."""
    recorded = json.loads(RECORDED_REQUEST.read_text())

    assert recorded["path"].startswith("/model/")
    assert recorded["path"].endswith("/invoke-with-response-stream")
    body = recorded["body"]
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert "messages" in body and "system" in body and "tools" in body
    assert "inferenceConfig" not in body   # i.e. it is not Converse
    assert "modelId" not in body           # the id is in the path
