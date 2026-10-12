"""Anthropic Messages ⇄ Bedrock Converse translation.

Pure functions, no I/O and no boto3, so every mapping is testable against a recorded
payload rather than against a live provider. :mod:`chia.models.proxy.server` is the thin
HTTP shell around this.

**Why this exists.** The Claude Code CLI is the most capable agent harness chia can
drive — Task subagents, hooks, session resume, MCP tools — and its Bedrock mode sends
Anthropic Messages payloads. Non-Anthropic Bedrock models (GLM, Nova, Qwen, Llama)
reject that schema; they speak Converse. Translating between the two is what makes
"run GLM-5 as the orchestrator with Nova Pro as the delegate, through Claude Code"
a configuration rather than a rewrite.

**Measured, not assumed.** Against ``claude-cli 2.1.222`` the CLI sends::

    POST /model/{modelId}/invoke-with-response-stream

with the model id in the **path**, and a body of ``anthropic_version``,
``max_tokens``, ``messages``, ``system`` (as content blocks), ``tools``, ``thinking``,
``metadata`` and ``anthropic_beta``. A recorded capture of exactly that request is in
``test/fixtures/cli_2_1_222_invoke_stream.json`` and the tests replay it, so a CLI
change that alters the schema fails a test instead of failing a grid.

**Three things do not survive translation, and saying so is part of the contract:**

* ``thinking`` becomes ``additionalModelRequestFields`` when the target model accepts
  it, and is dropped otherwise — Converse has no portable thinking parameter.
* ``anthropic_beta`` and ``metadata`` are Anthropic-specific and are dropped.
* Anthropic's ``cache_control`` breakpoints become Converse ``cachePoint`` blocks only
  where the target family supports prompt caching; elsewhere they are dropped, and the
  usage that comes back will report no cache tokens because there were none.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

#: Model-id substrings whose family speaks Anthropic Messages natively, so the CLI's
#: request can be forwarded to Bedrock untranslated.
ANTHROPIC_FAMILIES = ("anthropic.", "claude")

#: Converse ``stopReason`` -> Anthropic ``stop_reason``. Anthropic has no vocabulary for
#: a guardrail or a content filter, so both map to ``"stop_sequence"``: the turn ended
#: for a reason outside the model's control, which is the closest honest reading. The
#: original is preserved in the ``message_delta``'s ``chia_stop_reason`` so a caller can
#: still tell them apart.
STOP_REASONS = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "guardrail_intervened": "stop_sequence",
    "content_filtered": "stop_sequence",
}

#: Model families that accept a Converse ``reasoning_config`` / thinking block.
THINKING_CAPABLE = ("claude", "anthropic.", "deepseek", "qwen3", "glm")

#: Model families with Bedrock prompt caching, i.e. where a ``cachePoint`` is honored.
CACHE_CAPABLE = ("claude", "anthropic.", "nova")

#: Per-family ``maxTokens`` ceilings, longest-key-wins like the model registry.
#:
#: ``max_tokens`` is **not portable**. The Claude Code CLI sends one value for every
#: model — 32000, observed — and each Converse family enforces its own limit. Passing the
#: CLI's value straight through makes Bedrock reject the request with a
#: ``ValidationException``, the CLI retries, and the run fails in a way that looks like a
#: provider outage rather than a translation bug. This was found by running the grid, not
#: by reading the docs: every Nova cell failed with "The maximum tokens you requested
#: exceeds the model limit of 10000".
#:
#: Clamping is the faithful translation. A request the target cannot serve is not a
#: faithful rendering of a request the source could, and refusing instead would make a
#: whole model family unusable over a parameter the caller never chose.
MAX_OUTPUT_TOKENS = {
    "nova-micro": 10_000,
    "nova-lite": 10_000,
    "nova-pro": 10_000,
    "nova": 10_000,
    "llama": 8_192,
    "mistral": 8_192,
    "command": 4_096,
}

#: Ceiling applied when no family entry matches — the CLI's own observed request, so a
#: family with a higher real limit loses nothing it was going to use.
DEFAULT_MAX_OUTPUT_TOKENS = 32_000


def is_anthropic_model(model_id: str) -> bool:
    """Whether *model_id* speaks Anthropic Messages natively.

    :param model_id: A Bedrock model or inference-profile id.
    :type model_id: str
    :rtype: bool

    Such a request needs no translation at all and is forwarded verbatim, which keeps
    the proxy transparent for the models it does not need to touch — and keeps the
    Anthropic arm of a harness comparison free of any translation artefact.
    """
    lowered = (model_id or "").lower()
    return any(token in lowered for token in ANTHROPIC_FAMILIES)


def _family_supports(model_id: str, families: Iterable[str]) -> bool:
    lowered = (model_id or "").lower()
    return any(token in lowered for token in families)


def max_output_tokens(model_id: str) -> int:
    """The ``maxTokens`` ceiling for *model_id*.

    :param model_id: A Bedrock model or inference-profile id.
    :type model_id: str
    :rtype: int

    Resolved longest-key-wins over :data:`MAX_OUTPUT_TOKENS`, falling back to
    :data:`DEFAULT_MAX_OUTPUT_TOKENS`. See that constant for why clamping rather than
    forwarding is the faithful translation.
    """
    lowered = (model_id or "").lower()
    best: Optional[str] = None
    for key in MAX_OUTPUT_TOKENS:
        if key in lowered and (best is None or (len(key), key) > (len(best), best)):
            best = key
    return MAX_OUTPUT_TOKENS[best] if best else DEFAULT_MAX_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# Anthropic Messages -> Bedrock Converse
# ---------------------------------------------------------------------------


def _content_blocks(content: Any) -> List[dict]:
    """Anthropic message content -> Converse content blocks.

    Accepts the string shorthand as well as the block list, because both appear in
    real payloads.
    """
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return []

    out: List[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text") or ""
            if text:
                out.append({"text": text})
        elif kind == "tool_use":
            out.append({"toolUse": {
                "toolUseId": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }})
        elif kind == "tool_result":
            out.append({"toolResult": {
                "toolUseId": block.get("tool_use_id", ""),
                "content": _tool_result_content(block.get("content")),
                "status": "error" if block.get("is_error") else "success",
            }})
        elif kind == "thinking":
            # Round-tripped so a multi-turn conversation keeps its reasoning context on
            # models that accept it; models that do not will reject the block, which is
            # why `thinking` is only forwarded to THINKING_CAPABLE families.
            out.append({"reasoningContent": {
                "reasoningText": {
                    "text": block.get("thinking", ""),
                    "signature": block.get("signature", ""),
                },
            }})
        elif kind == "image":
            image = _image_block(block)
            if image is not None:
                out.append(image)
        elif kind == "document":
            # No portable Converse equivalent for an arbitrary document block; its text
            # is preserved rather than the block being silently dropped.
            text = json.dumps(block.get("source", {}))[:4000]
            out.append({"text": f"[document] {text}"})
    return out


def _tool_result_content(content: Any) -> List[dict]:
    """Anthropic tool_result content -> Converse toolResult content blocks."""
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        blocks = _content_blocks(content)
        return blocks or [{"text": ""}]
    if content is None:
        return [{"text": ""}]
    return [{"text": json.dumps(content)}]


def _image_block(block: dict) -> Optional[dict]:
    """Anthropic image block -> Converse image block, or ``None`` if untranslatable."""
    source = block.get("source") or {}
    if source.get("type") != "base64":
        # A URL source has no Converse equivalent; dropping it is better than sending a
        # block the model will reject and blaming the model.
        return None
    media_type = str(source.get("media_type", ""))
    fmt = media_type.rsplit("/", 1)[-1] or "png"
    import base64

    try:
        raw = base64.b64decode(source.get("data", ""), validate=False)
    except Exception:
        return None
    return {"image": {"format": fmt, "source": {"bytes": raw}}}


def _system_blocks(system: Any, model_id: str) -> List[dict]:
    """Anthropic ``system`` -> Converse ``system``, translating cache breakpoints."""
    if isinstance(system, str):
        return [{"text": system}] if system else []
    if not isinstance(system, list):
        return []
    out: List[dict] = []
    cache_ok = _family_supports(model_id, CACHE_CAPABLE)
    for block in system:
        if isinstance(block, str):
            out.append({"text": block})
            continue
        if not isinstance(block, dict):
            continue
        text = block.get("text") or ""
        if text:
            out.append({"text": text})
        if block.get("cache_control") and cache_ok:
            out.append({"cachePoint": {"type": "default"}})
    return out


def _tool_config(body: dict) -> Optional[dict]:
    """Anthropic ``tools``/``tool_choice`` -> Converse ``toolConfig``."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    specs = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        schema = tool.get("input_schema") or {"type": "object", "properties": {}}
        spec = {"name": tool["name"], "inputSchema": {"json": schema}}
        if tool.get("description"):
            spec["description"] = tool["description"]
        specs.append({"toolSpec": spec})
    if not specs:
        return None
    config: Dict[str, Any] = {"tools": specs}

    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type")
        if kind == "any":
            config["toolChoice"] = {"any": {}}
        elif kind == "tool" and choice.get("name"):
            config["toolChoice"] = {"tool": {"name": choice["name"]}}
        elif kind == "auto":
            config["toolChoice"] = {"auto": {}}
        # "none" has no Converse equivalent; omitting toolChoice is auto, and forcing
        # tool use would be the opposite of what was asked.
    return config


def to_converse(body: dict, model_id: str) -> dict:
    """Translate an Anthropic Messages request into Converse kwargs.

    :param body: The CLI's request body (Bedrock-flavour Anthropic Messages).
    :param model_id: Target Bedrock model id, taken from the request path.
    :type body: dict
    :type model_id: str
    :returns: kwargs for ``bedrock-runtime``'s ``converse`` / ``converse_stream``.
    :rtype: dict

    Everything Converse has no place for is dropped rather than smuggled into a field
    that happens to accept it; the module docstring lists what and why.
    """
    messages = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        blocks = _content_blocks(message.get("content"))
        if not blocks:
            # Converse rejects a message with empty content, and an empty turn carries
            # no information, so it is skipped rather than padded with a blank string.
            continue
        role = message.get("role")
        messages.append({"role": "assistant" if role == "assistant" else "user",
                         "content": blocks})

    kwargs: Dict[str, Any] = {"modelId": model_id, "messages": messages}

    system = _system_blocks(body.get("system"), model_id)
    if system:
        kwargs["system"] = system

    inference: Dict[str, Any] = {}
    if isinstance(body.get("max_tokens"), int):
        # Clamped, not forwarded: see MAX_OUTPUT_TOKENS. The CLI's value is the upper
        # bound the caller asked for, so min() never grants more than was requested.
        inference["maxTokens"] = min(body["max_tokens"], max_output_tokens(model_id))
    for source, dest in (("temperature", "temperature"), ("top_p", "topP")):
        if isinstance(body.get(source), (int, float)):
            inference[dest] = body[source]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        inference["stopSequences"] = [str(s) for s in stops]
    if inference:
        kwargs["inferenceConfig"] = inference

    tool_config = _tool_config(body)
    if tool_config:
        kwargs["toolConfig"] = tool_config

    thinking = body.get("thinking")
    if (isinstance(thinking, dict) and thinking.get("type") == "enabled"
            and _family_supports(model_id, THINKING_CAPABLE)):
        kwargs["additionalModelRequestFields"] = {
            "thinking": {"type": "enabled",
                         "budget_tokens": thinking.get("budget_tokens", 1024)},
        }
    return kwargs


# ---------------------------------------------------------------------------
# Bedrock Converse stream -> Anthropic SSE
# ---------------------------------------------------------------------------


def usage_to_anthropic(usage: Optional[dict]) -> dict:
    """Converse ``usage`` -> Anthropic ``usage``, keeping the cache buckets.

    :param usage: A Converse ``metadata.usage`` dict, or ``None``.
    :type usage: Optional[dict]
    :rtype: dict

    Both sides count cache tokens *outside* the plain input figure, so the mapping is
    direct — but it has to be done at all. Collapsing the buckets here would silently
    reintroduce the mispricing that ``chia.base.usage`` exists to prevent, on every
    call that goes through the proxy.
    """
    usage = usage or {}

    def _int(key: str) -> int:
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) else 0

    out = {
        "input_tokens": _int("inputTokens"),
        "output_tokens": _int("outputTokens"),
        "cache_read_input_tokens": _int("cacheReadInputTokens"),
        "cache_creation_input_tokens": _int("cacheWriteInputTokens"),
    }
    return out


def _sse(event_type: str, payload: dict) -> bytes:
    """One Anthropic SSE frame."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def to_anthropic_sse(
    events: Iterable[dict],
    *,
    model: str,
    message_id: str = "msg_chia_proxy",
    on_usage: Optional[Callable[[dict], None]] = None,
) -> Iterator[bytes]:
    """Translate a Converse event stream into Anthropic SSE frames.

    :param events: Converse stream events (``messageStart``, ``contentBlockDelta``, ...).
    :param model: Model id to echo back in ``message_start``.
    :param message_id: Id to echo back; the CLI does not require a real one.
    :param on_usage: Called once with the final Anthropic-shaped usage dict, before the
        closing frames are emitted. This is the only place the *real* token counts for a
        proxied call exist: the client downstream prices what it believes it called, so
        for a non-Anthropic model its self-reported cost is wrong by whatever the two
        models' rates differ by. A caller that wants correct accounting has to read the
        counts here.
    :type events: Iterable[dict]
    :type model: str
    :type message_id: str
    :type on_usage: Optional[Callable[[dict], None]]
    :rtype: Iterator[bytes]

    A generator, so the CLI starts receiving tokens as Bedrock produces them rather
    than after the turn completes — an agent loop's wall-clock is dominated by waiting
    for output, and buffering the whole turn would make the proxy look slower than the
    native path for reasons that have nothing to do with translation.

    Frame order follows the Anthropic streaming contract the CLI parses:
    ``message_start`` → per-block ``content_block_start`` / ``content_block_delta`` /
    ``content_block_stop`` → ``message_delta`` (stop reason + usage) → ``message_stop``.
    """
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage_to_anthropic(None),
        },
    })

    stop_reason = "end_turn"
    converse_stop_reason = ""
    usage: dict = {}
    open_blocks: set = set()

    for event in events:
        if not isinstance(event, dict):
            continue

        if "contentBlockStart" in event:
            index = int(event["contentBlockStart"].get("contentBlockIndex", 0))
            start = event["contentBlockStart"].get("start") or {}
            tool_use = start.get("toolUse") or {}
            block = ({"type": "tool_use", "id": tool_use.get("toolUseId", ""),
                      "name": tool_use.get("name", ""), "input": {}}
                     if tool_use else {"type": "text", "text": ""})
            open_blocks.add(index)
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": block,
            })

        elif "contentBlockDelta" in event:
            payload = event["contentBlockDelta"]
            index = int(payload.get("contentBlockIndex", 0))
            delta = payload.get("delta") or {}
            if index not in open_blocks:
                # Converse omits contentBlockStart for a plain text block; Anthropic's
                # parser requires one, so synthesise it on first delta.
                open_blocks.add(index)
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""},
                })
            if "text" in delta:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": delta["text"]},
                })
            elif "toolUse" in delta:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": delta["toolUse"].get("input", "")},
                })
            elif "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                if "text" in reasoning:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "thinking_delta",
                                  "thinking": reasoning["text"]},
                    })
                elif "signature" in reasoning:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "signature_delta",
                                  "signature": reasoning["signature"]},
                    })

        elif "contentBlockStop" in event:
            index = int(event["contentBlockStop"].get("contentBlockIndex", 0))
            open_blocks.discard(index)
            yield _sse("content_block_stop",
                       {"type": "content_block_stop", "index": index})

        elif "messageStop" in event:
            converse_stop_reason = str(event["messageStop"].get("stopReason", ""))
            stop_reason = STOP_REASONS.get(converse_stop_reason, "end_turn")

        elif "metadata" in event:
            usage = usage_to_anthropic(event["metadata"].get("usage"))

    # Close anything Converse left open, so the CLI's parser never sees an unbalanced
    # block (which it reports as a malformed stream rather than as a provider error).
    for index in sorted(open_blocks):
        yield _sse("content_block_stop",
                   {"type": "content_block_stop", "index": index})

    final_usage = usage or usage_to_anthropic(None)
    if on_usage is not None:
        # Deliberately not wrapped in try/except: a recorder that raises is a bug in the
        # recorder, and swallowing it here would produce a stream that looks fine while
        # silently accounting for nothing.
        on_usage(dict(final_usage))

    delta_payload = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": final_usage,
    }
    if converse_stop_reason and converse_stop_reason != stop_reason:
        # Preserved rather than lost: Anthropic has no guardrail/content-filter stop
        # reason, and a study that cannot tell a refusal from a normal end is missing
        # the interesting rows.
        delta_payload["chia_stop_reason"] = converse_stop_reason
    yield _sse("message_delta", delta_payload)
    yield _sse("message_stop", {"type": "message_stop"})


def to_anthropic_message(
    response: dict,
    *,
    model: str,
    message_id: str = "msg_chia_proxy",
) -> dict:
    """Translate a non-streaming Converse response into an Anthropic Messages reply.

    :param response: A ``bedrock-runtime`` ``converse`` response.
    :param model: Model id to echo back.
    :param message_id: Id to echo back.
    :type response: dict
    :type model: str
    :type message_id: str
    :rtype: dict

    The CLI uses ``invoke-with-response-stream`` in practice, but it does fall back to
    the non-streaming ``invoke`` path, and the recorded probe caught it doing so.
    """
    message = ((response.get("output") or {}).get("message") or {})
    content: List[dict] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            content.append({"type": "text", "text": block["text"]})
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            content.append({"type": "tool_use", "id": tool_use.get("toolUseId", ""),
                            "name": tool_use.get("name", ""),
                            "input": tool_use.get("input", {})})
        elif "reasoningContent" in block:
            reasoning = (block["reasoningContent"].get("reasoningText") or {})
            content.append({"type": "thinking",
                            "thinking": reasoning.get("text", ""),
                            "signature": reasoning.get("signature", "")})

    converse_stop = str(response.get("stopReason", "") or "")
    out = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": STOP_REASONS.get(converse_stop, "end_turn"),
        "stop_sequence": None,
        "usage": usage_to_anthropic(response.get("usage")),
    }
    if converse_stop and converse_stop != out["stop_reason"]:
        out["chia_stop_reason"] = converse_stop
    return out
