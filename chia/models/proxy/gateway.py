"""Provider-agnostic Anthropic Messages gateway for agent harnesses.

Claude Code speaks the Anthropic Messages wire format even when its configured
agents use different model identifiers.  This module makes that transport an
interface rather than a Bedrock special case: model ids are written as
``provider/model``, routed to a backend, and returned in Messages format.

The gateway is deliberately unauthenticated and defaults to loopback in the
CLI entry point.  Provider credentials are loaded from the environment only
when a backend makes a request; they are never stored in configuration or
included in errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Protocol, Sequence
from urllib import request as urllib_request

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chia.models.agents import ModelRef, ProviderSpec
from chia.models.proxy.translate import (
    is_anthropic_model,
    to_anthropic_message,
    to_anthropic_sse,
    to_converse,
)


VERIFIED_CLAUDE_CODE_VERSION = "2.1.233"


class MessagesBackend(Protocol):
    """Backend contract consumed by :class:`ProviderRouter`."""

    def complete(self, model: str, body: Mapping[str, Any]) -> dict:
        """Return one Anthropic Messages response."""

    def stream(self, model: str, body: Mapping[str, Any]) -> Iterable[bytes]:
        """Return Anthropic SSE frames."""


def _sse_from_message(message: Mapping[str, Any]) -> Iterator[bytes]:
    """Turn a complete response into the already-verified proxy SSE shape."""
    events = []
    for index, block in enumerate(message.get("content") or []):
        block_type = block.get("type") if isinstance(block, dict) else None
        if block_type == "text":
            events.append({"contentBlockDelta": {
                "contentBlockIndex": index, "delta": {"text": block.get("text", "")},
            }})
        elif block_type == "tool_use":
            events.append({"contentBlockStart": {
                "contentBlockIndex": index,
                "start": {"toolUse": {"toolUseId": block.get("id"),
                                       "name": block.get("name")}},
            }})
            events.append({"contentBlockDelta": {
                "contentBlockIndex": index,
                "delta": {"toolUse": {
                    "input": json.dumps(block.get("input") or {}, separators=(",", ":")),
                }},
            }})
        else:
            continue
        events.append({"contentBlockStop": {"contentBlockIndex": index}})
    stop_reason = {"end_turn": "end_turn", "tool_use": "tool_use",
                   "max_tokens": "max_tokens"}.get(message.get("stop_reason"), "end_turn")
    events.append({"messageStop": {"stopReason": stop_reason}})
    usage = dict(message.get("usage") or {})
    events.append({"metadata": {"usage": {
        "inputTokens": usage.get("input_tokens", 0),
        "outputTokens": usage.get("output_tokens", 0),
        "cacheReadInputTokens": usage.get("cache_read_input_tokens", 0),
        "cacheWriteInputTokens": usage.get("cache_creation_input_tokens", 0),
    }}})
    yield from to_anthropic_sse(
        events, model=str(message.get("model", "unknown")),
        message_id=str(message.get("id", "msg_chia_gateway")),
    )


def _sse(event: str, payload: Mapping[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


class BedrockConverseBackend:
    """Messages backend backed by boto3 Bedrock Runtime.

    Anthropic Bedrock models are forwarded in their native schema. Other model
    families are translated to and from Converse with the existing, thoroughly
    tested Chia translator.
    """

    def __init__(self, client: Any):
        self.client = client

    def complete(self, model: str, body: Mapping[str, Any]) -> dict:
        request_body = dict(body)
        request_body.pop("model", None)
        request_body.pop("stream", None)
        if is_anthropic_model(model):
            response = self.client.invoke_model(
                modelId=model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            raw = response["body"].read()
            return json.loads(raw)
        response = self.client.converse(**to_converse(request_body, model))
        return to_anthropic_message(response, model=model)

    def stream(self, model: str, body: Mapping[str, Any]) -> Iterable[bytes]:
        request_body = dict(body)
        request_body.pop("model", None)
        request_body.pop("stream", None)
        if is_anthropic_model(model):
            response = self.client.invoke_model_with_response_stream(
                modelId=model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            for event in response.get("body", []):
                raw = ((event or {}).get("chunk") or {}).get("bytes")
                if raw:
                    payload = json.loads(raw)
                    yield _sse(payload.get("type", "message_delta"), payload)
            return
        response = self.client.converse_stream(**to_converse(request_body, model))
        events = (event for event in response.get("stream", []) if isinstance(event, dict))
        yield from to_anthropic_sse(events, model=model)


@dataclass
class _HTTPResponse:
    status: int
    body: bytes


class OpenAICompatibleBackend:
    """Messages backend for a Chat Completions compatible endpoint.

    The HTTP callable is injectable, which keeps all tests offline and also
    makes gateways with custom authentication easy to support.  Streaming is
    synthesized from a non-streaming completion for broad compatibility; the
    wire contract remains streaming from Claude Code's perspective.
    """

    def __init__(
        self,
        base_url: str,
        credential_env: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        http_post=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.credential_env = credential_env
        self.options = dict(options or {})
        self._http_post = http_post or self._post

    def complete(self, model: str, body: Mapping[str, Any]) -> dict:
        outbound = _messages_to_chat_completions(model, body)
        outbound.update(self.options.get("request", {}))
        response = self._http_post(
            f"{self.base_url}/chat/completions", outbound, self._headers()
        )
        status = getattr(response, "status", 200)
        raw = getattr(response, "body", response)
        if hasattr(raw, "read"):
            raw = raw.read()
        if isinstance(raw, str):
            raw = raw.encode()
        payload = json.loads(raw)
        if status >= 400:
            message = ((payload.get("error") or {}).get("message")
                       if isinstance(payload, dict) else None)
            raise RuntimeError(message or f"OpenAI-compatible provider returned HTTP {status}")
        return _chat_completion_to_message(payload, model)

    def stream(self, model: str, body: Mapping[str, Any]) -> Iterable[bytes]:
        return _sse_from_message(self.complete(model, body))

    def _headers(self) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.credential_env:
            credential = os.environ.get(self.credential_env)
            if not credential:
                raise RuntimeError(f"credential environment variable {self.credential_env} is not set")
            headers["authorization"] = f"Bearer {credential}"
        headers.update(self.options.get("headers", {}))
        return headers

    @staticmethod
    def _post(url: str, payload: Mapping[str, Any], headers: Mapping[str, str]) -> _HTTPResponse:
        req = urllib_request.Request(
            url, data=json.dumps(payload).encode(), headers=dict(headers), method="POST"
        )
        with urllib_request.urlopen(req) as response:
            return _HTTPResponse(status=response.status, body=response.read())


def _messages_to_chat_completions(model: str, body: Mapping[str, Any]) -> dict:
    messages = []
    system = body.get("system")
    if system:
        if isinstance(system, list):
            system = "\n".join(str(block.get("text", "")) for block in system
                               if isinstance(block, dict) and block.get("type") == "text")
        messages.append({"role": "system", "content": str(system)})

    for message in body.get("messages") or []:
        role = message.get("role")
        content = message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text_parts = []
        tool_calls = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"), "type": "function",
                    "function": {"name": block.get("name"),
                                 "arguments": json.dumps(block.get("input") or {})},
                })
            elif block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if not isinstance(result_content, str):
                    result_content = json.dumps(result_content)
                messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                                 "content": result_content})
        converted: Dict[str, Any] = {"role": role, "content": "\n".join(text_parts) or None}
        if tool_calls:
            converted["tool_calls"] = tool_calls
        if text_parts or tool_calls:
            messages.append(converted)

    tools = []
    for tool in body.get("tools") or []:
        tools.append({"type": "function", "function": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        }})
    result: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": False,
    }
    if tools:
        result["tools"] = tools
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    return result


def _chat_completion_to_message(payload: Mapping[str, Any], model: str) -> dict:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible response has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            arguments = {"_raw": function.get("arguments")}
        content.append({"type": "tool_use", "id": call.get("id"),
                        "name": function.get("name"), "input": arguments})
    usage = payload.get("usage") or {}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    total_prompt = usage.get("prompt_tokens", 0) or 0
    fresh_input = max(0, total_prompt - cached)
    stop_reason = "tool_use" if message.get("tool_calls") else {
        "stop": "end_turn", "length": "max_tokens",
    }.get(choice.get("finish_reason"), "end_turn")
    return {
        "id": payload.get("id", "msg_chia_gateway"),
        "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": fresh_input,
                  "output_tokens": usage.get("completion_tokens", 0),
                  "cache_read_input_tokens": cached},
    }


class ProviderRouter:
    """Route qualified model ids to registered backend instances."""

    def __init__(self, providers: Mapping[str, MessagesBackend],
                 specs: Optional[Sequence[ProviderSpec]] = None,
                 default_provider: Optional[str] = None):
        self.providers = dict(providers)
        self.specs = {spec.id: spec for spec in specs or ()}
        self.default_provider = default_provider
        if len(self.providers) != len(providers):
            raise ValueError("duplicate provider ids")

    def resolve(self, value: str) -> tuple[MessagesBackend, str]:
        if "/" in value:
            ref = ModelRef.parse(value)
        elif self.default_provider:
            ref = ModelRef(self.default_provider, value)
        else:
            raise ValueError("model must be qualified as 'provider/model'")
        backend = self.providers.get(ref.provider)
        if backend is None:
            raise ValueError(f"unknown provider {ref.provider!r}")
        spec = self.specs.get(ref.provider)
        if spec is not None and not spec.has_model(ref.model):
            raise ValueError(f"provider {ref.provider!r} does not declare model {ref.model!r}")
        return backend, ref.model


def build_gateway_app(router: ProviderRouter) -> FastAPI:
    """Build the loopback Messages API consumed by Claude Code."""
    app = FastAPI(title="chia provider gateway", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "verified_claude_code_version": VERIFIED_CLAUDE_CODE_VERSION}

    @app.post("/v1/messages")
    async def messages(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            model = body.get("model")
            if not isinstance(model, str) or not model:
                raise ValueError("request requires a model")
            backend, backend_model = router.resolve(model)
            if body.get("stream"):
                return StreamingResponse(backend.stream(backend_model, body),
                                         media_type="text/event-stream")
            return JSONResponse(backend.complete(backend_model, body))
        except ValueError as exc:
            return JSONResponse({"type": "error", "error": {
                "type": "invalid_request_error", "message": str(exc),
            }}, status_code=400)
        except Exception as exc:
            return JSONResponse({"type": "error", "error": {
                "type": "api_error", "message": str(exc),
            }}, status_code=502)

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chia provider-agnostic Messages gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--openai-provider", default="openai")
    parser.add_argument("--openai-base-url", required=True)
    parser.add_argument("--credential-env", default="OPENAI_API_KEY")
    parser.add_argument("--i-understand-this-is-unauthenticated", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not \
            args.i_understand_this_is_unauthenticated:
        parser.error("refusing a non-loopback bind without the explicit unauthenticated override")
    backend = OpenAICompatibleBackend(args.openai_base_url, args.credential_env)
    app = build_gateway_app(ProviderRouter({args.openai_provider: backend}))
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
