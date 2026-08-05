"""A local Bedrock-shaped endpoint that lets the Claude Code CLI drive any Bedrock model.

Run this as a per-worker sidecar and point the CLI at it::

    python -m chia.models.proxy.server --port 8123 &

    CLAUDE_CODE_USE_BEDROCK=1 \\
    CLAUDE_CODE_SKIP_BEDROCK_AUTH=1 \\
    CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD=1 \\
    ANTHROPIC_BEDROCK_BASE_URL=http://127.0.0.1:8123 \\
    ANTHROPIC_MODEL=zai.glm-5 \\
    CLAUDE_CODE_SUBAGENT_MODEL=us.amazon.nova-pro-v1:0 \\
        claude --print -p 'list the files here'

Requests for Anthropic models are forwarded verbatim; everything else is translated to
Converse and streamed back as Anthropic SSE (see :mod:`chia.models.proxy.translate`).
That is what makes ``ModelTier(primary="glm5", subagent="nova-pro")`` a configuration
rather than a rewrite: ``ANTHROPIC_MODEL`` and ``CLAUDE_CODE_SUBAGENT_MODEL`` are
separate levers, so one proxy can route each tier to a different provider.

``fastapi`` is already a pinned chia dependency, so this adds nothing to the dependency
surface — the decisive advantage over pulling in a general-purpose LLM gateway, and it
keeps tool-call and cache-token fidelity under chia's own tests.

Three env vars, and why each is needed
--------------------------------------

``CLAUDE_CODE_USE_BEDROCK=1``
    Selects the CLI's Bedrock transport, which is what sends the Anthropic-Messages
    body to ``/model/{id}/invoke-with-response-stream``.

``CLAUDE_CODE_SKIP_BEDROCK_AUTH=1``
    Skips SigV4 entirely: verified against ``claude-cli 2.1.222``, the invoke request
    carries no ``Authorization``, ``x-amz-date``, ``x-amz-security-token``,
    ``x-amz-content-sha256`` or ``x-api-key`` header at all. The proxy therefore needs
    no signature verification — and must only ever be bound to loopback, since anything
    that can reach it can spend the credentials it holds.

``CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD=1``
    **A documented deviation, not a default.** In Bedrock mode the CLI expects
    ``application/vnd.amazon.eventstream`` — AWS binary event-stream framing — and
    rejects ``text/event-stream`` by name. The guard exists for a good reason: a real
    Bedrock gateway must pass binary framing through. Disabling it lets this proxy
    reply with plain Anthropic SSE, which is why it needs no eventstream encoder. The
    flag is undocumented CLI surface and could change, so the tests pin the version it
    was verified against (2.1.222) in their fixtures.

Security posture
----------------

The proxy authenticates nobody and holds live AWS credentials via the standard boto3
chain. It binds ``127.0.0.1`` by default and refuses any other host unless
``--i-understand-this-is-unauthenticated`` is passed, because a sidecar reachable from
the network is a credential-spending endpoint reachable from the network.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Iterator, Optional

# Imported at module scope, not lazily: FastAPI resolves a route handler's annotations
# with get_type_hints, and under `from __future__ import annotations` a name bound only
# inside build_app cannot be resolved — it would silently treat `request: Request` as a
# query parameter and answer every call with a 422. fastapi is a pinned chia dependency,
# so there is nothing to defer.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chia.models.proxy.translate import (
    is_anthropic_model,
    to_anthropic_message,
    to_anthropic_sse,
    to_converse,
)

logger = logging.getLogger("chia.models.proxy")

#: The CLI version this proxy's request/response shapes were verified against.
VERIFIED_CLI_VERSION = "2.1.222"

#: Loopback only. See "Security posture" in the module docstring.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8123


def _client(region: Optional[str] = None):
    """A ``bedrock-runtime`` client. Imported lazily so importing this module is cheap."""
    import boto3

    resolved = (region or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    return boto3.client("bedrock-runtime", region_name=resolved)


def _forward_anthropic_stream(client, model_id: str, body: dict) -> Iterator[bytes]:
    """Forward an Anthropic-model request to Bedrock and re-emit it as Anthropic SSE.

    The body needs no translation, but the *response* does: Bedrock replies in AWS
    binary event-stream framing, and the CLI has been told (by the content-type guard
    flag) to expect SSE. boto3 already decodes the framing for us, so this only has to
    re-frame each chunk — the payloads inside are already Anthropic events.
    """
    response = client.invoke_model_with_response_stream(
        modelId=model_id, body=json.dumps(body),
        contentType="application/json", accept="application/json",
    )
    for event in response.get("body", []):
        chunk = (event or {}).get("chunk") or {}
        raw = chunk.get("bytes")
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        event_type = payload.get("type") or "message_delta"
        yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _converse_stream(client, model_id: str, body: dict) -> Iterator[bytes]:
    """Translate to Converse, stream it, and re-emit as Anthropic SSE."""
    kwargs = to_converse(body, model_id)
    response = client.converse_stream(**kwargs)
    events = (event for event in response.get("stream", []) if isinstance(event, dict))
    return to_anthropic_sse(events, model=model_id)


def build_app(region: Optional[str] = None, client=None):
    """Build the FastAPI app.

    :param region: AWS region for the outbound leg; defaults to the environment.
    :param client: Injected ``bedrock-runtime`` client, for tests. When given, no boto3
        client is constructed and no credentials are needed.
    :type region: Optional[str]
    :rtype: fastapi.FastAPI

    Routes mirror what the CLI was measured to call, and nothing more:

    ``GET /inference-profiles``
        The aws-sdk-js discovery call the CLI makes before its first invoke. An empty
        summary list is a valid answer and the CLI proceeds — verified, not assumed.
    ``POST /model/{model_id}/invoke-with-response-stream``
        The streaming path, which is what the CLI actually uses.
    ``POST /model/{model_id}/invoke``
        The non-streaming fallback. The probe caught the CLI using it, so it is
        implemented rather than left to 404 mid-session.
    ``GET /healthz``
        So a launcher can wait for readiness instead of sleeping.
    """
    app = FastAPI(title="chia bedrock-converse proxy", docs_url=None, redoc_url=None)
    app.state.client = client
    app.state.region = region

    def _get_client():
        if app.state.client is None:
            app.state.client = _client(app.state.region)
        return app.state.client

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "verified_cli_version": VERIFIED_CLI_VERSION}

    @app.get("/inference-profiles")
    async def inference_profiles():
        # The CLI only needs a well-formed response here; it does not require the list
        # to be non-empty, and inventing profiles would make it request ids that do not
        # exist.
        return {"inferenceProfileSummaries": []}

    # One catch-all rather than two literal routes, because a Bedrock model id may
    # itself contain slashes (an inference-profile ARN does) and Starlette's ``:path``
    # converter is greedy — it cannot be followed by further path segments. So the
    # action is split off the tail here instead of being matched by the router.
    @app.post("/model/{rest:path}")
    async def model_post(rest: str, request: Request):
        for suffix, streaming in (("/invoke-with-response-stream", True),
                                  ("/invoke", False)):
            if rest.endswith(suffix):
                model_id = rest[: -len(suffix)]
                break
        else:
            return JSONResponse(
                {"message": f"unsupported model action: /model/{rest}"},
                status_code=404,
            )
        if not model_id:
            return JSONResponse({"message": "no model id in path"}, status_code=400)

        body = await _json_body(request)
        if body is None:
            return JSONResponse({"message": "malformed JSON body"}, status_code=400)

        client_obj = _get_client()
        try:
            if streaming:
                frames = (_forward_anthropic_stream(client_obj, model_id, body)
                          if is_anthropic_model(model_id)
                          else _converse_stream(client_obj, model_id, body))
                # text/event-stream, which needs
                # CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD=1 on the client (see
                # the module docstring).
                return StreamingResponse(frames, media_type="text/event-stream")
            if is_anthropic_model(model_id):
                response = client_obj.invoke_model(
                    modelId=model_id, body=json.dumps(body),
                    contentType="application/json", accept="application/json",
                )
                return JSONResponse(json.loads(response["body"].read()))
            converse = client_obj.converse(**to_converse(body, model_id))
            return JSONResponse(to_anthropic_message(converse, model=model_id))
        except Exception as exc:
            logger.warning("%s failed for %s: %s",
                           "invoke-with-response-stream" if streaming else "invoke",
                           model_id, exc)
            return JSONResponse({"message": str(exc)}, status_code=502)

    return app


async def _json_body(request) -> Optional[Dict[str, Any]]:
    """The request body as a dict, or ``None`` when it is not parseable JSON."""
    try:
        payload = json.loads(await request.body())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Binds loopback only unless you explicitly opt out; it authenticates "
               "nobody and holds live AWS credentials.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--region", default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--i-understand-this-is-unauthenticated", action="store_true",
                        help="required to bind anything but loopback")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1") and not \
            args.i_understand_this_is_unauthenticated:
        parser.error(
            f"refusing to bind {args.host}: this proxy authenticates nobody and can "
            f"spend the AWS credentials it holds. Pass "
            f"--i-understand-this-is-unauthenticated to override."
        )

    import uvicorn

    uvicorn.run(build_app(region=args.region), host=args.host, port=args.port,
                log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
