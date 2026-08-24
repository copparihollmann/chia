"""A local Bedrock-shaped endpoint that lets the Claude Code CLI drive any Bedrock model.

Run this as a per-worker sidecar and point the CLI at it::

    python -m chia.models.proxy.server --port 8123 &

    CLAUDE_CODE_USE_BEDROCK=1 \\
    CLAUDE_CODE_SKIP_BEDROCK_AUTH=1 \\
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

Two env vars, and why each is needed
------------------------------------

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
    **No longer needed, and should not be used.** The proxy answers in
    ``application/vnd.amazon.eventstream`` — the framing the CLI's Bedrock transport
    actually asks for (see :mod:`chia.models.proxy.eventstream`).

    It used to be required, because the proxy replied with plain SSE. That looked like it
    worked end to end: the CLI printed the answer. It was not working. Once per-call usage
    recording was switched on, every proxied turn appeared at Bedrock **twice** — once on
    ``/invoke-with-response-stream`` and again, same body, on ``/invoke``. The CLI was
    failing to parse the SSE stream and silently retrying non-streaming, then reporting
    the cost of the one call it accepted. A 2x bill, invisible to the client's own
    accounting. Cache tokens made it unambiguous: the stream call wrote the prompt cache
    and the fallback read it, which requires both to have reached the provider.

    ``--framing sse`` keeps the old behaviour for anyone who needs it, and says in its
    help that it doubles provider spend.

Security posture
----------------

The proxy authenticates nobody and holds live AWS credentials via the standard boto3
chain. It binds ``127.0.0.1`` by default and refuses any other host unless
``--i-understand-this-is-unauthenticated`` is passed, because a sidecar reachable from
the network is a credential-spending endpoint reachable from the network.

Why it records usage
--------------------

``--usage-log`` is not telemetry garnish. A client on the far side of this proxy prices
what it *believes* it called: point the Claude Code CLI at a Nova or GLM model and it
applies Claude's rate card to the tokens, then reports the result as
``cost_source="billed"`` — authoritative. The only place the real counts for a proxied
call exist is here, in the provider's own response, so the proxy writes them down: one
JSON line per call, with the model id it was actually asked for.

It has already earned its place twice. It is what exposed the SSE double-billing above,
and it is what lets a study price a proxied call against the right rate card rather than
inheriting the client's.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

# Imported at module scope, not lazily: FastAPI resolves a route handler's annotations
# with get_type_hints, and under `from __future__ import annotations` a name bound only
# inside build_app cannot be resolved — it would silently treat `request: Request` as a
# query parameter and answer every call with a 422. fastapi is a pinned chia dependency,
# so there is nothing to defer.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chia.models.proxy.eventstream import CONTENT_TYPE as EVENTSTREAM_CONTENT_TYPE
from chia.models.proxy.eventstream import frames_from_sse
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

#: Response framings. ``eventstream`` is what the CLI's Bedrock transport asks for and the
#: only one that does not make it retry the turn non-streaming.
FRAMINGS = ("eventstream", "sse")
DEFAULT_FRAMING = "eventstream"

#: Env var read when ``--usage-log`` is not given, so a launcher can enable recording
#: without rewriting the command line.
USAGE_LOG_ENV = "CHIA_PROXY_USAGE_LOG"

#: Keys every usage line carries, in order. Stable, because downstream readers select by
#: name and a study's CSV should not change shape under a proxy upgrade.
USAGE_FIELDS = (
    "ts", "model_id", "route", "translated",
    "input_tokens", "output_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens",
)


class UsageRecorder:
    """Append-only JSONL record of what each proxied call really consumed.

    :param path: File to append to. Parent directories are created.
    :type path: Union[str, pathlib.Path]

    One line per completed call. Appends are serialised with a lock and each line is
    written with a single ``write`` of a whole line, because the CLI opens several
    concurrent streams (a subagent turn overlaps its parent's) and a half-written line is
    a row a reader silently drops.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, *, model_id: str, route: str, usage: Optional[dict],
               translated: bool) -> None:
        """Append one call's usage.

        :param model_id: The Bedrock model id as it arrived in the request path.
        :param route: ``"stream"`` or ``"invoke"``.
        :param usage: Anthropic-shaped usage dict, or ``None`` when none was reported.
        :param translated: Whether the call went through the Converse translation.
        :type model_id: str
        :type route: str
        :type usage: Optional[dict]
        :type translated: bool

        A call that reported no usage is still recorded, with zeros. Dropping it would
        make the line count disagree with the call count, and "how many calls did the
        harness make" is one of the questions the log exists to answer.
        """
        usage = usage or {}
        line = {
            # Full precision, not rounded: two concurrent streams can complete inside the
            # same millisecond, and a reader selecting by time would then drop one.
            "ts": time.time(),
            "model_id": model_id,
            "route": route,
            "translated": bool(translated),
        }
        for key in USAGE_FIELDS[4:]:
            value = usage.get(key)
            line[key] = int(value) if isinstance(value, (int, float)) else 0
        payload = json.dumps(line, sort_keys=False) + "\n"
        with self._lock:
            with self.path.open("a") as handle:
                handle.write(payload)


def read_usage_log(path: Union[str, Path], *, since: float = 0.0,
                   offset: int = 0) -> list:
    """Usage lines from *path*, optionally only the ones after a mark.

    :param path: The JSONL file written by :class:`UsageRecorder`.
    :param since: Only return calls whose ``ts`` is strictly greater than this epoch time.
    :param offset: Skip this many valid leading records. Applied after *since*.
    :type path: Union[str, pathlib.Path]
    :type since: float
    :type offset: int
    :rtype: list[dict]

    *offset* exists because it is the exact way to attribute usage to one unit of work:
    count the lines before dispatching, then take everything after that count. A
    time-based mark is approximate at best — the clock's resolution has to beat the rate
    at which a harness opens streams, and a subagent turn can overlap its parent's.

    Malformed lines are skipped rather than raising: the file is appended to by a live
    server, so a reader can legitimately catch a partial final line.
    """
    out = []
    file_path = Path(path)
    if not file_path.is_file():
        return out
    with file_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and float(record.get("ts") or 0.0) > since:
                out.append(record)
    return out[offset:]


def count_usage_lines(path: Union[str, Path]) -> int:
    """How many usage records *path* holds, for use as an :func:`read_usage_log` offset.

    :param path: The JSONL file written by :class:`UsageRecorder`.
    :type path: Union[str, pathlib.Path]
    :rtype: int
    """
    return len(read_usage_log(path))


def _client(region: Optional[str] = None):
    """A ``bedrock-runtime`` client. Imported lazily so importing this module is cheap."""
    import boto3

    resolved = (region or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    return boto3.client("bedrock-runtime", region_name=resolved)


def _forward_anthropic_stream(client, model_id: str, body: dict,
                              recorder: Optional[UsageRecorder] = None,
                              ) -> Iterator[bytes]:
    """Forward an Anthropic-model request to Bedrock and re-emit it as Anthropic SSE.

    The body needs no translation, but the *response* does: Bedrock replies in AWS
    binary event-stream framing, and the CLI has been told (by the content-type guard
    flag) to expect SSE. boto3 already decodes the framing for us, so this only has to
    re-frame each chunk — the payloads inside are already Anthropic events.

    Usage is accumulated across frames rather than read from one: Anthropic reports input
    tokens on ``message_start`` and output tokens on ``message_delta``, so either frame
    alone gives half the answer.
    """
    response = client.invoke_model_with_response_stream(
        modelId=model_id, body=json.dumps(body),
        contentType="application/json", accept="application/json",
    )
    seen: Dict[str, int] = {}
    for event in response.get("body", []):
        chunk = (event or {}).get("chunk") or {}
        raw = chunk.get("bytes")
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        _absorb_usage(seen, payload.get("usage"))
        _absorb_usage(seen, (payload.get("message") or {}).get("usage")
                      if isinstance(payload.get("message"), dict) else None)
        event_type = payload.get("type") or "message_delta"
        yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()
    if recorder is not None:
        recorder.record(model_id=model_id, route="stream", usage=seen, translated=False)


def _absorb_usage(into: Dict[str, int], usage: Any) -> None:
    """Merge an Anthropic usage dict into *into*, keeping the larger of each count.

    Larger rather than summed: a stream may restate a running total on several frames,
    and summing restatements would multiply the bill.
    """
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            into[key] = max(int(value), int(into.get(key, 0)))


def _converse_stream(client, model_id: str, body: dict,
                     recorder: Optional[UsageRecorder] = None) -> Iterator[bytes]:
    """Translate to Converse, stream it, and re-emit as Anthropic SSE."""
    kwargs = to_converse(body, model_id)
    response = client.converse_stream(**kwargs)
    events = (event for event in response.get("stream", []) if isinstance(event, dict))
    on_usage = None
    if recorder is not None:
        def on_usage(usage: dict) -> None:
            recorder.record(model_id=model_id, route="stream", usage=usage,
                            translated=True)
    return to_anthropic_sse(events, model=model_id, on_usage=on_usage)


def build_app(region: Optional[str] = None, client=None,
              usage_log: Union[str, Path, UsageRecorder, None] = None,
              framing: str = DEFAULT_FRAMING):
    """Build the FastAPI app.

    :param region: AWS region for the outbound leg; defaults to the environment.
    :param client: Injected ``bedrock-runtime`` client, for tests. When given, no boto3
        client is constructed and no credentials are needed.
    :param usage_log: Path to append per-call usage to, or a ready
        :class:`UsageRecorder`. ``None`` falls back to ``$CHIA_PROXY_USAGE_LOG``, and
        recording stays off when neither is set. See "Why it records usage".
    :param framing: ``"eventstream"`` (default) or ``"sse"``. See the module docstring:
        ``sse`` makes the CLI retry every turn non-streaming, doubling provider spend.
    :type region: Optional[str]
    :type usage_log: Union[str, pathlib.Path, UsageRecorder, None]
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
    if usage_log is None:
        usage_log = os.environ.get(USAGE_LOG_ENV) or None
    app.state.usage = (usage_log if isinstance(usage_log, UsageRecorder)
                       else UsageRecorder(usage_log) if usage_log else None)
    if framing not in FRAMINGS:
        raise ValueError(f"framing must be one of {FRAMINGS}, got {framing!r}")
    app.state.framing = framing

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
        recorder = app.state.usage
        try:
            if streaming:
                frames = (
                    _forward_anthropic_stream(client_obj, model_id, body, recorder)
                    if is_anthropic_model(model_id)
                    else _converse_stream(client_obj, model_id, body, recorder)
                )
                if app.state.framing == "eventstream":
                    return StreamingResponse(frames_from_sse(frames),
                                             media_type=EVENTSTREAM_CONTENT_TYPE)
                return StreamingResponse(frames, media_type="text/event-stream")
            if is_anthropic_model(model_id):
                response = client_obj.invoke_model(
                    modelId=model_id, body=json.dumps(body),
                    contentType="application/json", accept="application/json",
                )
                payload = json.loads(response["body"].read())
                if recorder is not None:
                    recorder.record(model_id=model_id, route="invoke",
                                    usage=payload.get("usage"), translated=False)
                return JSONResponse(payload)
            converse = client_obj.converse(**to_converse(body, model_id))
            message = to_anthropic_message(converse, model=model_id)
            if recorder is not None:
                recorder.record(model_id=model_id, route="invoke",
                                usage=message.get("usage"), translated=True)
            return JSONResponse(message)
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
    parser.add_argument("--framing", default=DEFAULT_FRAMING, choices=list(FRAMINGS),
                        help="response framing for the streaming route. The default is "
                             "what the CLI's Bedrock transport asks for; 'sse' requires "
                             "CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD=1 on the "
                             "client AND makes it retry every turn non-streaming, which "
                             "doubles provider spend.")
    parser.add_argument("--usage-log", default=None, type=Path,
                        help="append one JSON line per call with the real token counts. "
                             "For a proxied non-Anthropic model this is the only "
                             "trustworthy cost source; the client prices what it thinks "
                             "it called. Defaults to $" + USAGE_LOG_ENV + ".")
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

    uvicorn.run(build_app(region=args.region, usage_log=args.usage_log,
                          framing=args.framing),
                host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
