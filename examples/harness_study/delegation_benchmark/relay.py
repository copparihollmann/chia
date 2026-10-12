"""Privacy-safe OpenAI-compatible relay for matched harness accounting.

Both CLI harnesses must point at a separate instance of this loopback relay
for each arm. Every upstream Chat Completions request, including OpenCode's
automatic title request, gets one JSONL record with model, times, status and
provider-reported token classes. Prompts, responses and credentials are never
written to the log. A missing usage block is recorded as incomplete; such an
arm cannot support a full cost comparison.

The relay makes no provider request until a client posts to its endpoint.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


def _usage(payload: Any) -> dict:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {"usage_complete": False}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None or completion is None:
        return {"usage_complete": False}
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    try:
        cached = int(details.get("cached_tokens") or 0)
        prompt = int(prompt)
        completion = int(completion)
        written = int(usage.get("cache_creation_input_tokens") or
                      usage.get("cache_write_tokens") or 0)
    except (TypeError, ValueError):
        return {"usage_complete": False}
    if min(cached, prompt, completion, written) < 0 or cached + written > prompt:
        return {"usage_complete": False}
    return {
        "usage_complete": True,
        "prompt_tokens": prompt,
        "input_tokens": prompt - cached - written,
        "output_tokens": completion,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": written,
    }


def _stream_usage(buffer: bytearray, chunk: bytes) -> dict | None:
    """Read complete SSE frames from *chunk* and retain the latest usage block."""
    buffer.extend(chunk)
    buffer[:] = buffer.replace(b"\r\n", b"\n")
    found = None
    while b"\n\n" in buffer:
        frame, _, remainder = buffer.partition(b"\n\n")
        buffer[:] = remainder
        data = b"\n".join(line[5:].strip() for line in frame.splitlines()
                          if line.startswith(b"data:"))
        if not data or data == b"[DONE]":
            continue
        try:
            payload = json.loads(data)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            found = _usage(payload)
    return found


def _price_usage(usage: dict, rates: Mapping[str, float]) -> float:
    return sum(usage[field] * rates[kind] / 1_000_000 for field, kind in (
        ("input_tokens", "input"),
        ("cache_read_input_tokens", "read"),
        ("cache_creation_input_tokens", "write"),
        ("output_tokens", "output"),
    ))


class _Budget:
    def __init__(self, *, cap_usd: float, prices: Mapping[str, Mapping[str, float]],
                 output_limit: int):
        if not math.isfinite(cap_usd) or cap_usd <= 0 or output_limit <= 0:
            raise ValueError("cap_usd and output_limit must be positive")
        for model, rates in prices.items():
            if any(kind not in rates or not isinstance(rates[kind], (int, float))
                   or not math.isfinite(rates[kind]) or rates[kind] < 0 for kind in
                   ("input", "read", "write", "output")):
                raise ValueError(f"model {model!r} needs four nonnegative prices")
        self.cap_usd = cap_usd
        self.prices = prices
        self.output_limit = output_limit
        self.spent = 0.0
        self.reserved = 0.0
        self.lock = asyncio.Lock()

    async def reserve(self, body: dict) -> float | None:
        rates = self.prices.get(body["model"])
        if rates is None:
            raise ValueError("model has no price entry")
        field = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        declared = body.get(field)
        if declared is not None and (type(declared) is not int or declared < 0):
            raise ValueError("invalid output-token limit")
        if declared is None or declared > self.output_limit:
            body[field] = self.output_limit
        # In this text-only study, each input token consumes at least one UTF-8
        # byte. The extra 8192 covers chat-format framing and hidden system text.
        input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 8192
        bound = (input_bound * max(rates[kind] for kind in
                 ("input", "read", "write")) +
                 self.output_limit * rates["output"]) / 1_000_000
        async with self.lock:
            if self.spent + self.reserved + bound > self.cap_usd:
                return None
            self.reserved += bound
        return bound

    async def settle(self, record: dict, reservation: float) -> None:
        usage_complete = record.get("usage_complete") is True
        actual = _price_usage(record, self.prices[record["model"]]) if usage_complete else None
        debit = actual if actual is not None else reservation
        async with self.lock:
            self.reserved -= reservation
            self.spent += debit
        record["cost_usd"] = actual
        record["cost_source"] = ("computed_from_provider_usage_and_supplied_rates"
                                 if actual is not None else "unavailable")
        record["budget_debit_usd"] = debit


def build_relay_app(*, upstream_base_url: str, credential_env: str,
                    usage_log: Path, arm_id: str, transport=None,
                    prices: Mapping[str, Mapping[str, float]] | None = None,
                    cap_usd: float | None = None,
                    output_limit: int = 4096) -> FastAPI:
    """Build a loopback relay; caller owns listener and its provider credentials."""
    if not upstream_base_url.startswith(("http://", "https://")):
        raise ValueError("upstream_base_url must be HTTP(S)")
    usage_log = Path(usage_log)
    if not usage_log.parent.is_dir():
        raise ValueError("usage log parent directory must already exist")
    app = FastAPI(title="Chia study usage relay", docs_url=None, redoc_url=None)
    write_lock = asyncio.Lock()
    if (prices is None) != (cap_usd is None):
        raise ValueError("prices and cap_usd must be provided together")
    budget = (_Budget(cap_usd=cap_usd, prices=prices, output_limit=output_limit)
              if cap_usd is not None else None)
    endpoint = upstream_base_url.rstrip("/") + "/chat/completions"
    timeout = httpx.Timeout(connect=15, read=600, write=30, pool=15)

    async def write_record(record: dict, reservation: float = 0.0) -> None:
        if budget is not None:
            await budget.settle(record, reservation)
        async with write_lock:
            with usage_log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, separators=(",", ":")) + "\n")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        key = os.environ.get(credential_env)
        if not key:
            return JSONResponse({"error": "provider credential is not configured"}, status_code=503)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            return JSONResponse({"error": "model is required"}, status_code=400)
        body = dict(body)
        streaming = bool(body.get("stream"))
        if streaming:
            options = body.get("stream_options") or {}
            if not isinstance(options, dict):
                options = {}
            body["stream_options"] = {**options, "include_usage": True}
        reservation = 0.0
        if budget is not None:
            try:
                reserved = await budget.reserve(body)
            except ValueError as error:
                return JSONResponse({"error": str(error)}, status_code=400)
            if reserved is None:
                return JSONResponse({"error": "study dollar cap reached"}, status_code=429)
            reservation = reserved
        record = {
            "request_id": uuid.uuid4().hex, "arm_id": arm_id,
            "model": body["model"], "stream": streaming,
            "start_ts": time.time(),
        }
        headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        if not streaming:
            try:
                async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
                    response = await client.post(endpoint, json=body, headers=headers)
                record.update({"end_ts": time.time(), "status": response.status_code})
                record["upstream_request_id"] = (
                    response.headers.get("x-request-id") or response.headers.get("request-id"))
                if response.is_success:
                    try:
                        payload = response.json()
                        record.update(_usage(payload))
                        record["provider_response_id"] = payload.get("id")
                    except ValueError:
                        record["usage_complete"] = False
                else:
                    record["usage_complete"] = False
                await write_record(record, reservation)
                return Response(response.content, status_code=response.status_code,
                                media_type=response.headers.get("content-type", "application/json"))
            except httpx.HTTPError:
                record.update({"end_ts": time.time(), "status": 502, "usage_complete": False})
                await write_record(record, reservation)
                return JSONResponse({"error": "upstream request failed"}, status_code=502)

        client = httpx.AsyncClient(transport=transport, timeout=timeout)
        outbound = client.build_request("POST", endpoint, json=body, headers=headers)
        try:
            response = await client.send(outbound, stream=True)
        except httpx.HTTPError:
            await client.aclose()
            record.update({"end_ts": time.time(), "status": 502, "usage_complete": False})
            await write_record(record, reservation)
            return JSONResponse({"error": "upstream request failed"}, status_code=502)
        record["upstream_request_id"] = (
            response.headers.get("x-request-id") or response.headers.get("request-id"))
        if not response.is_success:
            content = await response.aread()
            record.update({"end_ts": time.time(), "status": response.status_code,
                           "usage_complete": False})
            await write_record(record, reservation)
            await response.aclose()
            await client.aclose()
            return Response(content, status_code=response.status_code,
                            media_type=response.headers.get("content-type", "application/json"))

        async def relay_stream():
            buffer = bytearray()
            found = None
            try:
                async for chunk in response.aiter_bytes():
                    parsed = _stream_usage(buffer, chunk)
                    if parsed is not None:
                        found = parsed
                    yield chunk
            finally:
                record.update({"end_ts": time.time(), "status": response.status_code})
                record.update(found or {"usage_complete": False})
                await write_record(record, reservation)
                await response.aclose()
                await client.aclose()

        return StreamingResponse(relay_stream(), media_type="text/event-stream")

    return app
