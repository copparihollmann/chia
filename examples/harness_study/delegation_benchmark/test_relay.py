import asyncio
import json

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from examples.harness_study.delegation_benchmark.relay import build_relay_app


def test_relay_counts_title_main_and_streamed_child_without_logging_content(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_TOKEN", "private-provider-key")
    upstream = FastAPI()
    seen = []

    @upstream.post("/v1/chat/completions")
    async def completion(request: Request):
        body = await request.json()
        assert request.headers["authorization"] == "Bearer private-provider-key"
        seen.append(body)
        if body.get("stream"):
            assert body["stream_options"]["include_usage"] is True

            async def chunks():
                yield b'data: {"choices":[{"delta":{"content":"secret reply"}}]}\n\n'
                yield b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":5}}}\n\n'
                yield b'data: [DONE]\n\n'

            return StreamingResponse(chunks(), media_type="text/event-stream")
        return JSONResponse({
            "choices": [{"message": {"content": "secret reply"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })

    log = tmp_path / "provider_usage.jsonl"
    relay = build_relay_app(
        upstream_base_url="http://upstream/v1", credential_env="TEST_PROVIDER_TOKEN",
        usage_log=log, arm_id="arm-one", transport=httpx.ASGITransport(upstream),
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(relay),
                                     base_url="http://relay") as client:
            for purpose in ("title", "main"):
                response = await client.post("/v1/chat/completions", json={
                    "model": "strong", "messages": [{"role": "user", "content": purpose}],
                })
                assert response.status_code == 200
            streamed = await client.post("/v1/chat/completions", json={
                "model": "weak", "stream": True,
                "messages": [{"role": "user", "content": "private prompt"}],
            })
            assert streamed.status_code == 200
            assert "secret reply" in streamed.text

    asyncio.run(exercise())
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(seen) == len(records) == 3
    assert [record["model"] for record in records] == ["strong", "strong", "weak"]
    assert all(record["usage_complete"] for record in records)
    assert sum(record["input_tokens"] for record in records) == 27
    assert sum(record["output_tokens"] for record in records) == 7
    assert records[2]["cache_read_input_tokens"] == 5
    assert all(record["start_ts"] <= record["end_ts"] for record in records)
    assert "private" not in log.read_text()
    assert "secret reply" not in log.read_text()


def test_relay_marks_missing_provider_usage_incomplete(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_TOKEN", "private-provider-key")
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def completion(_request: Request):
        return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

    log = tmp_path / "provider_usage.jsonl"
    relay = build_relay_app(
        upstream_base_url="http://upstream/v1", credential_env="TEST_PROVIDER_TOKEN",
        usage_log=log, arm_id="arm-two", transport=httpx.ASGITransport(upstream),
    )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(relay),
                                     base_url="http://relay") as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "strong", "messages": [],
            })
            assert response.status_code == 200

    asyncio.run(exercise())
    record = json.loads(log.read_text())
    assert record["usage_complete"] is False
