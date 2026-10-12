"""Opt-in installed-CLI smoke for the sandboxed, budgeted study runner."""

import hashlib
import json
import os
import shutil
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from examples.harness_study.delegation_benchmark import benchmark, run_arms


pytestmark = pytest.mark.skipif(
    os.environ.get("CHIA_RUN_ISOLATED_ARM_FIXTURE") != "1"
    or shutil.which("claude") is None or shutil.which("opencode") is None,
    reason="opt-in local CLI fixture; no paid provider calls",
)


@pytest.mark.parametrize("harness", ["claude", "opencode"])
def test_one_isolated_arm_reaches_relay_and_grades(harness, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_TOKEN", "fixture-only")
    upstream = FastAPI()

    @upstream.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @upstream.post("/v1/chat/completions")
    async def completion(request: Request):
        body = await request.json()
        model = body["model"]
        tools = [item.get("function", {}) for item in body.get("tools") or []]
        delegate = next((item for item in tools
                         if item.get("name", "").lower() in ("task", "agent")), None)
        has_tool_result = any(item.get("role") == "tool"
                              for item in body.get("messages") or [])
        if model == "weak":
            message = {"role": "assistant", "content": "EVIDENCE: usage is folded in claude.py."}
            finish_reason = "stop"
        elif delegate is None:
            message = {"role": "assistant", "content": "Fixture title"}
            finish_reason = "stop"
        elif has_tool_result:
            message = {"role": "assistant", "content": "FINAL: delegated evidence received"}
            finish_reason = "stop"
        else:
            properties = (delegate.get("parameters") or {}).get("properties") or {}
            required = (delegate.get("parameters") or {}).get("required") or []
            inputs = {}
            for key in required:
                if key == "description":
                    inputs[key] = "Inspect token usage"
                elif key in ("prompt", "task"):
                    inputs[key] = "Locate where Claude Code usage is folded."
                elif key in ("subagent_type", "agent", "agent_name", "name"):
                    inputs[key] = "explorer"
                else:
                    kind = (properties.get(key) or {}).get("type")
                    inputs[key] = [] if kind == "array" else False if kind == "boolean" else "explorer"
            if "subagent_type" in properties:
                inputs["subagent_type"] = "explorer"
            if "run_in_background" in properties:
                inputs["run_in_background"] = False
            message = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "call_delegate", "type": "function",
                                       "function": {"name": delegate["name"],
                                                    "arguments": json.dumps(inputs)}}]}
            finish_reason = "tool_calls"
        usage = {"prompt_tokens": 10, "completion_tokens": 3,
                 "total_tokens": 13}
        if body.get("stream"):
            async def chunks():
                delta = {"role": "assistant"}
                if message.get("tool_calls"):
                    delta["tool_calls"] = [
                        {"index": 0, **message["tool_calls"][0]}]
                else:
                    delta["content"] = message["content"]
                for part, finish in ((delta, None), ({}, finish_reason)):
                    row = {"id": "chatcmpl-local", "object": "chat.completion.chunk",
                           "created": int(time.time()), "model": model,
                           "choices": [{"index": 0, "delta": part,
                                        "finish_reason": finish}]}
                    if finish:
                        row["usage"] = usage
                    yield ("data: " + json.dumps(row) + "\n\n").encode()
                yield b"data: [DONE]\n\n"
            return StreamingResponse(chunks(), media_type="text/event-stream")
        return JSONResponse({
            "id": "chatcmpl-local", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish_reason}],
            "usage": usage,
        })

    task = benchmark.TASKS["claude_usage"]
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps({
        "cap_usd": 1.0,
        "arms": [{"task": task["id"], "repeat": 0, "harness": harness,
                  "baseline": task["baseline"],
                  "oracle_sha256": hashlib.sha256(
                      benchmark.oracle_path(task).read_bytes()).hexdigest()}],
    }))
    prices_path = tmp_path / "prices.json"
    prices_path.write_text(json.dumps({
        name: {"input": 0.001, "read": 0.001,
               "write": 0.001, "output": 0.001}
        for name in ("strong", "weak")
    }))
    with run_arms.serve(upstream) as upstream_port:
        args = SimpleNamespace(
            credential_env="TEST_PROVIDER_TOKEN", schedule=schedule_path,
            cap_usd=1.0, prices_json=prices_path, strong="strong", weak="weak",
            max_arms=1, upstream_base_url=f"http://127.0.0.1:{upstream_port}/v1",
            output_limit=1024, timeout=45,
        )
        result_path = run_arms.execute(args)
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["provider_requests"] >= 1
    assert rows[0]["full_usage"] is True
    assert rows[0]["grade_pass"] is False
    assert rows[0]["weak_model_requests"] >= 1
    assert rows[0]["chia_usage"]["billing_mode"] == "per_token"
    expected_residual = 0 if harness == "claude" else 10
    assert rows[0]["provider_minus_chia_tokens"]["input_tokens"] == expected_residual
    assert rows[0]["profile_request_count"] == (1 if harness == "claude" else 3)
    events = [json.loads(line) for line in open(rows[0]["profile_log"])]
    profile = [event for event in events if event.get("schema") == "chia.agent_profile"]
    if harness == "claude":
        assert any(event["type"] == "agent_start" and
                   event.get("parent_agent_id") == "lead" for event in profile)
        assert any(event["type"] == "llm_request" and
                   event.get("model") == "mixed" for event in profile)
    else:
        assert any(event["type"] == "llm_request" and
                   event.get("model") == "study/weak" and
                   event.get("parent_agent_id", "").startswith("opencode:")
                   for event in profile)
