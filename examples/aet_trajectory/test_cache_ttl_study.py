"""Offline checks for the controlled prompt-cache gap protocol."""

import asyncio
import importlib.util
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "cache_ttl_study", Path(__file__).with_name("cache_ttl_study.py"))
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_schedule_separates_concurrent_cache_prefixes():
    arms = study.schedule(20261012, 3, run_id="fixed-run")
    assert len(arms) == 9
    assert {arm["gap"] for arm in arms} == set(study.GAPS)
    prefixes = [study.cached_prefix(arm["arm_id"], 4) for arm in arms]
    assert len(set(prefixes)) == len(prefixes)
    assert all(study.request_body("model", prefix)["max_tokens"] == 0
               for prefix in prefixes)
    assert all(study.request_body("model", prefix)["system"][0]["cache_control"]
               == {"type": "ephemeral"} for prefix in prefixes)


def test_gap_starts_at_confirmed_read_request_and_summary_keeps_only_valid(tmp_path):
    clock = [100.0]
    calls = []
    events = []
    arm = {"arm_id": "unique-past_5m-1", "gap": "past_5m",
           "gap_seconds": 360, "replicate": 1}

    async def call(body):
        calls.append(body)
        index = len(calls)
        usage = {"input_tokens": 10, "output_tokens": 0,
                 "cache_creation_input_tokens": 5000 if index != 2 else 0,
                 "cache_read_input_tokens": 5000 if index == 2 else 0}
        start = clock[0]
        clock[0] += 2
        return {"start_monotonic": start, "start_ts": start,
                "end_ts": clock[0], "usage": usage,
                "cost_usd": 0.01}

    async def sleep(seconds):
        clock[0] += seconds

    async def write_event(event):
        events.append(event)

    asyncio.run(study.run_arm(arm, model="model", lines=4, call=call,
                              sleep=sleep, now=lambda: clock[0],
                              write_event=write_event))
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert events[-1]["actual_gap_seconds"] == 360
    assert len({event["prefix_sha256"] for event in events}) == 1
    assert "system" not in events[-1] and "text" not in events[-1]
    log = tmp_path / "usage.jsonl"
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = study.summarize(log)
    assert result["gaps"]["past_5m"]["valid_arms"] == 1
    assert result["gaps"]["past_5m"]["cache_read_arms"] == 0
    assert result["measured_cost_usd"] == 0.03


def test_paid_run_rejects_nonfinite_cap_before_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-only")
    args = SimpleNamespace(price_input=1, price_read=0.1,
                           price_write=1.25, price_output=2,
                           cap_usd=math.nan)
    with pytest.raises(ValueError):
        asyncio.run(study.run_study(args))


def test_full_http_study_uses_independent_prefixes_and_logs_only_usage(
        tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-only")
    monkeypatch.setattr(study, "GAPS", {"short": 0.05, "past_5m": 0.08,
                                        "simulation": 0.12})
    calls = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            assert self.path == "/v1/messages"
            assert self.headers["x-api-key"] == "fixture-only"
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            assert body["max_tokens"] == 0
            prefix = body["system"][0]["text"]
            assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
            with lock:
                calls[prefix] = calls.get(prefix, 0) + 1
                number = calls[prefix]
            if number == 1 or (number == 3 and "-short-" not in prefix):
                read, write = 0, 5000
            else:
                read, write = 5000, 0
            response = json.dumps({
                "id": f"msg_{number}",
                "usage": {"input_tokens": 10, "output_tokens": 0,
                          "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("request-id", f"req_{number}")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "usage.jsonl"
    args = SimpleNamespace(
        seed=20261012, repetitions=1, model="fixture-model", lines=4,
        cap_usd=1.0, concurrency=3,
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        output=output, price_input=0.001, price_read=0.001,
        price_write=0.001, price_output=0.001,
    )
    try:
        asyncio.run(study.run_study(args))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert len(calls) == 3 and set(calls.values()) == {3}
    result = study.summarize(output)
    assert result["requests"] == 9 and result["arms"] == 3
    assert result["gaps"]["short"]["cache_read_arms"] == 1
    assert result["gaps"]["past_5m"]["cache_read_arms"] == 0
    assert result["gaps"]["simulation"]["cache_read_arms"] == 0
    logged = output.read_text()
    assert "fixture-only" not in logged
    assert "Cache study arm" not in logged
    assert len({json.loads(row)["prefix_sha256"] for row in logged.splitlines()}) == 3
