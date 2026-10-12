"""Offline checks for the controlled prompt-cache gap protocol."""

import asyncio
import importlib.util
import json
from pathlib import Path


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
