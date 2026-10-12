"""Controlled Anthropic prompt-cache gap study.

``plan`` and ``summarize`` make no model request. ``run`` makes paid requests
only when invoked explicitly with credentials, model prices and a dollar cap.
The JSONL log stores response usage and timing, never prompts or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import time
import uuid
from pathlib import Path

import httpx


GAPS = {"short": 30, "past_5m": 360, "simulation": 3900}


def schedule(seed: int, repetitions: int, *, run_id: str) -> list[dict]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    arms = [
        {"arm_id": f"{run_id}-{gap}-{rep}", "gap": gap,
         "gap_seconds": seconds, "replicate": rep}
        for gap, seconds in GAPS.items() for rep in range(1, repetitions + 1)
    ]
    random.Random(seed).shuffle(arms)
    return arms


def cached_prefix(arm_id: str, lines: int) -> str:
    """Unique per arm, exactly stable across its three calls."""
    if lines < 1:
        raise ValueError("lines must be positive")
    return (f"Cache study arm {arm_id}. This prefix is held exactly constant.\n"
            + "".join(f"Record {i:05d}: blue stone, green branch, measured signal.\n"
                      for i in range(lines)))


def request_body(model: str, prefix: str) -> dict:
    return {
        "model": model,
        "max_tokens": 0,
        "system": [{"type": "text", "text": prefix,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Check this fixed prefix."}],
    }


def usage_from_response(response: dict) -> dict:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("response has no usage block")
    out = {}
    for field in ("input_tokens", "cache_read_input_tokens",
                  "cache_creation_input_tokens", "output_tokens"):
        value = usage.get(field, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid usage field: {field}")
        out[field] = value
    return out


def cost_upper_bound(body: dict, prices: dict) -> float:
    """Conservative preflight: bytes plus framing headroom per request.

    Tokens encode nonempty byte sequences. The extra 4096 tokens covers API
    framing for this fixed, tool-free, zero-output request. The caller checks
    this bound for all planned calls before making any of them.
    """
    input_bound = len(json.dumps(body, ensure_ascii=False).encode()) + 4096
    return input_bound * max(prices[kind] for kind in ("input", "read", "write")) / 1_000_000


def measured_cost(usage: dict, prices: dict) -> float:
    return sum(usage[field] * prices[price] / 1_000_000 for field, price in (
        ("input_tokens", "input"),
        ("cache_read_input_tokens", "read"),
        ("cache_creation_input_tokens", "write"),
        ("output_tokens", "output"),
    ))


async def run_arm(arm: dict, *, model: str, lines: int, call, sleep,
                  now, write_event) -> dict:
    prefix = cached_prefix(arm["arm_id"], lines)
    body = request_body(model, prefix)
    prefix_hash = hashlib.sha256(prefix.encode()).hexdigest()
    events = []
    for phase in ("warm", "validate", "delayed"):
        if phase == "delayed":
            target = events[1]["start_monotonic"] + arm["gap_seconds"]
            await sleep(max(0, target - now()))
        event = await call(body)
        event = {"schema_version": 1, "arm_id": arm["arm_id"],
                 "gap": arm["gap"], "gap_seconds": arm["gap_seconds"],
                 "replicate": arm["replicate"], "phase": phase,
                 "model": model, "declared_ttl_seconds": 300,
                 "prefix_sha256": prefix_hash, **event}
        if phase == "delayed":
            event["actual_gap_seconds"] = (
                event["start_monotonic"] - events[1]["start_monotonic"])
        await write_event(event)
        events.append(event)
        if event.get("error"):
            break
        if phase == "warm" and event["usage"]["cache_creation_input_tokens"] == 0:
            break
        if phase == "validate" and event["usage"]["cache_read_input_tokens"] == 0:
            break
    return {"arm_id": arm["arm_id"], "completed_phases": len(events)}


async def run_study(args) -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is required for run")
    prices = {key: getattr(args, f"price_{key}")
              for key in ("input", "read", "write", "output")}
    if any(value < 0 for value in prices.values()) or args.cap_usd <= 0:
        raise ValueError("prices must be nonnegative and cap positive")
    run_id = uuid.uuid4().hex[:12]
    arms = schedule(args.seed, args.repetitions, run_id=run_id)
    bodies = [request_body(args.model, cached_prefix(arm["arm_id"], args.lines))
              for arm in arms]
    maximum = 3 * sum(cost_upper_bound(body, prices) for body in bodies)
    if maximum > args.cap_usd:
        raise ValueError(f"preflight upper bound ${maximum:.2f} exceeds cap "
                         f"${args.cap_usd:.2f}; reduce arms/prefix or raise cap")
    output = Path(args.output)
    if not output.parent.is_dir() or output.exists():
        raise ValueError("output parent must exist and output must be new")
    print(json.dumps({"run_id": run_id, "arms": len(arms),
                      "requests_at_most": 3 * len(arms),
                      "preflight_upper_usd": round(maximum, 4),
                      "output": str(output)}))
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    endpoint = args.base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    timeout = httpx.Timeout(connect=15, read=600, write=30, pool=15)

    async def write_event(event: dict) -> None:
        async with lock:
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, separators=(",", ":")) + "\n")

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def call(body: dict) -> dict:
            async with semaphore:
                start_monotonic = time.monotonic()
                start_ts = time.time()
                try:
                    response = await client.post(endpoint, json=body, headers=headers)
                    response.raise_for_status()
                    payload = response.json()
                    usage = usage_from_response(payload)
                    return {"start_monotonic": start_monotonic,
                            "start_ts": start_ts, "end_ts": time.time(),
                            "response_id": payload.get("id"),
                            "request_id": response.headers.get("request-id"),
                            "status": response.status_code,
                            "usage": usage,
                            "cost_usd": measured_cost(usage, prices)}
                except (httpx.HTTPError, ValueError) as error:
                    return {"start_monotonic": start_monotonic,
                            "start_ts": start_ts, "end_ts": time.time(),
                            "error": type(error).__name__}

        await asyncio.gather(*(run_arm(arm, model=args.model, lines=args.lines,
                                      call=call, sleep=asyncio.sleep,
                                      now=time.monotonic, write_event=write_event)
                               for arm in arms))
    print(json.dumps(summarize(output)))


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["arm_id"], {})[row["phase"]] = row
    by_gap = {}
    for arm in grouped.values():
        delayed = arm.get("delayed")
        if not delayed or delayed.get("error"):
            continue
        warm = arm["warm"]
        validate = arm["validate"]
        if (warm["usage"]["cache_creation_input_tokens"] == 0 or
                validate["usage"]["cache_read_input_tokens"] == 0):
            continue
        target = GAPS[delayed["gap"]]
        if abs(delayed["actual_gap_seconds"] - target) > max(5, target * 0.05):
            continue
        by_gap.setdefault(delayed["gap"], []).append(delayed)
    return {"requests": len(rows), "arms": len(grouped),
            "measured_cost_usd": round(sum(row.get("cost_usd", 0) for row in rows), 6),
            "gaps": {gap: {
                "valid_arms": len(items),
                "cache_read_arms": sum(item["usage"]["cache_read_input_tokens"] > 0
                                       for item in items),
                "median_cache_read_tokens": statistics.median(
                    item["usage"]["cache_read_input_tokens"] for item in items),
                "median_fresh_input_tokens": statistics.median(
                    item["usage"]["input_tokens"] for item in items),
                "median_cache_write_tokens": statistics.median(
                    item["usage"]["cache_creation_input_tokens"] for item in items),
                "median_delayed_request_cost_usd": statistics.median(
                    item["cost_usd"] for item in items),
                "median_actual_gap_seconds": statistics.median(
                    item["actual_gap_seconds"] for item in items),
            } for gap, items in sorted(by_gap.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="show the fixed gap schedule; no API calls")
    plan.add_argument("--seed", type=int, default=20261012)
    plan.add_argument("--repetitions", type=int, default=3)
    run = sub.add_parser("run", help="make paid API calls with an explicit cap")
    run.add_argument("--seed", type=int, default=20261012)
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--model", required=True)
    run.add_argument("--lines", type=int, default=3000)
    run.add_argument("--concurrency", type=int, default=3)
    run.add_argument("--base-url", default="https://api.anthropic.com")
    run.add_argument("--output", required=True)
    run.add_argument("--cap-usd", required=True, type=float)
    for kind in ("input", "read", "write", "output"):
        run.add_argument(f"--price-{kind}", required=True, type=float,
                         help="USD per million tokens; verify current model price")
    summary = sub.add_parser("summarize", help="summarize saved usage; no API calls")
    summary.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(schedule(args.seed, args.repetitions,
                                  run_id="UNIQUE-RUN-ID"), indent=2))
    elif args.command == "run":
        if args.lines < 1 or args.concurrency < 1:
            parser.error("lines and concurrency must be positive")
        asyncio.run(run_study(args))
    else:
        print(json.dumps(summarize(args.path), indent=2))


if __name__ == "__main__":
    main()
