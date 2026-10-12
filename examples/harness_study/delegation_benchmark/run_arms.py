"""Run a pre-registered matched delegation schedule through isolated CLI arms.

This makes paid model requests only with the explicit ``run`` command, an
upstream credential, a current model price table, and a dollar cap. Each arm
uses a fresh archived source snapshot, a private CLI home, a bwrap network
namespace, and one Unix-socket bridge to a loopback usage relay. The oracle is
run outside the agent namespace after the CLI exits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from urllib import request as urllib_request
from urllib.parse import urlsplit

import uvicorn
import ray

from chia.models.agents import AgentDefinition, ModelRef, ProviderSpec
from chia.models.claude import ClaudeCodeLLM
from chia.models.opencode import OpenCodeLLM
from chia.models.proxy.gateway import (
    OpenAICompatibleBackend, ProviderRouter, build_gateway_app,
)
from chia.trace.profile_events import ProfileContext
from chia.trace.profiler import get_profiler, start_collector, stop_collector

if __package__:
    from . import benchmark, isolation, relay
else:
    import benchmark
    import isolation
    import relay


@contextmanager
def serve(app):
    """Run one loopback ASGI server on a reserved ephemeral port."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            try:
                urllib_request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                       timeout=0.2).read()
                break
            except Exception:
                time.sleep(0.02)
        else:
            raise RuntimeError("loopback study server did not start")
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


@contextmanager
def profile_arm(arm_dir: Path):
    """Record the actual Chia local-call trace for this arm."""
    fd = os.open(benchmark.task_temp(), os.O_RDONLY)
    short = f"/proc/{os.getpid()}/fd/{fd}/r{uuid.uuid4().hex[:6]}"
    Path(short).mkdir()
    log_dir = arm_dir / "profile"
    log_dir.mkdir()
    started = False
    try:
        ray.init(_temp_dir=short, include_dashboard=False, num_cpus=2,
                 logging_level="ERROR")
        started = True
        start_collector(log_dir=str(log_dir))
        get_profiler.reset()
        get_profiler().set_profile_context(
            ProfileContext(run_id=arm_dir.name, agent_id="lead"))
        yield log_dir / "ChiaProfileCollector.log"
    finally:
        if started:
            stop_collector()
            ray.shutdown()
        os.close(fd)


def agent_definitions(strong: str, weak: str) -> list[AgentDefinition]:
    return [
        AgentDefinition("lead", "Solve the repository task",
                        "Delegate the requested repository lookup to explorer before "
                        "editing. Work only in this source directory.",
                        role="primary", model=ModelRef("study", strong)),
        AgentDefinition("explorer", "Locate code and return evidence",
                        "Read the relevant files and report concise evidence to lead. "
                        "Do not edit files.", model=ModelRef("study", weak)),
    ]


def cli_env(workspace: Path, local_port: int) -> dict[str, str]:
    home = workspace / ".study_home"
    home.mkdir()
    for folder in ("tmp", "data", "config", "cache", "claude"):
        (home / folder).mkdir()
    return {
        "HOME": str(home), "TMPDIR": str(home / "tmp"),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "CLAUDE_CONFIG_DIR": str(home / "claude"),
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{local_port}",
        "ANTHROPIC_AUTH_TOKEN": "local-relay-only",
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_USE_GATEWAY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "RELAY_CLIENT_KEY": "local-relay-only",
    }


def run_cli(*, harness: str, prompt: str, workspace: Path,
            strong: str, weak: str, host_port: int, timeout: int,
            output_limit: int) -> tuple[object, dict]:
    program = "claude" if harness == "claude" else "opencode"
    local_port = 18543
    agents = agent_definitions(strong, weak)
    environment = cli_env(workspace, local_port)
    with isolation.RelayBridge(task_tmp=benchmark.task_temp(),
                               host_port=host_port) as bridge:
        spec = isolation.arm_spec(
            workspace, bridge.socket_path, program,
            preserve_env=("ANTHROPIC_AUTH_TOKEN", "RELAY_CLIENT_KEY"),
        )
        if harness == "claude":
            llm = ClaudeCodeLLM(
                model=f"study/{strong}", agents=agents, primary_agent="lead",
                timeout_seconds=timeout, retries=1, log_stream=True,
                sandbox_spec=spec, sandbox_backend="bwrap",
                extra_env=environment,
            )
        else:
            provider = ProviderSpec(
                "study", "openai-compatible",
                {name: {"limit": {"context": 131072, "output": output_limit}}
                 for name in (strong, weak)},
                base_url=f"http://127.0.0.1:{local_port}/v1",
                credential_env="RELAY_CLIENT_KEY",
            )
            llm = OpenCodeLLM(
                model=f"study/{strong}", agents=agents, primary_agent="lead",
                providers=[provider], work_dir=str(workspace),
                timeout_seconds=timeout, retries=1,
                sandbox_spec=spec, sandbox_backend="bwrap",
                extra_env=environment,
            )
        llm.sandbox_argv = lambda argv: isolation.isolated_argv(
            argv, spec, bridge.socket_path, local_port)
        # The CLI and its local state use only paths visible in the namespace.
        started = time.monotonic()
        result = llm.prompt(prompt, [])
        wall = time.monotonic() - started
    return result, {"wall_seconds": wall, "metadata": llm._last_metadata,
                    "public_usage": asdict(result.usage)}


def snapshot_diff(baseline: str, workspace: Path, task_id: str) -> str:
    clean = benchmark.snapshot(baseline, f"{task_id}-clean")
    cmd = ["diff", "-ruN", "--exclude=.study_home", "--exclude=__pycache__",
           "--exclude=.pytest_cache", str(clean), str(workspace)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"snapshot diff failed: {result.stderr[-500:]}")
    return result.stdout


def usage_summary(path: Path, weak: str) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    token_fields = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                    "cache_creation_input_tokens")
    return {
        "provider_requests": len(records),
        "weak_model_requests": sum(row["model"] == weak for row in records),
        "full_usage": bool(records) and all(row.get("usage_complete") for row in records),
        "cost_usd": sum(row.get("cost_usd") or 0 for row in records),
        "budget_debit_usd": sum(row.get("budget_debit_usd") or 0 for row in records),
        "provider_tokens": {field: sum(row.get(field) or 0 for row in records)
                            for field in token_fields},
    }


def execute(args) -> Path:
    if not os.environ.get(args.credential_env):
        raise ValueError(f"provider credential env {args.credential_env} is absent")
    schedule = json.loads(args.schedule.read_text())
    if (not math.isfinite(args.cap_usd) or args.cap_usd <= 0
            or args.cap_usd > schedule["cap_usd"]):
        raise ValueError("approved cap must be positive and no higher than scheduled cap")
    prices = json.loads(args.prices_json.read_text())
    if args.strong not in prices or args.weak not in prices:
        raise ValueError("price table must cover strong and weak model IDs")
    provider_url = urlsplit(args.upstream_base_url)
    if (provider_url.scheme not in ("http", "https") or not provider_url.hostname
            or provider_url.username or provider_url.password
            or provider_url.query or provider_url.fragment):
        raise ValueError("upstream URL must be a plain HTTP(S) base URL")
    if (provider_url.scheme == "http"
            and provider_url.hostname not in ("127.0.0.1", "localhost", "::1")):
        raise ValueError("remote provider URLs must use HTTPS")
    run_dir = benchmark.task_temp() / f"delegation-run-{uuid.uuid4().hex[:12]}"
    run_dir.mkdir()
    shutil.copyfile(args.schedule, run_dir / "schedule.json")
    shutil.copyfile(args.prices_json, run_dir / "prices.json")
    summary_path = run_dir / "results.jsonl"
    total_debit = 0.0
    arms = schedule["arms"][:args.max_arms] if args.max_arms else schedule["arms"]
    for index, arm in enumerate(arms):
        if total_debit >= args.cap_usd:
            break
        task = benchmark.TASKS[arm["task"]]
        if arm["baseline"] != task["baseline"]:
            raise ValueError("scheduled baseline changed")
        from hashlib import sha256
        if arm["oracle_sha256"] != sha256(benchmark.oracle_path(task).read_bytes()).hexdigest():
            raise ValueError("oracle changed after schedule was frozen")
        workspace = benchmark.snapshot(task["baseline"], task["id"])
        arm_dir = run_dir / f"{index:03d}-{arm['task']}-{arm['harness']}"
        arm_dir.mkdir()
        usage_log = arm_dir / "provider_usage.jsonl"
        relay_app = relay.build_relay_app(
            upstream_base_url=args.upstream_base_url,
            credential_env=args.credential_env,
            usage_log=usage_log, arm_id=arm_dir.name,
            prices=prices, cap_usd=args.cap_usd - total_debit,
            output_limit=args.output_limit,
        )
        with profile_arm(arm_dir) as profile_log:
            with serve(relay_app) as relay_port:
                gateway_backend = OpenAICompatibleBackend(
                    f"http://127.0.0.1:{relay_port}/v1")
                gateway_app = build_gateway_app(
                    ProviderRouter({"study": gateway_backend}))
                with serve(gateway_app) as gateway_port:
                    host_port = gateway_port if arm["harness"] == "claude" else relay_port
                    started = time.time()
                    error = None
                    try:
                        result, detail = run_cli(
                            harness=arm["harness"], prompt=task["prompt"],
                            workspace=workspace, strong=args.strong, weak=args.weak,
                            host_port=host_port, timeout=args.timeout,
                            output_limit=args.output_limit,
                        )
                    except Exception as exc:
                        result, detail = None, {"wall_seconds": time.time() - started}
                        error = type(exc).__name__
        grade = benchmark.grade(task, workspace)
        patch = snapshot_diff(task["baseline"], workspace, task["id"])
        (arm_dir / "patch.diff").write_text(patch)
        if result is not None:
            (arm_dir / "answer.txt").write_text(result.result)
        usage = usage_summary(usage_log, args.weak) if usage_log.exists() else {
            "provider_requests": 0, "weak_model_requests": 0,
            "full_usage": False, "cost_usd": 0, "budget_debit_usd": 0,
            "provider_tokens": {},
        }
        total_debit += usage["budget_debit_usd"]
        public_usage = detail.get("public_usage") or {}
        token_residual = {
            field: usage["provider_tokens"].get(field, 0) - public_usage.get(field, 0)
            for field in ("input_tokens", "output_tokens",
                          "cache_read_input_tokens", "cache_creation_input_tokens")
        }
        profile_events = [json.loads(line) for line in profile_log.read_text().splitlines()
                          if line] if profile_log.exists() else []
        profile_requests = [event for event in profile_events
                            if event.get("schema") == "chia.agent_profile"
                            and event.get("type") == "llm_request"]
        row = {
            "arm": arm, "prompt": task["prompt"],
            "strong_model": args.strong, "weak_model": args.weak,
            "agents": [asdict(agent) for agent in agent_definitions(args.strong, args.weak)],
            "provider_route": {
                "scheme": provider_url.scheme,
                "host": provider_url.hostname,
                "port": provider_url.port,
                "path": provider_url.path,
            },
            "output_limit": args.output_limit,
            "study_cost_source": "computed_from_provider_usage_and_supplied_rates",
            "claude_version": subprocess.check_output(["claude", "--version"], text=True).strip(),
            "opencode_version": subprocess.check_output(["opencode", "--version"], text=True).strip(),
            "workspace": str(workspace), "started_ts": started,
            "cli_returncode": result.returncode if result is not None else None,
            "error_type": error, "wall_seconds": detail["wall_seconds"],
            "chia_usage": public_usage or None,
            "chia_usage_scope": detail.get("metadata", {}).get("usage_scope"),
            "provider_minus_chia_tokens": token_residual,
            "profile_log": str(profile_log),
            "profile_request_count": len(profile_requests),
            "grade_pass": grade.returncode == 0,
            "grade_output_tail": grade.stdout[-1000:],
            "patch_bytes": len(patch.encode()), **usage,
        }
        with summary_path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps({"arm": index, "task": arm["task"],
                          "harness": arm["harness"], "grade_pass": row["grade_pass"],
                          "weak_requests": usage["weak_model_requests"],
                          "cost_usd": usage["cost_usd"]}))
        if not usage["full_usage"]:
            print("stopping: provider usage is incomplete; spending cannot be reconciled")
            break
    print(f"results: {summary_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["run"])
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--prices-json", type=Path, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--credential-env", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument("--weak", required=True)
    parser.add_argument("--cap-usd", type=float, required=True)
    parser.add_argument("--max-arms", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-limit", type=int, default=4096)
    args = parser.parse_args()
    if args.strong == args.weak or args.max_arms < 0 or args.timeout < 1:
        parser.error("distinct models, nonnegative max-arms, and positive timeout required")
    execute(args)


if __name__ == "__main__":
    main()
