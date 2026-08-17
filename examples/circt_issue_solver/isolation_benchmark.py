"""Paired Docker-versus-bubblewrap benchmark for the CIRCT worker.

Image pull/rootfs export is intentionally outside this program. Each trial
measures cluster readiness, 100 no-op CHIA dispatches, the existing CIRCT
warm/rebuild/lit primitives, and teardown. Results are append-only JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from chia.base.ChiaFunction import ChiaFunction


@ChiaFunction(resources={"circt": 0.001})
def _noop(value: int) -> int:
    return value


@ChiaFunction(resources={"circt": 1})
def _circt_primitives(full_lit: bool, repro_script_text: str | None) -> dict:
    import os
    import subprocess
    import time

    from chia.chipyard.circt import (
        _CIRCT_SOURCE_TREE,
        circt_ninja_build,
        circt_run_lit,
        circt_warm_build,
    )

    def timed(call):
        start = time.perf_counter()
        result = call()
        return time.perf_counter() - start, result

    warm_seconds, warm = timed(
        lambda: circt_warm_build(("circt-opt", "firtool"), num_cpus=16))
    repro_seconds = None
    repro = None
    if repro_script_text:
        def run_repro():
            repro_script = os.path.join(
                _CIRCT_SOURCE_TREE, ".circtissues", "benchmark-repro.sh")
            os.makedirs(os.path.dirname(repro_script), exist_ok=True)
            with open(repro_script, "w") as handle:
                handle.write(repro_script_text)
            proc = subprocess.run(
                ["bash", repro_script], cwd=_CIRCT_SOURCE_TREE,
                capture_output=True, text=True, timeout=1200, check=False)
            return {
                "returncode": proc.returncode,
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
                "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
            }
        repro_seconds, repro = timed(run_repro)
    rebuild_seconds, rebuild = timed(
        lambda: circt_ninja_build(("circt-opt", "firtool"), num_cpus=16))

    lit_seconds = None
    lit = None
    if full_lit:
        test_root = os.path.join(_CIRCT_SOURCE_TREE, "test")
        paths = tuple(
            f"test/{name}" for name in sorted(os.listdir(test_root))
            if name != "CAPI" and os.path.isdir(os.path.join(test_root, name)))
        lit_seconds, lit = timed(
            lambda: circt_run_lit(paths, filter_out="circt-tblgen"))
    return {
        "warm_seconds": warm_seconds, "warm": warm,
        "repro_seconds": repro_seconds, "repro": repro,
        "rebuild_seconds": rebuild_seconds, "rebuild": rebuild,
        "lit_seconds": lit_seconds, "lit": lit,
    }


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False)


def _version(command: list[str]) -> str:
    try:
        result = _run(command, 30)
        return (result.stdout or result.stderr).splitlines()[0]
    except Exception as exc:  # noqa: BLE001 - recorded benchmark evidence
        return f"unavailable: {exc}"


def _host_load() -> dict:
    load = os.getloadavg()
    return {"load_1m": load[0], "load_5m": load[1], "load_15m": load[2]}


def _root_pid(arm: str) -> int | None:
    if arm == "bwrap":
        path = Path(
            f"/tmp/chia-bwrap/circt_isolation_bwrap_{os.environ['USER']}-0/pid")
        try:
            return int(path.read_text())
        except (FileNotFoundError, ValueError):
            return None
    name = f"circt_isolation_docker_{os.environ['USER']}-0"
    result = _run(["docker", "inspect", "-f", "{{.State.Pid}}", name], 30)
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def _process_metrics(root_pid: int | None) -> dict:
    if not root_pid:
        return {"root_pid": None, "process_count": 0, "rss_bytes": 0,
                "pss_bytes": 0, "cpu_seconds": 0.0, "identities": {}}
    process_table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
            tail = text[text.rfind(")") + 2:].split()
            process_table[int(entry.name)] = {
                "ppid": int(tail[1]), "start": tail[19],
                "cpu_ticks": int(tail[11]) + int(tail[12]),
            }
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, info in process_table.items():
            if info["ppid"] in selected and pid not in selected:
                selected.add(pid)
                changed = True
    rss_kib = 0
    pss_kib = 0
    for pid in selected:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Pss:"):
                    pss_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    ticks = os.sysconf("SC_CLK_TCK")
    return {
        "root_pid": root_pid,
        "process_count": len(selected),
        "rss_bytes": rss_kib * 1024,
        "pss_bytes": pss_kib * 1024,
        "cpu_seconds": sum(
            process_table.get(pid, {}).get("cpu_ticks", 0) for pid in selected
        ) / ticks,
        "identities": {
            str(pid): info["start"] for pid in selected
            if (info := process_table.get(pid)) is not None
        },
    }


def _leaked_processes(identities: dict[str, str]) -> list[int]:
    leaked = []
    for raw_pid, expected_start in identities.items():
        try:
            text = Path(f"/proc/{raw_pid}/stat").read_text()
            actual_start = text[text.rfind(")") + 2:].split()[19]
        except (FileNotFoundError, PermissionError, IndexError):
            continue
        if actual_start == expected_start:
            leaked.append(int(raw_pid))
    return leaked


def _prepare_bwrap_writable(rootfs: Path, writable_root: Path) -> dict:
    """Create fresh overlay-equivalent writable trees outside timed startup."""
    start = time.perf_counter()
    rootfs = rootfs.resolve()
    writable_root = writable_root.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    forbidden = (Path.home().resolve(), repo_root, rootfs)
    if writable_root == Path("/") or any(
        writable_root == path
        or writable_root in path.parents
        or path in writable_root.parents
        for path in forbidden
    ):
        raise ValueError(
            f"unsafe --bwrap-writable-root {writable_root}; it must be a "
            "dedicated directory outside /, home, the CHIA worktree, and rootfs")

    marker = writable_root / ".chia-bwrap-benchmark-snapshots"
    if writable_root.exists() and any(writable_root.iterdir()) and not marker.is_file():
        raise ValueError(
            f"refusing to clean unmarked writable root {writable_root}; "
            f"expected marker {marker}")
    writable_root.mkdir(parents=True, exist_ok=True)
    marker.write_text("chia-bwrap-benchmark-snapshots-v1\n")
    copied = []
    for source_relative, destination_name in (
        (Path("workspace"), "workspace"),
        (Path("home/ray"), "home-ray"),
    ):
        source = rootfs / source_relative
        destination = writable_root / destination_name
        if destination.is_symlink():
            raise ValueError(f"refusing to remove symlinked snapshot {destination}")
        if destination.exists():
            shutil.rmtree(destination)
        result = _run(
            ["cp", "-a", "--reflink=auto", str(source), str(destination)],
            timeout=3600)
        if result.returncode:
            raise RuntimeError(
                f"snapshot copy failed for {source}: {result.stderr}")
        copied.append(str(destination))
    return {
        "prep_seconds": time.perf_counter() - start,
        "prep_policy": "fresh reflink/copy of image workspace and home",
        "writable_paths": copied,
    }


def _one_trial(
    arm: str,
    config: Path,
    dispatches: int,
    full_lit: bool,
    repro_script_text: str | None,
    ray_address: str,
) -> dict:
    import ray

    record = {
        "schema": 1, "arm": arm, "config": str(config.resolve()),
        "started_unix_ns": time.time_ns(), "load_before": _host_load(),
    }
    up_start = time.perf_counter()
    up = _run(["chia", "up", "--yes", str(config)], timeout=7200)
    record.update({
        "up_seconds": time.perf_counter() - up_start,
        "up_returncode": up.returncode,
        "up_stdout_tail": "\n".join(up.stdout.splitlines()[-80:]),
        "up_stderr_tail": "\n".join(up.stderr.splitlines()[-80:]),
    })
    if up.returncode:
        return record

    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        record["ray_resources"] = ray.cluster_resources()
        start = time.perf_counter()
        values = [_noop.chia_remote_blocking(i) for i in range(dispatches)]
        record.update({
            "dispatch_seconds": time.perf_counter() - start,
            "dispatch_count": dispatches,
            "dispatch_valid": values == list(range(dispatches)),
            "circt": _circt_primitives.chia_remote_blocking(
                full_lit, repro_script_text),
        })
        record["process_metrics_before_down"] = _process_metrics(_root_pid(arm))
    finally:
        ray.shutdown()
        start = time.perf_counter()
        down = _run(["chia", "down", "--yes", str(config)], timeout=600)
        record.update({
            "down_seconds": time.perf_counter() - start,
            "down_returncode": down.returncode,
            "down_stdout_tail": "\n".join(down.stdout.splitlines()[-80:]),
            "down_stderr_tail": "\n".join(down.stderr.splitlines()[-80:]),
            "load_after": _host_load(),
        })
        identities = (record.get("process_metrics_before_down") or {}).get(
            "identities", {})
        leaked = _leaked_processes(identities)
        record["leaked_processes"] = leaked
        record["leaked_process_count"] = len(leaked)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker-config", type=Path, required=True)
    parser.add_argument("--bwrap-config", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--dispatches", type=int, default=100)
    parser.add_argument("--full-lit", action="store_true")
    parser.add_argument("--repro-script", type=Path)
    parser.add_argument("--expected-repro-exit", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("isolation-benchmark.jsonl"))
    parser.add_argument("--seed", type=int, default=10568)
    parser.add_argument("--ray-address", default="127.0.0.1:46379")
    parser.add_argument(
        "--bwrap-rootfs", type=Path,
        default=Path("/scratch/agustin/cache/chia-bwrap-rootfs/sha256-5c61bd07a4716d2d1f235300138f6c5dd97c42738f7195cb1d3570c9539493c8"))
    parser.add_argument(
        "--bwrap-writable-root", type=Path,
        default=Path("/scratch/agustin/cache/chia-bwrap-benchmark"))
    args = parser.parse_args()
    if args.trials < 1 or args.dispatches < 1:
        parser.error("--trials and --dispatches must be positive")

    manifest = {
        "event": "manifest", "schema": 1,
        "git_commit": _version(["git", "rev-parse", "HEAD"]),
        "kernel": platform.release(), "python": sys.version,
        "ray": _version([sys.executable, "-c", "import ray; print(ray.__version__)"]),
        "docker": _version(["docker", "--version"]),
        "bwrap": _version(["bwrap", "--version"]),
        "image_id": "sha256:5c61bd07a4716d2d1f235300138f6c5dd97c42738f7195cb1d3570c9539493c8",
        "image_digest": "sha256:09e59dfa30035b773dd978a84e5c3c76b9326eb9e846ced025093cdeca96145d",
        "issue": "https://github.com/llvm/circt/issues/10568",
        "repro_script_sha256": (
            hashlib.sha256(args.repro_script.read_bytes()).hexdigest()
            if args.repro_script else None),
        "expected_repro_exit_code": args.expected_repro_exit,
        "trials": args.trials, "dispatches": args.dispatches,
        "full_lit": args.full_lit, "seed": args.seed,
        "ray_address": args.ray_address,
        "writable_policy": (
            "Docker gets a fresh container overlay and bwrap gets fresh "
            "workspace/home reflink-or-copy snapshots before every arm. "
            "Both begin with the build state baked into the pinned image."),
        "timing_policy": "snapshot preparation is recorded but excluded from up_seconds",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output:
        output.write(json.dumps(manifest, sort_keys=True) + "\n")
        output.flush()
        rng = random.Random(args.seed)
        configs = {"docker": args.docker_config, "bwrap": args.bwrap_config}
        for pair in range(args.trials):
            arms = ["docker", "bwrap"]
            rng.shuffle(arms)
            for order, arm in enumerate(arms):
                prep = (
                    _prepare_bwrap_writable(
                        args.bwrap_rootfs, args.bwrap_writable_root)
                    if arm == "bwrap"
                    else {
                        "prep_seconds": 0.0,
                        "prep_policy": "fresh Docker overlay created by chia up",
                        "writable_paths": [],
                    }
                )
                record = _one_trial(
                    arm, configs[arm], args.dispatches, args.full_lit,
                    args.repro_script.read_text() if args.repro_script else None,
                    args.ray_address)
                record["preparation"] = prep
                record.update({"pair": pair, "pair_order": order})
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                if record.get("up_returncode") or record.get("down_returncode"):
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
