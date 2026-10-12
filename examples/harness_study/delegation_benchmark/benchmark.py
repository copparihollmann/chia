"""Prepare and grade fixed Chia delegation tasks without provider calls.

The CLI creates ordinary source snapshots from pinned commits under $TMPDIR.
It does not clone a repository or create a Git worktree. The oracle stays
outside the agent workspace until grading. ``verify`` checks each task is red
at its pinned starting commit and green at the integrated solution commit.

Examples::

    TMPDIR=/path/to/managed-task/.tmp python benchmark.py verify
    TMPDIR=/path/to/managed-task/.tmp python benchmark.py schedule --seed 20261012
    TMPDIR=/path/to/managed-task/.tmp python benchmark.py prepare claude_usage
    TMPDIR=/path/to/managed-task/.tmp python benchmark.py grade claude_usage /path/to/snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TASKS = {task["id"]: task for task in json.loads((HERE / "tasks.json").read_text())}


def task_temp() -> Path:
    value = os.environ.get("TMPDIR")
    if not value:
        raise RuntimeError("set TMPDIR to the managed task's .tmp directory")
    root = Path(value).resolve()
    if not root.is_dir():
        raise RuntimeError(f"TMPDIR does not exist: {root}")
    return root


def oracle_path(task: dict) -> Path:
    return HERE / "oracles" / task["oracle"]


def snapshot(commit: str, task_id: str) -> Path:
    target = Path(tempfile.mkdtemp(prefix=f"chia-{task_id}-", dir=task_temp()))
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit], cwd=REPO,
        check=True, capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(target, filter="data")
    return target


def grade(task: dict, workspace: Path) -> subprocess.CompletedProcess[str]:
    workspace = workspace.resolve()
    if not workspace.is_dir() or not workspace.is_relative_to(task_temp()):
        raise RuntimeError("grade only accepts a snapshot under the managed task TMPDIR")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(oracle_path(task))],
        cwd=workspace, env=env, capture_output=True, text=True,
        timeout=90,
    )


def verify() -> int:
    solved = snapshot("HEAD", "solution")
    failures = []
    for task in TASKS.values():
        baseline = snapshot(task["baseline"], task["id"])
        before = grade(task, baseline)
        after = grade(task, solved)
        expected = task["expected_baseline_failure"]
        valid = (before.returncode == 1 and expected in before.stdout
                 and after.returncode == 0)
        print(f"{task['id']}: baseline={before.returncode} solution={after.returncode} "
              f"{'VALID' if valid else 'INVALID'}")
        if not valid:
            failures.append(task["id"])
            print("baseline output:", before.stdout[-1600:], before.stderr[-500:])
            print("solution output:", after.stdout[-1600:], after.stderr[-500:])
    return 1 if failures else 0


def schedule(seed: int, repeats: int, cap_usd: float) -> Path:
    if repeats <= 0 or cap_usd <= 0:
        raise ValueError("repeats and cap_usd must be positive")
    rng = random.Random(seed)
    pairs = [(task_id, repeat) for task_id in TASKS for repeat in range(repeats)]
    rng.shuffle(pairs)
    arms = []
    for task_id, repeat in pairs:
        order = ["claude", "opencode"]
        rng.shuffle(order)
        for harness in order:
            task = TASKS[task_id]
            arms.append({
                "task": task_id, "repeat": repeat, "harness": harness,
                "baseline": task["baseline"],
                "oracle_sha256": hashlib.sha256(oracle_path(task).read_bytes()).hexdigest(),
            })
    path = task_temp() / f"chia-delegation-schedule-seed{seed}.json"
    path.write_text(json.dumps({
        "seed": seed, "repeats": repeats, "cap_usd": cap_usd,
        "arms": arms,
    }, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("verify")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("task", choices=TASKS)
    grading = commands.add_parser("grade")
    grading.add_argument("task", choices=TASKS)
    grading.add_argument("workspace", type=Path)
    planning = commands.add_parser("schedule")
    planning.add_argument("--seed", type=int, default=20261012)
    planning.add_argument("--repeats", type=int, default=5)
    planning.add_argument("--cap-usd", type=float, default=25.0)
    args = parser.parse_args()
    if args.command == "list":
        for task in TASKS.values():
            print(f"{task['id']}  baseline={task['baseline'][:7]}  {task['prompt']}")
        return 0
    if args.command == "verify":
        return verify()
    if args.command == "prepare":
        print(snapshot(TASKS[args.task]["baseline"], args.task))
        return 0
    if args.command == "grade":
        result = grade(TASKS[args.task], args.workspace)
        print(result.stdout, end="")
        print(result.stderr, file=sys.stderr, end="")
        return result.returncode
    if args.command == "schedule":
        print(schedule(args.seed, args.repeats, args.cap_usd))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
