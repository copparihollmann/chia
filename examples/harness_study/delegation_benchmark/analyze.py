"""Analyze matched Claude Code versus OpenCode task outcomes.

Provider request usage is authoritative; cost is derived from the supplied
model price table. A task counts as delegated success only
when its held-out oracle passes and the weak model received a provider request.
The paired cost/time comparison uses pairs in which both harnesses succeeded.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def _cluster_interval(observations: list[tuple[str, float]], statistic,
                      *, seed: int = 20261012,
                      resamples: int = 10_000) -> list[float] | None:
    """Resample task IDs, keeping all repetitions of a sampled task together."""
    if not observations:
        return None
    grouped = {}
    for task, value in observations:
        grouped.setdefault(task, []).append(value)
    tasks = sorted(grouped)
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        selected = rng.choices(tasks, k=len(tasks))
        values = [value for task in selected for value in grouped[task]]
        samples.append(statistic(values))
    samples.sort()
    return [samples[int(0.025 * (resamples - 1))],
            samples[int(0.975 * (resamples - 1))]]


def analyze(rows: list[dict], schedule: dict) -> dict:
    planned = {(arm["task"], arm["repeat"], arm["harness"])
               for arm in schedule["arms"]}
    observed = {}
    for row in rows:
        arm = row["arm"]
        key = (arm["task"], arm["repeat"], arm["harness"])
        if key not in planned or key in observed:
            raise ValueError(f"unexpected or duplicate arm: {key}")
        observed[key] = row
    pair_keys = sorted({(task, repeat) for task, repeat, _ in planned})
    pairs = []
    for task, repeat in pair_keys:
        claude = observed.get((task, repeat, "claude"))
        opencode = observed.get((task, repeat, "opencode"))
        if claude is None or opencode is None:
            continue
        pairs.append((task, claude, opencode))

    def delegated_success(row):
        return bool(row.get("grade_pass") and row.get("weak_model_requests", 0) > 0)

    success_diffs = [(task, int(delegated_success(c)) - int(delegated_success(o)))
                     for task, c, o in pairs]
    both_succeeded = [(task, c, o) for task, c, o in pairs
                      if delegated_success(c) and delegated_success(o)]
    priced = [(task, c, o) for task, c, o in both_succeeded
              if c.get("full_usage") and o.get("full_usage")]
    cost_diffs = [(task, c["cost_usd"] - o["cost_usd"]) for task, c, o in priced]
    time_diffs = [(task, c["wall_seconds"] - o["wall_seconds"])
                  for task, c, o in both_succeeded]
    return {
        "planned_pairs": len(pair_keys),
        "complete_pairs": len(pairs),
        "incomplete_arms": 2 * len(pair_keys) - len(observed),
        "complete_task_clusters": len({task for task, _, _ in pairs}),
        "delegated_success": {
            "claude": sum(delegated_success(c) for _, c, _ in pairs),
            "opencode": sum(delegated_success(o) for _, _, o in pairs),
            "paired_mean_difference_claude_minus_opencode": (
                statistics.mean(value for _, value in success_diffs)
                if success_diffs else None),
            "task_cluster_bootstrap_95_interval": _cluster_interval(
                success_diffs, statistics.mean),
        },
        "both_succeeded_pairs": len(both_succeeded),
        "both_succeeded_fully_priced_pairs": len(priced),
        "provider_cost_usd": {
            "median_paired_difference_claude_minus_opencode": (
                statistics.median(value for _, value in cost_diffs)
                if cost_diffs else None),
            "task_cluster_bootstrap_95_interval": _cluster_interval(
                cost_diffs, statistics.median),
        },
        "cli_wall_seconds": {
            "median_paired_difference_claude_minus_opencode": (
                statistics.median(value for _, value in time_diffs)
                if time_diffs else None),
            "task_cluster_bootstrap_95_interval": _cluster_interval(
                time_diffs, statistics.median),
        },
        "cannot_rank_full_study": (len(pairs) != len(pair_keys)
                                   or any(not row.get("full_usage")
                                          for row in observed.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / "results.jsonl").read_text().splitlines()
            if line]
    schedule = json.loads((args.run_dir / "schedule.json").read_text())
    print(json.dumps(analyze(rows, schedule), indent=2))


if __name__ == "__main__":
    main()
