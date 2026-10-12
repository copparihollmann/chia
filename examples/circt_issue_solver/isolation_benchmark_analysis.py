"""Machine-readable paired analysis for ``isolation_benchmark.py``."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def _representative_seconds(record: dict) -> float:
    circt = record.get("circt") or {}
    return sum(float(value or 0) for value in (
        record.get("up_seconds"),
        record.get("dispatch_seconds"),
        circt.get("warm_seconds"),
        circt.get("repro_seconds"),
        circt.get("rebuild_seconds"),
        circt.get("lit_seconds"),
        record.get("down_seconds"),
    ))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def _bootstrap_median_ci(
    values: list[float], samples: int, seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    bootstrapped = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return _percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)


def analyze(
    records: list[dict],
    min_pairs: int = 10,
    threshold_percent: float = 5.0,
    bootstrap_samples: int = 10_000,
    seed: int = 10568,
) -> dict:
    manifest = next(
        (record for record in records if record.get("event") == "manifest"), {})
    workload_complete = (
        manifest.get("full_lit") is True
        and bool(manifest.get("repro_script_sha256"))
    )
    expected_repro_exit = manifest.get("expected_repro_exit_code")
    by_pair: dict[int, dict[str, dict]] = {}
    for record in records:
        if "pair" in record and record.get("arm") in ("docker", "bwrap"):
            by_pair.setdefault(int(record["pair"]), {})[record["arm"]] = record
    pairs = [arms for arms in by_pair.values() if set(arms) == {"docker", "bwrap"}]
    if not pairs:
        raise ValueError("no complete Docker/bwrap pairs found")

    representative = {
        arm: [_representative_seconds(pair[arm]) for pair in pairs]
        for arm in ("docker", "bwrap")
    }
    startup = {
        arm: [float(pair[arm]["up_seconds"]) for pair in pairs]
        for arm in ("docker", "bwrap")
    }
    representative_delta = [
        docker - bwrap for docker, bwrap in
        zip(representative["docker"], representative["bwrap"], strict=True)
    ]
    startup_delta = [
        docker - bwrap for docker, bwrap in
        zip(startup["docker"], startup["bwrap"], strict=True)
    ]
    rep_ci = _bootstrap_median_ci(
        representative_delta, bootstrap_samples, seed)
    startup_ci = _bootstrap_median_ci(
        startup_delta, bootstrap_samples, seed + 1)
    docker_median = statistics.median(representative["docker"])
    bwrap_median = statistics.median(representative["bwrap"])
    improvement_percent = 100 * (docker_median - bwrap_median) / docker_median

    successful = all(
        record.get("up_returncode") == 0
        and record.get("down_returncode") == 0
        and record.get("dispatch_valid") is True
        and record.get("leaked_process_count", 0) == 0
        and (record.get("circt") or {}).get("warm", {}).get("success") is True
        and (record.get("circt") or {}).get("rebuild", {}).get("success") is True
        and (record.get("circt") or {}).get("lit", {}).get("success") is True
        and isinstance((record.get("circt") or {}).get("repro"), dict)
        and (record.get("circt") or {}).get("repro", {}).get("returncode")
        == expected_repro_exit
        for pair in pairs for record in pair.values()
    )
    enough_pairs = len(pairs) >= min_pairs
    threshold_met = improvement_percent >= threshold_percent
    ci_excludes_zero = rep_ci[0] > 0
    go = (
        successful and workload_complete and enough_pairs
        and threshold_met and ci_excludes_zero
    )
    return {
        "schema": 1,
        "decision": "go" if go else "no-go",
        "criteria": {
            "all_trials_successful": successful,
            "representative_workload_complete": workload_complete,
            "minimum_pairs": min_pairs,
            "enough_pairs": enough_pairs,
            "threshold_percent": threshold_percent,
            "threshold_met": threshold_met,
            "paired_bootstrap_ci_excludes_zero": ci_excludes_zero,
        },
        "representative_end_to_end": {
            "docker_median_seconds": docker_median,
            "bwrap_median_seconds": bwrap_median,
            "median_improvement_seconds": statistics.median(representative_delta),
            "improvement_percent": improvement_percent,
            "paired_bootstrap_95_ci_seconds": list(rep_ci),
        },
        "startup_secondary": {
            "docker_median_seconds": statistics.median(startup["docker"]),
            "bwrap_median_seconds": statistics.median(startup["bwrap"]),
            "median_improvement_seconds": statistics.median(startup_delta),
            "paired_bootstrap_95_ci_seconds": list(startup_ci),
        },
        "complete_pairs": len(pairs),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--threshold-percent", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=10568)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    summary = analyze(
        records, min_pairs=args.min_pairs,
        threshold_percent=args.threshold_percent,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
