"""Render paired Docker/bwrap CIRCT benchmark evidence as PNG and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PHASES = (
    ("startup", "up_seconds"),
    ("dispatch", "dispatch_seconds"),
    ("warm build", "circt.warm_seconds"),
    ("repro", "circt.repro_seconds"),
    ("rebuild", "circt.rebuild_seconds"),
    ("full lit", "circt.lit_seconds"),
    ("teardown", "down_seconds"),
)


def _value(record: dict, path: str) -> float:
    value = record
    for component in path.split("."):
        value = value[component]
    return float(value)


def _load(path: Path) -> list[dict[str, dict]]:
    pairs: dict[int, dict[str, dict]] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "pair" in record and record.get("arm") in {"docker", "bwrap"}:
            pairs.setdefault(int(record["pair"]), {})[record["arm"]] = record
    complete = [pairs[index] for index in sorted(pairs)
                if set(pairs[index]) == {"docker", "bwrap"}]
    if not complete:
        raise ValueError("no complete pairs")
    return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    pairs = _load(args.input)
    rows = []
    for pair_index, pair in enumerate(pairs):
        for arm in ("docker", "bwrap"):
            record = pair[arm]
            phases = {name: _value(record, path) for name, path in PHASES}
            metrics = record.get("process_metrics_before_down") or {}
            rows.append({
                "pair": pair_index, "arm": arm, **phases,
                "end_to_end": sum(phases.values()),
                "rss_mib": float(metrics.get("rss_bytes") or 0) / 2**20,
                "process_count": int(metrics.get("process_count") or 0),
                "cpu_seconds": float(metrics.get("cpu_seconds") or 0),
            })
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    colors = {"docker": "#3572A5", "bwrap": "#D97706"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    x = range(len(PHASES))
    width = 0.36
    for offset, arm in ((-width / 2, "docker"), (width / 2, "bwrap")):
        medians = [statistics.median(
            row[name] for row in rows if row["arm"] == arm)
            for name, _ in PHASES]
        ax.bar([value + offset for value in x], medians, width,
               label=arm, color=colors[arm])
    ax.set_xticks(list(x), [name for name, _ in PHASES], rotation=35, ha="right")
    ax.set_ylabel("median seconds")
    ax.set_title("Representative CIRCT phase cost")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    docker_e2e = [row["end_to_end"] for row in rows if row["arm"] == "docker"]
    bwrap_e2e = [row["end_to_end"] for row in rows if row["arm"] == "bwrap"]
    for docker, bwrap in zip(docker_e2e, bwrap_e2e, strict=True):
        ax.plot((0, 1), (docker, bwrap), color="#9CA3AF", alpha=0.65)
        ax.scatter((0, 1), (docker, bwrap),
                   color=(colors["docker"], colors["bwrap"]), s=30)
    ax.set_xticks((0, 1), ("Docker", "bwrap"))
    ax.set_ylabel("seconds")
    ax.set_title("Paired end-to-end trials (one line per pair)")

    ax = axes[1, 0]
    deltas = [docker - bwrap for docker, bwrap in
              zip(docker_e2e, bwrap_e2e, strict=True)]
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(range(len(deltas)), deltas,
           color=[colors["bwrap"] if value > 0 else colors["docker"] for value in deltas])
    ax.set_xlabel("pair")
    ax.set_ylabel("Docker − bwrap seconds")
    ax.set_title("Positive means bwrap is faster")

    ax = axes[1, 1]
    labels = ("RSS", "processes", "CPU time")
    keys = ("rss_mib", "process_count", "cpu_seconds")
    medians_by_arm = {
        arm: [statistics.median(
            row[key] for row in rows if row["arm"] == arm) for key in keys]
        for arm in ("docker", "bwrap")
    }
    for offset, arm in ((-width / 2, "docker"), (width / 2, "bwrap")):
        normalized = [value / baseline for value, baseline in zip(
            medians_by_arm[arm], medians_by_arm["docker"], strict=True)]
        bars = ax.bar([value + offset for value in range(3)], normalized, width,
                      label=arm, color=colors[arm])
        units = ("MiB", "", "s")
        for bar, value, unit in zip(
                bars, medians_by_arm[arm], units, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.1f}{unit}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(range(3), labels)
    ax.axhline(1, color="#9CA3AF", linewidth=1, zorder=0)
    ax.set_ylabel("ratio to Docker median")
    ax.set_ylim(0, max(
        value / baseline
        for values in medians_by_arm.values()
        for value, baseline in zip(values, medians_by_arm["docker"], strict=True)
    ) * 1.16)
    ax.set_title("Worker resource footprint before teardown")
    ax.legend(frameon=False)

    fig.suptitle(
        f"CHIA logical-worker isolation — CIRCT issue #10568, n={len(pairs)} paired trials",
        fontsize=14,
    )
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=180)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
