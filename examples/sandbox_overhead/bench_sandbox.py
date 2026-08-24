"""Measure what a per-call sandbox costs, per call.

This is where the "bwrap or docker?" question actually lives. Neither replaces the
other and neither is free: the question is how much wall-clock each adds to a *single*
agent call, because a grid of hundreds of calls pays that tax hundreds of times.

The command benchmarked is a no-op (``true``), so the number is setup overhead and
nothing else — no model, no network, no tokens. It costs $0 to reproduce.

Both backends are measured on the same host in the same run, since container start-up
is dominated by daemon and image-layer behaviour that varies wildly between machines;
a figure quoting one host's absolute milliseconds as a general fact would be wrong.
What generalises is the *ratio* and the shape of the distribution.

Usage::

    python examples/sandbox_overhead/bench_sandbox.py --repeats 200
    python examples/sandbox_overhead/bench_sandbox.py --plot      # draw from the CSV
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chia.base.sandbox import (  # noqa: E402
    available_backends,
    spec_for_command,
    wrap_argv,
)

HERE = Path(__file__).parent
DEFAULT_CSV = HERE / "sandbox_overhead.csv"
DEFAULT_PNG = HERE / "sandbox_overhead.png"

COLUMNS = ("backend", "repeat", "wall_ms", "returncode")


def time_one(backend: str, workspace: Path, docker_image: str) -> Dict:
    """Run a no-op command once under *backend* and return its wall time in ms."""
    spec = spec_for_command(workspace, "true", docker_image=docker_image)
    # The bare name on purpose: wrap_argv resolves it against the host for none/bwrap
    # and leaves it for docker to resolve inside the image.
    argv = wrap_argv(["true"], spec, backend=backend)
    start = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, timeout=120)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"backend": backend, "wall_ms": round(elapsed_ms, 3),
            "returncode": proc.returncode}


def measure(repeats: int, backends: List[str], workspace: Path,
            docker_image: str) -> List[Dict]:
    """Interleave the backends across repeats.

    Interleaved rather than run in blocks on purpose: a block-per-backend design
    attributes any drift in host load to whichever backend happened to run during it.
    """
    rows: List[Dict] = []
    # One untimed warm-up per backend, so the first measured call does not carry
    # page-cache and image-layer costs that no later call pays.
    for backend in backends:
        try:
            time_one(backend, workspace, docker_image)
        except Exception as exc:
            print(f"warm-up failed for {backend}: {exc}", file=sys.stderr)
    for repeat in range(repeats):
        for backend in backends:
            try:
                row = time_one(backend, workspace, docker_image)
            except Exception as exc:
                print(f"{backend} repeat {repeat} failed: {exc}", file=sys.stderr)
                continue
            row["repeat"] = repeat
            rows.append(row)
    return rows


def summarise(rows: List[Dict]) -> None:
    """Print median / p90 / n per backend, so a run is readable without the figure."""
    by_backend: Dict[str, List[float]] = {}
    for row in rows:
        if row["returncode"] == 0:
            by_backend.setdefault(row["backend"], []).append(float(row["wall_ms"]))
    baseline = statistics.median(by_backend.get("none", [0.0])) or 0.0
    print(f"{'backend':10} {'n':>5} {'median ms':>10} {'p90 ms':>9} {'vs none':>9}")
    for backend, samples in by_backend.items():
        samples.sort()
        median = statistics.median(samples)
        p90 = samples[int(len(samples) * 0.9) - 1] if samples else float("nan")
        delta = f"+{median - baseline:.1f}" if baseline else "-"
        print(f"{backend:10} {len(samples):5d} {median:10.2f} {p90:9.2f} {delta:>9}")


def plot(csv_path: Path, png_path: Path) -> None:
    """Draw the distribution per backend, plus what a 500-call grid would pay."""
    try:
        import matplotlib
    except ImportError:
        sys.exit("matplotlib is required for --plot")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["returncode"] == "0"]
    by_backend: Dict[str, List[float]] = {}
    for row in rows:
        by_backend.setdefault(row["backend"], []).append(float(row["wall_ms"]))
    order = [b for b in ("none", "bwrap", "docker") if b in by_backend]
    if not order:
        sys.exit(f"no successful rows in {csv_path}")

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linestyle": ":"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.3),
                                      gridspec_kw={"width_ratios": [1.2, 1],
                                                   "wspace": 0.34})

    left.boxplot([by_backend[b] for b in order], tick_labels=order, showfliers=False,
                 medianprops={"color": "#c05621"})
    left.set_ylabel("wall time per call (ms)")
    left.set_yscale("log")
    n = min(len(by_backend[b]) for b in order)
    left.set_title(f"Setup overhead, no-op command (n = {n} each)", loc="left",
                   fontsize=9.5)

    baseline = statistics.median(by_backend["none"]) if "none" in by_backend else 0.0
    grid_calls = 500
    added = [(statistics.median(by_backend[b]) - baseline) * grid_calls / 1000
             for b in order]
    colours = {"none": "#718096", "bwrap": "#2f855a", "docker": "#c05621"}
    right.bar(order, added, color=[colours.get(b, "#718096") for b in order],
              width=0.6)
    for index, value in enumerate(added):
        right.text(index, value, f" {value:.0f}s", ha="center", va="bottom",
                   fontsize=8)
    right.set_ylabel(f"added wall time over\n{grid_calls} calls (s)")
    right.set_title(f"What a {grid_calls}-call grid pays", loc="left", fontsize=9.5)

    if "bwrap" in by_backend and "docker" in by_backend:
        ratio = ((statistics.median(by_backend["docker"]) - baseline)
                 / max(statistics.median(by_backend["bwrap"]) - baseline, 1e-9))
        fig.suptitle(
            f"Per-call docker costs {ratio:.0f}x what bwrap does — and needs a daemon "
            f"and group membership bwrap does not",
            fontsize=10, y=1.04,
        )
    fig.text(0.5, -0.12,
             "one host, one run, interleaved; absolute milliseconds are "
             "host-specific — the ratio and the shape are what generalise",
             ha="center", fontsize=7.5, color="#718096")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"wrote {png_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--docker-image", default="alpine:3",
                        help="a tiny image, so the figure measures docker's start-up "
                             "rather than a large agent image's layers")
    parser.add_argument("--plot", action="store_true",
                        help="draw from an existing CSV instead of measuring")
    args = parser.parse_args(argv)

    if args.plot:
        plot(args.csv, args.png)
        return 0

    backends = available_backends()
    print(f"measuring {backends} x {args.repeats} repeats")
    workspace = HERE / "_bench_ws"
    workspace.mkdir(parents=True, exist_ok=True)
    rows = measure(args.repeats, backends, workspace, args.docker_image)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})
    print(f"wrote {len(rows)} rows to {args.csv}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
