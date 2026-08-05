"""Produce a chia run that ``aet plot`` can draw, and prove the run directory is the contract.

Why this example exists
-----------------------

``chia.trace.aet_sink`` claims that a chia run directory *is* an aet run: point ``aet plot``
at it and a figure comes out, with no chia code in the rendering path. Nothing in the
repository demonstrated that. The sink had unit tests, and ``examples/harness_study``
reads aet run directories, but no artifact anywhere showed the whole path -- profiled
``@ChiaFunction`` to collector actor to sink to ``aet plot`` -- actually closing.

Neither existing vehicle could show it. ``examples/harness_study/grid.py`` runs its rows
in-process with no Ray and no profiler, so the collector path is never touched;
``examples/memcpy`` is gated on ``resources={"chipyard": 0.05}`` and needs a toolchain
cluster to say anything at all. So: a small example whose only job is the path.

What it runs
------------

N calls to ``ClaudeCodeLLM.prompt`` -- chia's own ``@ChiaFunction``, dispatched through
Ray, recording its usage to the profiler exactly as it does in any other example -- sharing
one deliberately long system prompt. The first call writes that prompt into the provider's
cache and every later call reads it back, which is what puts real cache-write-then-
cache-read structure on the curve.

That structure is the point. Cache writes bill at 1.25x the input rate and reads at 0.1x,
twelve and a half times apart, so a trajectory carrying only their sum cannot say where a
run's cache spend went. The questions themselves are trivial arithmetic, because the figure
is about the accounting and not about the answers.

The two commands
----------------

The run is produced from chia's venv and the figure from aet's, reading the same
directory::

    # chia's venv: run it, write the aet run dir
    python examples/aet_trajectory/run.py --calls 8 --run-dir /tmp/chia_aet_demo

    # aet's venv: render it, with no chia on the path at all
    aet plot /tmp/chia_aet_demo --kind trajectory --split-cache \
        --out examples/aet_trajectory/figures/trajectory.png

The venv split is not a workaround to apologise for -- it is the evidence. chia's venv has
aet installed but no matplotlib; aet's has matplotlib but no chia. The two processes share
nothing but the directory, so a figure coming out the far side means the directory carried
everything.

Reproducing it for $0
---------------------

``--replay`` rebuilds the same run directory from the per-call log the live run wrote
alongside it, with no credentials and no spend::

    python examples/aet_trajectory/run.py \
        --replay examples/aet_trajectory/data/cold_calls.jsonl --run-dir /tmp/cold

Those rows are the profiler's own records from the live run, costs included as the backend
reported them -- nothing is re-priced from a local rate table, which would let the replay
drift away from the run it claims to be. ``test_run.py`` asserts the replay reconstructs
byte-identical trajectory series.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

from chia.base.budget import BudgetExceeded, check_budget  # noqa: E402

#: Cheapest Anthropic tier on Bedrock. Anthropic rather than Nova because the CLI's native
#: Bedrock transport only speaks to Anthropic models, and because prompt caching -- the
#: thing this example is built to show -- is what the CLI enables for them by default.
DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

#: A system prompt long enough to be worth caching. A provider only creates a cache entry
#: above a minimum prefix length (~1k tokens for Haiku), so a short prompt gives a run with
#: no cache tokens at all and a figure with nothing to split. Padded with filler that says
#: it is filler, so nobody reads meaning into the length.
_PREAMBLE = (
    "You are a calculator. Answer with the number and nothing else -- no units, no "
    "working, no punctuation. "
)
_FILLER = (
    "The following sentence exists only to push this system prompt past the provider's "
    "minimum cacheable prefix length so that a cache entry is actually created; it carries "
    "no instructions and should be ignored entirely. "
)
SYSTEM_MESSAGE_TEMPLATE = _PREAMBLE + _FILLER * 40 + "Session tag: {nonce}. "


def system_message(nonce: str) -> str:
    """The system prompt, tagged so the run starts with a genuine cache miss.

    The nonce is not decoration. The cached prefix is keyed on its content, and the provider
    holds an entry for several minutes, so a second run started soon after the first inherits
    a warm cache and every one of its calls is a read -- a trajectory with no cache-write
    structure in it at all, which is exactly what the first attempt at this example produced.
    Tagging the prefix per run guarantees the first call is a write and the rest are reads,
    which is the shape a first-ever run has. It forces a cold start; it does not fake one.
    """
    return SYSTEM_MESSAGE_TEMPLATE.format(nonce=nonce)

#: One trivial question per call, all distinct -- two identical prompts could be served from
#: a response cache, which would flatten the curve being measured.
QUESTIONS = [f"What is {i} times {i + 7}?" for i in range(2, 64)]

#: The usage keys the sink folds out of a profiler event's ``extra``. Projecting to exactly
#: these is what keeps ``calls.jsonl`` committable: the profiler's raw events also carry
#: worker ips, node ids and pids, which are host details rather than measurements.
_USAGE_KEYS = (
    "model", "billing_mode", "num_turns", "cost_usd", "cost_source",
    "input_tokens", "output_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens",
)


# ---------------------------------------------------------------------------
# The per-call log — what makes the run reproducible without credentials
# ---------------------------------------------------------------------------


def event_to_row(event: dict) -> Optional[dict]:
    """Project one profiler event to a committable row, or ``None`` if it is not an LLM call."""
    extra = event.get("extra")
    if not isinstance(extra, dict):
        return None
    usage = {key: extra[key] for key in _USAGE_KEYS if extra.get(key) is not None}
    if not usage:
        return None
    return {
        "call_id": str(event.get("call_id", "")),
        "func": str(event.get("func", "")),
        "ts": float(event.get("ts", 0.0) or 0.0),
        **usage,
    }


def row_to_event(row: dict) -> dict:
    """A committed row back into the event shape :func:`collect_run_usage` folds."""
    return {
        "type": "complete",
        "call_id": row["call_id"],
        "func": row.get("func", "ClaudeCodeLLM.prompt"),
        "ts": float(row["ts"]),
        "extra": {key: row[key] for key in _USAGE_KEYS if row.get(key) is not None},
    }


def read_calls(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_calls(path: Path, rows: List[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


# ---------------------------------------------------------------------------
# Writing the run dir + the figure's source data
# ---------------------------------------------------------------------------


def record(events: List[dict], run_dir: Path, *, run_id: str) -> bool:
    """Fold *events* into an aet run directory. Returns whether one was written."""
    from chia.trace.aet_sink import record_run

    return record_run(events, run_dir=run_dir, run_id=run_id, project="chia",
                      suite="aet_trajectory", target="cache-split",
                      method="claude-cli-bedrock", enabled=True)


def trajectory_series(run_dir: Path) -> Dict[str, List[float]]:
    """The ``aet.traj.*`` step-metric families read back out of the run directory.

    Read back rather than kept from memory on purpose: this is what a reader checks the
    figure against, so it must come from the same place the figure does. If the run
    directory lost something, this loses it too and the discrepancy is visible.
    """
    rows: Dict[str, Dict[int, float]] = {}
    for line in (run_dir / "logs" / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record_ = json.loads(line)
        name = record_.get("name", "")
        if record_.get("step") is None or not name.startswith("aet.traj."):
            continue
        rows.setdefault(name, {})[int(record_["step"])] = record_["value"]
    return {name: [value for _, value in sorted(steps.items())]
            for name, steps in rows.items()}


#: ``csv column -> step-metric name``. The figure's lines, and nothing else.
_CSV_COLUMNS = (
    ("t_s", "aet.traj.t_s"),
    ("cum_input_tokens", "aet.traj.cum_input_tokens"),
    ("cum_output_tokens", "aet.traj.cum_output_tokens"),
    ("cum_cache_read_tokens", "aet.traj.cum_cache_read_tokens"),
    ("cum_cache_creation_tokens", "aet.traj.cum_cache_creation_tokens"),
    ("cum_cache_tokens", "aet.traj.cum_cache_tokens"),
    ("cum_cost_usd", "aet.traj.cum_cost_usd"),
)


def write_source_data(run_dir: Path, out_csv: Path) -> Path:
    """Write the figure's numbers as a CSV so every line in it can be checked."""
    series = trajectory_series(run_dir)
    n = len(series.get("aet.traj.t_s", []))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point"] + [name for name, _ in _CSV_COLUMNS])
        for index in range(n):
            writer.writerow([index] + [
                series.get(metric, [""] * n)[index] for _, metric in _CSV_COLUMNS])
    return out_csv


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


def run_live(args) -> List[dict]:
    """Run the calls through Ray with the profiler collecting; return the per-call rows."""
    import ray

    from chia.base.ChiaFunction import get
    from chia.models.bedrock_config import bedrock_model_env
    from chia.models.claude import ClaudeCodeLLM
    from chia.trace.profiler import get_collector, start_collector, stop_collector

    questions = QUESTIONS[:args.calls]
    if len(questions) < args.calls:
        raise SystemExit(f"only {len(QUESTIONS)} distinct questions are defined; "
                         f"--calls {args.calls} would have to repeat one")

    overlay = bedrock_model_env(primary=args.model, region=args.region, require_tools=False)
    os.environ.update(overlay)

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
            # `prompt` is gated on the `claude_creds` custom resource -- that gate is how
            # chia stops N workers from stampeding one credential. A local cluster
            # advertises no custom resources, so without this the task queues forever
            # instead of failing, which looks like a hang and is really a missing gate.
            resources={"claude_creds": 1},
            # The overlay has to reach the *worker*: the CLI subprocess is launched there,
            # and env vars set on the driver do not cross the boundary on their own.
            runtime_env={"env_vars": overlay},
        )

    # The collector's log dir goes inside the run dir, so a run carries its own raw trace
    # next to the folded record aet reads.
    start_collector(str(args.run_dir / "profiler"))

    llm = ClaudeCodeLLM(
        model=args.model,
        system_message=system_message(args.nonce),
        # None, not the default: the default is a cluster path
        # (/home/ray/.claude/projects/...) that does not exist on a local run.
        projects_cwd=None,
        log_dir=str(args.run_dir / "cli_logs"),
        # True, and not for the logging. The streaming path is the one that reads the CLI's
        # `stream-json` output, which is where the per-turn token counts live; the plain
        # `--print` path returns text and no usage at all, so a run with log_stream=False
        # produces a trajectory of zeros. That is not a knob here, it is a requirement.
        log_stream=True,
        retries=2,
    )

    rows: List[dict] = []
    try:
        for index, question in enumerate(questions):
            # Serial, not fanned out. The cold-then-warm shape only exists if the first
            # call has finished writing the cache before the second asks for it;
            # dispatching all N at once would race them into N cache writes.
            started = time.time()
            result = get(llm.prompt.chia_remote(llm, question, []))
            usage = result.usage
            print(f"[{index + 1}/{len(questions)}] {time.time() - started:5.1f}s  "
                  f"in={usage.input_tokens} out={usage.output_tokens} "
                  f"write={usage.cache_creation_input_tokens} "
                  f"read={usage.cache_read_input_tokens} "
                  f"cost={usage.cost_usd} ({usage.cost_source})")
    finally:
        # Read the events *before* stopping: stop_collector kills the actor. The sink's own
        # flush also happens inside stop_collector (that is the documented path, and it is
        # what writes the run dir when CHIA_AET_SINK is set), so this ordering matters.
        collector = get_collector()
        events = ray.get(collector.get_events.remote()) if collector is not None else []
        stop_collector()

    for event in events:
        row = event_to_row(event)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: row["ts"])
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce a chia run directory that `aet plot` can draw.")
    parser.add_argument("--calls", type=int, default=8,
                        help="How many LLM calls to make (default: 8)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Bedrock model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--region", default="us-west-2",
                        help="AWS region (default: us-west-2)")
    parser.add_argument("--run-dir", type=Path, default=Path("aet_run"),
                        help="Where to write the aet run directory (default: ./aet_run)")
    parser.add_argument("--run-id", default="",
                        help="Run id (default: derived from the model)")
    parser.add_argument("--calls-log", type=Path, default=None,
                        help="Where to write the per-call log (default: <run-dir>/calls.jsonl)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Where to write the figure's source data "
                             "(default: figures/trajectory.csv)")
    parser.add_argument("--replay", type=Path, default=None,
                        help="Rebuild the run dir from a committed per-call log. No calls, $0.")
    parser.add_argument("--cap-usd", type=float, default=2.0,
                        help="Refuse to dispatch if the projection crosses this (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would run and what it would cost. No calls.")
    parser.add_argument("--nonce", default="",
                        help="Tag for the cached prefix (default: fresh). Pass a fixed value "
                             "to deliberately reuse another run's warm cache.")
    args = parser.parse_args(argv)
    args.nonce = args.nonce or uuid.uuid4().hex[:12]

    args.run_dir = args.run_dir.resolve()
    args.csv = args.csv or (HERE / "figures" / "trajectory.csv")
    run_id = args.run_id or f"chia-{args.model.split('.')[-1].split('-v1')[0]}"

    if args.dry_run:
        print(f"would make {args.calls} calls to {args.model} in {args.region}")
        print(f"would write {args.run_dir}")
        try:
            status = check_budget(args.cap_usd, (), projected_calls=args.calls)
        except BudgetExceeded as exc:
            print(f"budget: REFUSED -- {exc}")
            return 1
        print(f"budget: allowed (per-call estimate: {status.per_call_usd})")
        return 0

    if args.replay is not None:
        rows = read_calls(args.replay)
        print(f"replaying {len(rows)} recorded calls from {args.replay} -- no calls, $0")
    else:
        # Checked before dispatch, not after: a projection consulted once the money is gone
        # is not a cap. With no recorded history the estimate is unavailable and the run is
        # allowed; check_budget reports that on the status rather than inventing a number.
        try:
            check_budget(args.cap_usd, (), projected_calls=args.calls)
        except BudgetExceeded as exc:
            print(f"refusing to dispatch: {exc}")
            return 1
        rows = run_live(args)

    if not rows:
        print("no LLM-call usage was recorded; nothing to write")
        return 1

    calls_log = args.calls_log or (args.run_dir / "calls.jsonl")
    write_calls(calls_log, rows)

    if not record([row_to_event(row) for row in rows], args.run_dir, run_id=run_id):
        print("the aet sink wrote nothing -- is aet installed? (pip install 'chia[aet]')")
        return 1

    csv_path = write_source_data(args.run_dir, args.csv)
    series = trajectory_series(args.run_dir)
    reads = series.get("aet.traj.cum_cache_read_tokens", [0.0])[-1]
    writes = series.get("aet.traj.cum_cache_creation_tokens", [0.0])[-1]

    print(f"\nwrote {args.run_dir}")
    print(f"wrote {calls_log}  (replay this for $0)")
    print(f"wrote {csv_path}  (the figure's numbers)")
    print(f"\n{len(rows)} calls: {writes:,.0f} cache tokens written, {reads:,.0f} read")
    if not writes and not reads:
        # Said out loud rather than left for the figure to imply. Two lines flat at zero
        # look like a measurement of "no cache activity"; here it would mean the prompt
        # never got cached, and the split figure would be making a claim it cannot support.
        print("WARNING: no cache activity was recorded, so --split-cache has nothing to "
              "draw. Either the system prompt fell below the provider's minimum cacheable "
              "length or caching is off for this model.")
    print("\nnow render it, from aet's venv:")
    print(f"  aet plot {args.run_dir} --kind trajectory --split-cache \\\n"
          f"      --out {HERE / 'figures' / 'trajectory.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
