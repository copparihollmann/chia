"""Readers that turn recorded telemetry into the tidy CSVs the figures read.

Three sources, one row-per-LLM-run shape:

* **A chia profiler log** — the JSONL the collector actor writes. Folded with
  :func:`chia.trace.aet_sink.collect_run_usage`, so the CSV and the aet sink agree
  by construction rather than by two parallel readers.
* **aet run directories** — ``logs/metrics.jsonl`` per run, read under aet's own
  last-occurrence-wins rule. This is where the already-recorded evidence lives.
* **A spec-style ``results.jsonl``** — one row per experiment cell, carrying the four
  token classes plus the cost that was actually reported at the time.

Every reader emits the same columns, so the figures never need to know which source
a row came from::

    source, run_id, model, input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, cache_unsplit_tokens, reported_cost_usd, billing_mode,
    retries, calls

Run it as a script to (re)generate the committed CSVs; ``--help`` lists the paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

#: The one row shape every reader produces.
COLUMNS = (
    "source",
    "run_id",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "cache_unsplit_tokens",
    "reported_cost_usd",
    "billing_mode",
    "retries",
    "calls",
)

# aet's metric-name contract, shared with `aet runs` and `aet.trajectory.rollup`.
_M_COST = "aet.agent.cost_usd"
_M_INPUT = "gen_ai.usage.input_tokens"
_M_OUTPUT = "gen_ai.usage.output_tokens"
_M_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
_M_CACHE_CREATE = "gen_ai.usage.cache_creation.input_tokens"


def _read_jsonl(path: Path) -> Iterator[dict]:
    """Yield each parseable JSON object in *path*, skipping junk lines."""
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def _num(value) -> Optional[float]:
    """*value* as a float when it is genuinely numeric, else ``None``.

    ``bool`` is excluded deliberately: it is an ``int`` subclass, so a stray flag
    would otherwise be read as a token count of 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Source 1 — a chia profiler log
# ---------------------------------------------------------------------------


def rows_from_chia_profile(path: Path, run_id: str = "") -> List[dict]:
    """One row per billing mode found in a chia profiler JSONL.

    :param path: The collector's ``.log`` file.
    :param run_id: Row label; defaults to the file's stem.
    :type path: Path
    :type run_id: str
    :rtype: List[dict]

    Folding is delegated to :func:`chia.trace.aet_sink.collect_run_usage` rather than
    re-walked here, so a change to what counts as a call cannot make the CSV and the
    aet sink disagree.
    """
    from chia.trace.aet_sink import collect_run_usage

    run = collect_run_usage(_read_jsonl(path))
    rows = []
    for mode, usage in run.by_billing_mode.items():
        rows.append({
            "source": "chia_profile",
            "run_id": run_id or path.stem,
            "model": usage.model or run.dominant_model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "cache_creation_tokens": usage.cache_creation_input_tokens,
            "reported_cost_usd": usage.cost_usd if usage.cost_usd is not None else "",
            "billing_mode": mode,
            "retries": run.retries,
            "calls": len([c for c in run.calls if c.usage.billing_mode == mode]),
        })
    return rows


# ---------------------------------------------------------------------------
# Source 2 — aet run directories
# ---------------------------------------------------------------------------


def rows_from_aet_runs(root: Path) -> List[dict]:
    """One row per aet run directory found beneath *root*.

    :param root: A directory containing (or containing directories containing) runs.
    :type root: Path
    :rtype: List[dict]

    A run is any directory with ``logs/metrics.jsonl``. Scalars are read under aet's
    own rule — last occurrence of a name wins — because that is the rule the numbers
    were written under; reading them any other way would measure the reader.
    """
    rows = []
    seen: set = set()
    for metrics_path in sorted(root.rglob("logs/metrics.jsonl")):
        run_dir = metrics_path.parent.parent
        if str(run_dir) in seen:
            continue
        final: Dict[str, float] = {}
        for record in _read_jsonl(metrics_path):
            name, value = record.get("name"), _num(record.get("value"))
            if name and value is not None and record.get("step") is None:
                final[name] = value
        if any(k in final for k in (_M_INPUT, _M_OUTPUT)):
            row = {
                "input_tokens": int(final.get(_M_INPUT, 0)),
                "output_tokens": int(final.get(_M_OUTPUT, 0)),
                "cache_read_tokens": int(final.get(_M_CACHE_READ, 0)),
                "cache_creation_tokens": int(final.get(_M_CACHE_CREATE, 0)),
                "reported_cost_usd": final.get(_M_COST, ""),
                "model": _aet_run_model(run_dir),
                "calls": 1,
            }
        else:
            # Same fallback aet's own rollup uses: a run may carry only the
            # canonical trajectory artifact and no final scalars. Skipping those
            # would silently drop most of an existing corpus.
            row = _row_from_trajectory(run_dir)
            if row is None:
                continue
        seen.add(str(run_dir))
        row.update({"source": "aet_run", "run_id": run_dir.name,
                    "billing_mode": "", "retries": ""})
        rows.append(row)
    return rows


def _row_from_trajectory(run_dir: Path) -> Optional[dict]:
    """A row from ``metrics/trajectory.json``, or ``None`` when there is none."""
    try:
        traj = json.loads((run_dir / "metrics" / "trajectory.json").read_text())
    except Exception:
        return None
    if not isinstance(traj, dict):
        return None
    read = _num(traj.get("final_cache_read_tokens"))
    created = _num(traj.get("final_cache_creation_tokens"))
    merged = _num(traj.get("final_cache_tokens")) or 0.0
    return {
        "model": str(traj.get("model", "") or ""),
        "input_tokens": int(_num(traj.get("final_input_tokens")) or 0),
        "output_tokens": int(_num(traj.get("final_output_tokens")) or 0),
        "cache_read_tokens": int(read or 0),
        "cache_creation_tokens": int(created or 0),
        # A trajectory that recorded only the merged cache figure cannot be priced
        # correctly either — reads and writes differ by more than 10x — so the
        # unsplit total is carried in its own column instead of being guessed into
        # one of the two classes. Figures that need the split exclude these rows and
        # say so.
        "cache_unsplit_tokens": (int(merged) if read is None and created is None
                                 else ""),
        "reported_cost_usd": (traj["final_cost_usd"]
                              if _num(traj.get("final_cost_usd")) is not None else ""),
        "calls": int(_num(traj.get("num_rounds")) or 1),
    }


def _aet_run_model(run_dir: Path) -> str:
    """The model id an aet run recorded, from params.json or run_record.json."""
    params = run_dir / "logs" / "params.json"
    try:
        data = json.loads(params.read_text())
        model = data.get("gen_ai.response.model") or data.get("gen_ai.request.model")
        if model:
            return str(model)
    except Exception:
        pass
    try:
        record = json.loads((run_dir / "run_record.json").read_text())
        return str(record.get("model", "") or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Source 3 — a spec-style results.jsonl
# ---------------------------------------------------------------------------


def rows_from_results_jsonl(paths: Iterable[Path]) -> List[dict]:
    """One row per experiment cell that recorded token counts.

    :param paths: ``results.jsonl`` files, each row one cell of a grid.
    :type paths: Iterable[Path]
    :rtype: List[dict]

    Only identity, counts, cost and retries are carried across. Free-text failure
    detail, prompt paths and target names are deliberately dropped: the figures do
    not need them, and a committed CSV should carry no more of an experiment than the
    claim it supports.
    """
    rows = []
    for path in paths:
        for record in _read_jsonl(path):
            counts = {
                key: _num(record.get(key))
                for key in ("input_tokens", "output_tokens",
                            "cache_read_tokens", "cache_creation_tokens")
            }
            if not any(counts.values()):
                continue
            models = str(record.get("models_used") or "")
            rows.append({
                "source": "results_jsonl",
                "run_id": str(record.get("experiment_id", "") or ""),
                # A cell that used several models is labelled with the first; the
                # rate table resolves families, so a mixed cell is priced by its
                # orchestrator. Flagged rather than silently averaged.
                "model": models.split(",")[0],
                "input_tokens": int(counts["input_tokens"] or 0),
                "output_tokens": int(counts["output_tokens"] or 0),
                "cache_read_tokens": int(counts["cache_read_tokens"] or 0),
                "cache_creation_tokens": int(counts["cache_creation_tokens"] or 0),
                "reported_cost_usd": (record.get("cost_usd")
                                      if _num(record.get("cost_usd")) is not None
                                      else ""),
                "billing_mode": str(record.get("billing_mode", "") or ""),
                "retries": (int(record["retries"])
                            if _num(record.get("retries")) is not None else ""),
                "calls": int(_num(record.get("round_count")) or 1),
            })
    return rows


#: Row shape for the retry-visibility CSV.
RETRY_COLUMNS = ("source", "run_id", "retries", "status", "has_token_counts")


def retry_rows_from_results_jsonl(paths: Iterable[Path]) -> List[dict]:
    """One row per experiment cell that recorded a ``retries`` count.

    :param paths: ``results.jsonl`` files.
    :type paths: Iterable[Path]
    :rtype: List[dict]

    Deliberately a *wider* net than :func:`rows_from_results_jsonl`: a cell that
    retried and then failed outright has no token counts to carry, but it is exactly
    the case the retry figure is about. Filtering to token-bearing rows would drop
    most of the retries and make the incidence look far rarer than it was.
    """
    rows = []
    for path in paths:
        for record in _read_jsonl(path):
            retries = _num(record.get("retries"))
            if retries is None:
                continue
            has_counts = any(_num(record.get(key)) for key in
                             ("input_tokens", "output_tokens",
                              "cache_read_tokens", "cache_creation_tokens"))
            rows.append({
                "source": "results_jsonl",
                "run_id": str(record.get("experiment_id", "") or ""),
                "retries": int(retries),
                "status": str(record.get("status", "") or ""),
                "has_token_counts": "yes" if has_counts else "no",
            })
    return rows


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_csv(rows: List[dict], path: Path,
              columns: Iterable[str] = COLUMNS) -> Path:
    """Write *rows* to *path* using *columns*, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return path


def read_csv(path: Path) -> List[dict]:
    """Read a CSV written by :func:`write_csv`, coercing the numeric columns.

    Empty numeric cells stay ``None`` rather than becoming ``0`` — the distinction
    between "no cost recorded" and "cost was zero" is the whole point.
    """
    numeric = {"input_tokens", "output_tokens", "cache_read_tokens",
               "cache_creation_tokens", "cache_unsplit_tokens",
               "reported_cost_usd", "retries", "calls"}
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in numeric:
                value = (row.get(key) or "").strip()
                row[key] = float(value) if value else None
            rows.append(row)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "data" / "runs.csv",
                        help="destination CSV (default: data/runs.csv)")
    parser.add_argument("--chia-profile", type=Path, action="append", default=[],
                        help="a chia profiler JSONL; repeatable")
    parser.add_argument("--aet-runs", type=Path, action="append", default=[],
                        help="a root containing aet run directories; repeatable")
    parser.add_argument("--results-jsonl", type=Path, action="append", default=[],
                        help="a spec-style results.jsonl; repeatable")
    parser.add_argument("--retries-out", type=Path,
                        default=Path(__file__).parent / "data" / "retries.csv",
                        help="destination for the retry-incidence CSV")
    args = parser.parse_args(argv)

    rows: List[dict] = []
    for path in args.chia_profile:
        rows += rows_from_chia_profile(path)
    for root in args.aet_runs:
        rows += rows_from_aet_runs(root)
    if args.results_jsonl:
        rows += rows_from_results_jsonl(args.results_jsonl)

    if not rows:
        parser.error("no rows collected; pass at least one source")
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")

    if args.results_jsonl:
        retry_rows = retry_rows_from_results_jsonl(args.results_jsonl)
        write_csv(retry_rows, args.retries_out, RETRY_COLUMNS)
        print(f"wrote {len(retry_rows)} rows to {args.retries_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
