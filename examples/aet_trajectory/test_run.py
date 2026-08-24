"""Tests for the aet-trajectory example.

Two things are worth pinning here, and neither is about the LLM. The first is the
reproduction claim: the README says anyone can rebuild the committed figures from
``data/*.jsonl`` with no credentials and no spend, and a claim like that is worthless
unless something checks it. The second is that the committed figures' source CSV is
actually the numbers in the run directory, rather than a file that drifted away from it.

No network, no Ray, no credentials -- everything runs off the two committed per-call logs.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

pytest.importorskip("aet", reason="aet not installed; install chia[aet]")

import run as example  # noqa: E402

COLD_CALLS = HERE / "data" / "cold_calls.jsonl"
WARM_CALLS = HERE / "data" / "warm_calls.jsonl"
COLD_CSV = HERE / "figures" / "trajectory.csv"


def _rebuild(calls: Path, into: Path, run_id: str) -> dict:
    rows = example.read_calls(calls)
    assert example.record([example.row_to_event(row) for row in rows], into, run_id=run_id)
    return example.trajectory_series(into)


# --------------------------------------------------------------------------- reproduction
def test_the_committed_log_rebuilds_the_committed_csv(tmp_path):
    """The reproduction claim in the README, asserted rather than asserted-in-prose.

    Exact equality, not a tolerance: the committed rows carry the costs the backend
    reported, so nothing along this path re-derives a number from a rate table that could
    drift. If a tolerance were ever needed here, something started recomputing.
    """
    series = _rebuild(COLD_CALLS, tmp_path / "cold", "chia-cold")

    with COLD_CSV.open() as handle:
        committed = list(csv.DictReader(handle))

    assert len(committed) == len(series["aet.traj.t_s"])
    for index, row in enumerate(committed):
        for column, metric in example._CSV_COLUMNS:
            assert float(row[column]) == pytest.approx(series[metric][index], abs=0.0), \
                f"point {index}, column {column}"


def test_the_two_arms_differ_only_in_which_cache_class_they_used(tmp_path):
    """The finding the figures report, checked against the data rather than the caption.

    Both arms make the same eight calls with the same prompts and end up within a few dozen
    tokens of each other. The cold arm pays roughly 2.4x as much, and every bit of that gap
    is cache writes -- which is exactly the difference a two-class (input/output) accounting
    model cannot see, and the reason the split has to survive into the log.
    """
    cold = _rebuild(COLD_CALLS, tmp_path / "cold", "chia-cold")
    warm = _rebuild(WARM_CALLS, tmp_path / "warm", "chia-warm")

    def total(series):
        return (series["aet.traj.cum_input_tokens"][-1]
                + series["aet.traj.cum_output_tokens"][-1]
                + series["aet.traj.cum_cache_tokens"][-1])

    cold_total, warm_total = total(cold), total(warm)
    # Within 1%: the arms are the same workload, so the token counts are not the story.
    assert abs(cold_total - warm_total) / cold_total < 0.01

    cold_cost = cold["aet.traj.cum_cost_usd"][-1]
    warm_cost = warm["aet.traj.cum_cost_usd"][-1]
    assert cold_cost > 2.0 * warm_cost

    # The whole gap is cache writes: the warm arm has none, the cold arm has all of them.
    assert warm["aet.traj.cum_cache_creation_tokens"][-1] == 0
    assert cold["aet.traj.cum_cache_creation_tokens"][-1] > 0
    # ...and the warm arm reads more, not less -- it is not cheaper by doing less work.
    assert warm["aet.traj.cum_cache_read_tokens"][-1] > cold["aet.traj.cum_cache_read_tokens"][-1]


# --------------------------------------------------------------------------- the run dir
def test_the_rebuilt_run_dir_is_a_complete_aet_run(tmp_path):
    """What ``aet plot`` needs, present: points, a round, and the summary param.

    Named individually rather than checked as "the directory exists", because the sink used
    to write points and neither of the other two, and the figure that came out of *that*
    titled itself "0 rounds - 0 min" while looking perfectly finished.
    """
    from aet.trajectory.model import RunTrajectory

    into = tmp_path / "cold"
    _rebuild(COLD_CALLS, into, "chia-cold")

    assert (into / "run_record.json").is_file()
    traj = RunTrajectory.from_run_dir(into)
    assert len(traj.points) == 8
    assert traj.num_rounds == 1
    assert traj.duration_s > 0.0
    assert traj.run_id == "chia-cold"
    # the split survived into the reconstruction, which is the whole point of the branch
    assert traj.points[-1].cum_cache_read_tokens > 0
    assert traj.points[-1].cum_cache_creation_tokens > 0


def test_the_committed_logs_carry_no_prompts_or_credentials():
    """A per-call log is committed, so it has to be safe to commit.

    It holds counts and costs and nothing else -- no prompt text, no responses, no
    environment. Checked by field name rather than by scanning for secrets: an allowlist
    fails closed when someone adds a field, and a scan only fails once a secret is already
    in the file.
    """
    allowed = {"call_id", "func", "ts"} | set(example._USAGE_KEYS)
    for path in (COLD_CALLS, WARM_CALLS):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            extra = set(json.loads(line)) - allowed
            assert not extra, f"{path.name} carries unexpected fields: {sorted(extra)}"
