"""Does the ``ModelTier`` primary lever change anything measurable, at a fixed harness and task?

``chia.models.bedrock_config.ModelTier`` is the one feature in the study's set with no
measurement behind it. It is a dataclass with three fields and a ``validate``, and everything
about it up to now has been an argument that a tier mix is useful rather than evidence that
it is. This closes that, with one deliberately narrow question.

Why the question is narrow, and what that costs
-----------------------------------------------

``ModelTier`` has three levers. Only one of them can be measured here:

``primary``
    Sets the CLI's ``--model``. Measurable: it changes which model answers, so cost and
    correctness both move, and both are recorded per call.
``subagent`` / ``background``
    Set ``CLAUDE_CODE_SUBAGENT_MODEL`` and ``ANTHROPIC_SMALL_FAST_MODEL``. **Not measured, and
    the reason is a real gap rather than a scope choice.** Two things would have to be true
    and neither is: the task would have to delegate (this one is a single-turn code question
    with no tools, so a subagent model is never invoked and varying it would be inert by
    construction), and chia would have to record *per-model* usage from the CLI's stream so
    the routing could be observed. It does not. ``ClaudeCodeLLM``'s ``_last_metadata`` carries
    one ``model`` field and one set of token counts, folded across every model the CLI called
    — so even on a delegating task, chia's own telemetry cannot say which tier served which
    call. ``aet.tracking.claude_stream.ModelUsage`` exists for exactly this and chia does not
    populate it.

That is stated up front because the alternative was to run a tier-mix arm anyway, watch it
come out identical, and report "no detectable effect from the subagent lever". That sentence
would be true and worthless: nothing would have exercised the lever, so the null measures the
task, not the feature. A null result is only evidence when the thing varied was actually used.

The design
----------

One factor, two levels, everything else held fixed. Same harness (``cli_native``), same task
and oracle as ``grid.py``, same region, same fresh temp cwd per cell. Only
``ModelTier(primary=...)`` changes.

An impossible mix is refused before dispatch by ``ModelTier.validate``, which is the other
thing worth demonstrating: a tier that cannot work is a startup error rather than a confusing
mid-run failure.

Usage::

    python examples/harness_study/tier_ablation.py --dry-run
    python examples/harness_study/tier_ablation.py --repeats 5 --cap-usd 5
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from chia.base.budget import BudgetExceeded, check_budget  # noqa: E402
from chia.base.usage import TokenUsage  # noqa: E402

from grid import PROMPT, grade  # noqa: E402  — same task and oracle, deliberately

#: The arms. Aliases resolved through the model registry, so an unknown one fails here rather
#: than as an opaque Bedrock ValidationException halfway through a grid.
#:
#: Both are Anthropic, because the harness is held fixed at the CLI's *native* Bedrock
#: transport and that transport cannot speak to a non-Anthropic model. Adding a Nova arm would
#: mean changing the harness at the same time as the tier, which confounds the one factor this
#: is varying. The proxy exists to remove that constraint and its own evidence is in
#: ``README.md``; mixing the two questions into one grid would answer neither.
ARMS = ("sonnet", "haiku")

COLUMNS = (
    "arm", "primary", "repeat", "passed", "wall_s", "cost_usd", "cost_source",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens",
    "num_turns", "harness_failed", "error",
)


@dataclass
class Row:
    """One cell."""

    arm: str
    primary: str
    repeat: int
    passed: Optional[bool] = None
    wall_s: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    harness_failed: bool = False
    error: str = ""

    def as_csv(self) -> dict:
        u = self.usage
        return {
            "arm": self.arm, "primary": self.primary, "repeat": self.repeat,
            "passed": "" if self.passed is None else int(self.passed),
            "wall_s": round(self.wall_s, 3),
            "cost_usd": "" if u.cost_usd is None else round(u.cost_usd, 6),
            "cost_source": u.cost_source,
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_input_tokens,
            "cache_creation_tokens": u.cache_creation_input_tokens,
            "num_turns": u.num_turns,
            "harness_failed": int(self.harness_failed),
            "error": self.error[:300],
        }


def resolve_arm(alias: str, *, require_tools: bool = False) -> str:
    """The concrete Bedrock id for *alias*, refusing an impossible tier up front.

    :param alias: A registry alias or a raw Bedrock id.
    :param require_tools: Leave ``False`` for this single-turn task; ``True`` for anything
        agentic.
    :type alias: str
    :type require_tools: bool
    :rtype: str
    :raises chia.models.bedrock_config.TierError: The mix cannot work as configured.

    Validation happens here, before any dispatch, which is the point of having it: a tier that
    cannot work should be a startup error naming the offending tier, not a Bedrock
    ``ValidationException`` surfacing on cell 7 of 10.
    """
    from chia.models.bedrock_config import ModelTier

    tier = ModelTier(primary=alias)
    tier.validate(require_tools=require_tools)
    return tier.resolved()["primary"]


def run_cell(arm: str, repeat: int, *, region: str) -> Row:
    """Run one cell. Never raises: a failed cell is a recorded row, not a lost grid."""
    row = Row(arm=arm, primary="", repeat=repeat)
    try:
        row.primary = resolve_arm(arm)
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        row.harness_failed = True
        return row

    from chia.models.bedrock_config import bedrock_model_env
    from chia.models.claude import ClaudeCodeLLM

    overlay = bedrock_model_env(primary=row.primary, region=region, require_tools=False)
    previous_env = {key: os.environ.get(key) for key in overlay}
    # A fresh cwd per cell: the CLI has file-writing tools and the prompt asks for code, so a
    # shared directory lets cell N read what cell N-1 wrote -- which would make the later arm
    # look better than it is. An earlier run of grid.py left a collapse.py in the repo root.
    workspace = tempfile.mkdtemp(prefix=f"tier_{arm}_{repeat}_")
    previous_cwd = os.getcwd()
    started = time.perf_counter()
    try:
        os.environ.update(overlay)
        os.chdir(workspace)
        llm = ClaudeCodeLLM(model=row.primary, projects_cwd=None, log_stream=True, retries=2)
        result = llm.prompt(PROMPT, tools=[])
        row.wall_s = time.perf_counter() - started
        row.usage = result.usage
        text = (result.result or "").strip()
        if not text:
            # A harness that produced nothing did not answer wrongly -- it did not answer.
            # Grading an empty string as a failure would report an auth or config problem as a
            # quality result.
            row.error = "harness produced no output"
            row.harness_failed = True
        else:
            row.passed = grade(text)
    except Exception as exc:
        row.wall_s = time.perf_counter() - started
        row.error = f"{type(exc).__name__}: {exc}"
        row.harness_failed = True
    finally:
        os.chdir(previous_cwd)
        shutil.rmtree(workspace, ignore_errors=True)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return row


def _write(rows: List[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv())


def summarise(rows: List[Row]) -> None:
    """Per-arm pass rate and spend, every proportion carrying its denominator.

    Intervals rather than point estimates on the pass rate, because at these n a bare
    percentage invites a conclusion the sample cannot support. A range that spans both arms is
    reported as spanning both arms.
    """
    print("\narm      n   passed        mean $/call     total $   mean wall")
    for arm in sorted({row.arm for row in rows}):
        arm_rows = [row for row in rows if row.arm == arm]
        graded = [row for row in arm_rows if row.passed is not None]
        priced = [row.usage.cost_usd for row in arm_rows if row.usage.cost_usd is not None]
        n_pass = sum(1 for row in graded if row.passed)
        rate = f"{n_pass}/{len(graded)}" if graded else "-/0"
        mean_cost = statistics.fmean(priced) if priced else float("nan")
        walls = [row.wall_s for row in arm_rows if row.wall_s]
        print(f"{arm:8} {len(arm_rows):<3} {rate:<13} ${mean_cost:<14.5f} "
              f"${sum(priced):<9.4f} {statistics.fmean(walls) if walls else 0:.1f}s")

    failed = [row for row in rows if row.harness_failed]
    if failed:
        print(f"\n{len(failed)} harness failure(s) -- excluded from the pass rates above, "
              f"because a harness that did not answer is not a wrong answer:")
        for row in failed[:5]:
            print(f"  {row.arm} rep{row.repeat}: {row.error[:100]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--cap-usd", type=float, default=5.0,
                        help="metered-spend ceiling; the grid is refused if it would project "
                             "over this")
    parser.add_argument("--per-call-usd", type=float, default=0.10,
                        help="projection's per-call estimate. Defaults to a deliberately "
                             "pessimistic figure -- the observed sonnet cold-cache call.")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "tier_ablation.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cells = [(arm, repeat) for arm in args.arms for repeat in range(args.repeats)]

    # Resolved and validated before anything is dispatched, so an impossible tier is a startup
    # error naming the offending arm rather than a failure on cell 7.
    for arm in args.arms:
        try:
            print(f"  {arm:8} -> {resolve_arm(arm)}")
        except Exception as exc:
            print(f"REFUSED: arm {arm!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    try:
        status = check_budget(args.cap_usd, (), projected_calls=len(cells),
                             per_call_usd=args.per_call_usd)
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    # The *effective* rate, which is the override when one is passed. `status.per_call_usd` is
    # the median of recorded history and stays None on a first run -- printing that would have
    # reported a projection of $0.00 for a grid whose projection had in fact been checked
    # against $0.10/call.
    rate = args.per_call_usd if args.per_call_usd is not None else status.per_call_usd
    source = "override" if args.per_call_usd is not None else \
        f"median of {status.priced_calls} recorded"
    print(f"{len(cells)} cells, cap ${args.cap_usd:.2f}, "
          + (f"projecting ~${rate * len(cells):.2f} at ${rate:.4f}/call ({source})"
             if rate is not None else "no history to project from"))

    if args.dry_run:
        for arm, repeat in cells:
            print(f"  [live] {arm:8} rep{repeat}")
        return 0

    rows: List[Row] = []
    for index, (arm, repeat) in enumerate(cells, 1):
        row = run_cell(arm, repeat, region=args.region)
        rows.append(row)
        verdict = ("HARNESS-FAILED" if row.harness_failed
                   else "PASS" if row.passed else "fail")
        cost = "" if row.usage.cost_usd is None else f" ${row.usage.cost_usd:.4f}"
        print(f"[{index}/{len(cells)}] {arm:8} rep{repeat} -> {verdict}{cost} "
              f"{row.wall_s:.1f}s {row.error[:70]}")
        _write(rows, args.out)

    summarise(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
