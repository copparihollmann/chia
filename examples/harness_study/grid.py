"""The harness x model factorial: does the harness matter at a fixed model?

The question a reviewer asks about chia's Converse proxy is not "can you run GLM-5?" —
it obviously can — but **"why not just use opencode?"** opencode already drives any
provider. So the experiment holds the *model* fixed and varies the *harness*:

======================  ====================================================
Harness                 What it is
======================  ====================================================
``converse``            chia's own ``BedrockLLM`` — the Converse API directly.
``cli_native``          the Claude Code CLI on its native Bedrock transport.
``cli_proxy``           the Claude Code CLI through ``chia.models.proxy``.
``opencode``            the opencode CLI, provider-agnostic already.
======================  ====================================================

``sonnet`` is the bridge cell: it runs under all four, so a difference between rows at
that model is a harness difference and nothing else. A non-Anthropic model can only reach
``cli_native``'s column as a hole, and the hole is the finding — that column is exactly
what the proxy fills.

**This plot is allowed to come out against the proxy.** If the CLI harness adds nothing
over opencode at equal model, the honest contribution is a documented negative result and
a recommendation to use opencode — which still makes you the person who settled it.

The task
--------

One prompt, a deterministic oracle, no tools and no hardware: write a Python function,
and the grader runs it against hidden assertions. Deliberately not chia's ``memcpy``
example even though the plan named it — memcpy needs a chipyard container and verilator,
and a harness comparison does not need a hardware oracle to be valid. What it does need
is a grader that cannot be argued with, which ``exec`` plus assertions is.

Cost control
------------

``check_budget`` runs **before** any dispatch, projecting the whole grid from recorded
per-call spend (:mod:`chia.base.budget`). A grid that would cross the cap is refused
rather than discovered mid-run.

Usage::

    # See what it would do, and what it would cost. No calls.
    python examples/harness_study/grid.py --dry-run

    # Run it, refusing to exceed $20 of metered spend.
    python examples/harness_study/grid.py --cap-usd 20 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from chia.base.budget import BudgetExceeded, check_budget  # noqa: E402
from chia.base.usage import TokenUsage  # noqa: E402

HARNESSES = ("converse", "cli_native", "cli_proxy", "opencode")

#: ``(alias, bedrock_id, opencode_id)``. ``opencode_id`` is ``None`` where opencode has
#: no route to that model, which is itself a cell of the grid rather than an omission.
MODELS = {
    "sonnet": ("us.anthropic.claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
    "nova-lite": ("us.amazon.nova-lite-v1:0", None),
    "glm5": ("zai.glm-5", None),
}

COLUMNS = (
    "harness", "model", "repeat", "passed", "wall_s", "cost_usd", "cost_source",
    "billing_mode", "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_creation_tokens", "num_turns", "retries", "harness_failed",
    "skipped_reason", "error",
)


# ---------------------------------------------------------------------------
# The task and its oracle
# ---------------------------------------------------------------------------

PROMPT = """Write a single Python function with this exact signature:

    def collapse(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:

It takes a list of closed integer intervals (start, end) with start <= end, and returns
them merged: overlapping or exactly-adjacent intervals combined into one, sorted by
start. Two intervals are adjacent when one ends exactly where the next begins, or one
ends exactly one before the next begins (so (1,3) and (4,6) merge into (1,6)).

Reply with ONLY the function in a single ```python code block. No explanation."""

#: The oracle. Adjacency-by-one is the interesting case: a model that only handles
#: overlap passes the first three and fails the fourth, which is what makes this
#: discriminating rather than a formality.
ORACLE_CASES = [
    ([], []),
    ([(1, 3)], [(1, 3)]),
    ([(1, 3), (2, 6)], [(1, 6)]),
    ([(1, 3), (4, 6)], [(1, 6)]),
    ([(5, 6), (1, 3)], [(1, 3), (5, 6)]),
    ([(1, 10), (2, 3), (4, 5)], [(1, 10)]),
    ([(1, 2), (5, 6), (3, 4)], [(1, 6)]),
]

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def grade(response: str) -> bool:
    """Whether *response* contains a ``collapse`` that satisfies every oracle case.

    :param response: The model's raw reply.
    :type response: str
    :rtype: bool

    Runs the extracted code with ``exec`` in an empty namespace. That is acceptable here
    and nowhere else: the input is a model's answer to a prompt this file wrote, the
    grader is a local script, and the alternative (a parser that decides what the code
    *would* do) is a second implementation of Python.
    """
    match = _CODE_BLOCK.search(response or "")
    code = match.group(1) if match else (response or "")
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102 - see the docstring
        fn = namespace.get("collapse")
        if not callable(fn):
            return False
        for given, expected in ORACLE_CASES:
            if [tuple(x) for x in fn(list(given))] != expected:
                return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# One row per harness
# ---------------------------------------------------------------------------


@dataclass
class Row:
    harness: str
    model: str
    repeat: int
    passed: Optional[bool] = None
    wall_s: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    skipped_reason: str = ""
    error: str = ""
    harness_failed: bool = False

    def as_csv(self) -> dict:
        u = self.usage
        return {
            "harness": self.harness, "model": self.model, "repeat": self.repeat,
            "passed": "" if self.passed is None else int(self.passed),
            "wall_s": round(self.wall_s, 3),
            "cost_usd": "" if u.cost_usd is None else round(u.cost_usd, 6),
            "cost_source": u.cost_source, "billing_mode": u.billing_mode,
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_input_tokens,
            "cache_creation_tokens": u.cache_creation_input_tokens,
            "num_turns": u.num_turns, "retries": len(getattr(self, "_retries", ())),
            "harness_failed": int(self.harness_failed),
            "skipped_reason": self.skipped_reason, "error": self.error[:300],
        }


def _run_converse(model_id: str, region: str) -> tuple[str, TokenUsage]:
    """chia's own BedrockLLM — the Converse API, no CLI involved."""
    from chia.models.bedrock import BedrockLLM

    llm = BedrockLLM(model=model_id, region=region, max_tokens=2048, retries=2)
    result = llm.prompt(PROMPT, tools=[])
    return result.result, result.usage


def _run_claude_cli(model_id: str, region: str, proxy_url: Optional[str],
                    ) -> tuple[str, TokenUsage]:
    """The Claude Code CLI, natively or through chia's Converse proxy.

    The only difference between the two arms is ``proxy_url``, which is the point: the
    harness is byte-identical, so a difference in outcome is attributable to the
    translation and to nothing else.
    """
    from chia.models.bedrock_config import bedrock_model_env
    from chia.models.claude import ClaudeCodeLLM

    env_overlay = bedrock_model_env(primary=model_id, region=region,
                                    proxy_url=proxy_url, require_tools=False)
    previous = {k: os.environ.get(k) for k in env_overlay}
    os.environ.update(env_overlay)
    try:
        llm = ClaudeCodeLLM(model=model_id, log_stream=True, retries=2,
                           projects_cwd=None, max_tokens=2048)
        result = llm.prompt(PROMPT, tools=[])
        return result.result, result.usage
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_opencode(model_id: str, region: str, proxy_url: Optional[str],
                  ) -> tuple[str, TokenUsage]:
    """The opencode CLI — the harness the proxy has to justify itself against."""
    from chia.models.opencode import OpenCodeLLM

    llm = OpenCodeLLM(model=model_id, retries=2)
    result = llm.prompt(PROMPT, tools=[])
    return result.result, result.usage


def run_row(harness: str, model_alias: str, repeat: int, *, region: str,
            proxy_url: Optional[str]) -> Row:
    """Run one cell. Never raises: a failed cell is a recorded row, not a lost grid."""
    row = Row(harness=harness, model=model_alias, repeat=repeat)
    bedrock_id, opencode_id = MODELS[model_alias]

    if harness == "opencode" and not opencode_id:
        row.skipped_reason = "opencode has no route to this model"
        return row
    if harness == "cli_native" and not _is_anthropic(bedrock_id):
        # THE hole the proxy exists to fill. Recorded as a skip with the reason, so the
        # figure shows an absent cell rather than a missing one.
        row.skipped_reason = "the CLI's native Bedrock transport cannot speak Converse"
        return row

    started = time.perf_counter()
    try:
        if harness == "converse":
            text, usage = _run_converse(bedrock_id, region)
        elif harness == "cli_native":
            text, usage = _run_claude_cli(bedrock_id, region, None)
        elif harness == "cli_proxy":
            text, usage = _run_claude_cli(bedrock_id, region, proxy_url)
        else:
            text, usage = _run_opencode(opencode_id, region, proxy_url)
        row.wall_s = time.perf_counter() - started
        row.usage = usage
        if not (text or "").strip():
            # A harness that produced nothing did not answer wrongly — it did not answer.
            # Grading an empty string as a failure would report an auth or config problem
            # as a quality result, which is exactly the misleading claim this study exists
            # to avoid making.
            row.error = "harness produced no output (see harness_failed)"
            row.harness_failed = True
        else:
            row.passed = grade(text)
    except Exception as exc:
        row.wall_s = time.perf_counter() - started
        row.error = f"{type(exc).__name__}: {exc}"
        row.harness_failed = True
    return row


def _is_anthropic(model_id: str) -> bool:
    from chia.models.proxy.translate import is_anthropic_model

    return is_anthropic_model(model_id)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def plan(harnesses: List[str], models: List[str], repeats: int) -> List[tuple]:
    """The cells to run, as data. Skips are still planned so they are still recorded."""
    return [(h, m, r) for h in harnesses for m in models for r in range(repeats)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harnesses", nargs="+", default=list(HARNESSES),
                        choices=list(HARNESSES))
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        choices=list(MODELS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--proxy-url", default="http://127.0.0.1:8123")
    parser.add_argument("--cap-usd", type=float, default=20.0,
                        help="metered-spend ceiling; the grid is refused if it would "
                             "project over this")
    parser.add_argument("--per-call-usd", type=float, default=None,
                        help="override the projection's per-call estimate; needed for "
                             "the first run, which has no history")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "harness_grid.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the projected cost; call nothing")
    args = parser.parse_args(argv)

    cells = plan(args.harnesses, args.models, args.repeats)
    live = [c for c in cells
            if not (c[0] == "opencode" and not MODELS[c[1]][1])
            and not (c[0] == "cli_native" and not _is_anthropic(MODELS[c[1]][0]))]

    print(f"{len(cells)} cell(s) planned, {len(live)} of them live "
          f"({len(cells) - len(live)} structurally impossible and recorded as skips)")

    history = _load_history(args.out)
    try:
        status = check_budget(args.cap_usd, history, projected_calls=len(live),
                             per_call_usd=args.per_call_usd)
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    estimate = (f"~${status.per_call_usd:.4f}/call from {status.priced_calls} recorded"
                if status.per_call_usd else "no history to estimate from")
    print(f"budget: ${status.metered_usd:.2f} metered so far, cap ${args.cap_usd:.2f}, "
          f"{estimate}")

    if args.dry_run:
        for harness, model, repeat in cells:
            mark = "live" if (harness, model, repeat) in live else "skip"
            print(f"  [{mark}] {harness:11} {model:10} rep{repeat}")
        return 0

    rows: List[Row] = []
    for index, (harness, model, repeat) in enumerate(cells, 1):
        row = run_row(harness, model, repeat, region=args.region,
                      proxy_url=args.proxy_url)
        rows.append(row)
        verdict = ("skip" if row.skipped_reason
                   else "HARNESS-FAILED" if row.harness_failed
                   else "PASS" if row.passed else "fail")
        cost = "" if row.usage.cost_usd is None else f" ${row.usage.cost_usd:.4f}"
        print(f"[{index}/{len(cells)}] {harness:11} {model:10} rep{repeat} "
              f"-> {verdict}{cost} {row.wall_s:.1f}s "
              f"{row.error[:80]}{row.skipped_reason[:60]}")
        _write(rows, args.out)

    _summarise(rows)
    return 0


def _load_history(path: Path) -> List[TokenUsage]:
    """Prior rows from *path*, as usages, so the projection learns from real spend."""
    if not path.is_file():
        return []
    out = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cost = row.get("cost_usd") or ""
            out.append(TokenUsage(
                input_tokens=int(row.get("input_tokens") or 0),
                output_tokens=int(row.get("output_tokens") or 0),
                cost_usd=float(cost) if cost else None,
                cost_source=row.get("cost_source") or "unavailable",
                billing_mode=row.get("billing_mode") or "per_token",
            ))
    return out


def _write(rows: List[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv())


def _summarise(rows: List[Row]) -> None:
    print("\nharness      model      n  pass  $/task    med wall")
    groups: Dict[tuple, List[Row]] = {}
    for row in rows:
        if row.skipped_reason or row.harness_failed:
            # Neither is a quality datum: one never ran, the other never answered.
            continue
        groups.setdefault((row.harness, row.model), []).append(row)
    for (harness, model), items in sorted(groups.items()):
        passed = sum(1 for r in items if r.passed)
        costs = [r.usage.cost_usd for r in items if r.usage.cost_usd is not None]
        cost = f"${sum(costs) / len(costs):.4f}" if costs else "n/a"
        walls = sorted(r.wall_s for r in items)
        print(f"{harness:12} {model:10} {len(items):<2} {passed}/{len(items):<4} "
              f"{cost:9} {walls[len(walls) // 2]:.1f}s")
    print("\nproportions above are over the n in the same row; a rate over 3 trials is "
          "not a rate over 300.")


if __name__ == "__main__":
    sys.exit(main())
