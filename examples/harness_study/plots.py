"""Figures for the harness study. Each one reads a committed CSV and hardcodes nothing.

Three rules, because they are what make a figure survive review:

1. **Every figure reads a CSV.** No number is typed into this file. Regenerate the
   CSVs with ``collect.py`` and the figures follow.
2. **Every figure writes the subset it drew.** Alongside ``figures/<name>.png`` goes
   ``figures/<name>.csv`` — exactly the rows and columns behind that panel, so a
   reader can check the figure without re-deriving the pipeline.
3. **Every proportion is annotated with its denominator.** A rate over 29 runs must
   not render like a rate over 2900.

Figures, and the claim each one tests:

``cache_share``
    Cache tokens as a share of billed input. If the share is small, the four-class
    split is pedantry; the argument for it lives or dies here.
``pricing_error``
    Cost computed with a two-class model divided by cost computed with four
    classes, per run. 1.0 everywhere would mean the split changes no number.
``retry_visibility``
    How often a recorded run retried, against how often the tokens those attempts
    burned were recoverable from the record. This one is a *negative* result about
    the old telemetry, which is the point.
``reimplementation_count``
    Independent implementations of the same capability across chia / aet /
    oscar-merlin / spec. The G2 gate, drawn.
``deletion_ratio``
    Downstream workaround lines deleted per upstream source line added, per PR. The
    G3 gate, drawn — and it is unflattering to several of these PRs, which is
    information rather than a reason to omit it.

Usage::

    python examples/harness_study/plots.py                 # all figures
    python examples/harness_study/plots.py cache_share     # just one
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

HERE = Path(__file__).parent
DATA = HERE / "data"
FIGURES = HERE / "figures"

# collect.py and pricing.py are siblings, not part of the chia package: this
# directory is a self-contained study, so it has to be importable whether it is run
# as a script or imported.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# One colour per role, so the same idea is the same colour across figures.
C_CORRECT = "#2b6cb0"
C_WRONG = "#c05621"
C_NEUTRAL = "#718096"
C_ACCENT = "#2f855a"


def _require_matplotlib():
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - environment-dependent
        sys.exit("matplotlib is required to draw the figures: pip install matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    })
    return plt


def _save(fig, name: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    """Write ``figures/<name>.png`` plus the exact subset it drew (rule 2)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    with (FIGURES / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    print(f"wrote figures/{name}.png  (+ {name}.csv, {len(rows)} rows)")


def _load_runs() -> List[dict]:
    from collect import read_csv

    return read_csv(DATA / "runs.csv")


def _counts(row: dict):
    from pricing import Counts

    return Counts(
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        cache_read_tokens=int(row.get("cache_read_tokens") or 0),
        cache_creation_tokens=int(row.get("cache_creation_tokens") or 0),
    )


# ---------------------------------------------------------------------------
# 1 — cache tokens as a share of billed input
# ---------------------------------------------------------------------------


def fig_cache_share() -> None:
    """Does the cache actually dominate the prompt? Everything else depends on it."""
    plt = _require_matplotlib()
    rows = _load_runs()

    drawn = []
    for row in rows:
        counts = _counts(row)
        unsplit = int(row.get("cache_unsplit_tokens") or 0)
        cache = (counts.cache_read_tokens + counts.cache_creation_tokens) or unsplit
        billed = counts.input_tokens + cache
        if not billed:
            continue
        drawn.append({
            "run_id": row["run_id"],
            "source": row["source"],
            "model": row["model"],
            "fresh_input_tokens": counts.input_tokens,
            "cache_read_tokens": counts.cache_read_tokens,
            "cache_creation_tokens": counts.cache_creation_tokens,
            "cache_unsplit_tokens": unsplit,
            "billed_input_tokens": billed,
            "cache_share": cache / billed,
        })
    drawn.sort(key=lambda r: r["cache_share"])
    shares = [r["cache_share"] for r in drawn]
    n = len(shares)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                      gridspec_kw={"width_ratios": [1.3, 1]})

    # An ECDF rather than a bar per run: most runs sit at or near 100%, and a bar
    # chart of a saturated distribution reads as a rendering error rather than a
    # result. The step plot shows the mass at the top honestly.
    xs = [s * 100 for s in shares]
    ys = [(i + 1) / n * 100 for i in range(n)]
    left.step(xs, ys, where="post", color=C_ACCENT, linewidth=1.8)
    left.fill_between(xs, ys, step="post", color=C_ACCENT, alpha=0.15)
    # Only two markers: the distribution is trimodal (a cluster near half, a large
    # cluster at essentially all-cache, and a handful of uncached runs), so a ladder
    # of thresholds would print the same count three times and read as a bug.
    median = sorted(shares)[n // 2] * 100 if n else 0.0
    saturated = sum(1 for s in shares if s >= 0.99)
    left.axvline(median, color=C_WRONG, linewidth=1.2, linestyle="--")
    left.text(median - 2, 52, f"median {median:.0f}%", rotation=90, ha="right",
              va="center", fontsize=7.5, color=C_WRONG)
    left.text(98, 55, f"{saturated}/{n} runs are\n≥99% cache", rotation=90,
              ha="right", va="center", fontsize=7.5, color=C_NEUTRAL)
    left.set_xlabel("cache tokens as % of billed input")
    left.set_ylabel(f"% of runs at or below\n(n = {n})")
    left.set_xlim(0, 101)
    left.set_ylim(0, 101)
    left.set_title("Distribution across recorded runs", loc="left")

    totals = {
        "fresh\ninput": sum(r["fresh_input_tokens"] for r in drawn),
        "cache\nread": sum(r["cache_read_tokens"] for r in drawn),
        "cache\nwrite": sum(r["cache_creation_tokens"] for r in drawn),
        "cache\nunsplit": sum(r["cache_unsplit_tokens"] for r in drawn),
    }
    labels = [k for k, v in totals.items() if v]
    values = [totals[k] / 1e6 for k in labels]
    colours = [C_NEUTRAL if k.startswith("fresh") else C_ACCENT for k in labels]
    right.bar(labels, values, color=colours, width=0.62)
    grand = sum(values)
    for index, value in enumerate(values):
        right.text(index, value, f"{value / grand * 100:.0f}%", ha="center",
                   va="bottom", fontsize=8)
    right.set_ylabel("Mtok, all runs")
    right.set_ylim(0, max(values) * 1.28)
    right.tick_params(axis="x", labelsize=8)
    right.set_title("Aggregate by input class", loc="left")

    cached = grand - values[0] if values else 0
    fig.suptitle(
        f"{cached / grand * 100:.0f}% of billed input is cache, not fresh input — "
        f"which is why the classes cannot be collapsed",
        fontsize=10, y=1.05,
    )
    _save(fig, "cache_share", drawn, list(drawn[0]) if drawn else [])
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2 — what the two-class calculation gets wrong
# ---------------------------------------------------------------------------


def fig_pricing_error() -> None:
    """Two-class cost / four-class cost, per run. The G4 gate for PR5 and PR7."""
    plt = _require_matplotlib()
    from pricing import error_ratio, price_four_class, price_two_class

    rows = _load_runs()
    drawn = []
    for row in rows:
        model = row.get("model") or ""
        counts = _counts(row)
        # Rows whose cache figure was recorded merged cannot be split-priced at all;
        # excluded here and reported in the caption rather than approximated.
        if int(row.get("cache_unsplit_tokens") or 0):
            continue
        correct = price_four_class(counts, model)
        naive = price_two_class(counts, model)
        ratio = error_ratio(counts, model)
        if ratio is None:
            continue
        drawn.append({
            "run_id": row["run_id"], "model": model, "source": row["source"],
            "four_class_usd": round(correct, 6),
            "two_class_usd": round(naive, 6),
            "ratio": round(ratio, 4),
            "cache_read_tokens": counts.cache_read_tokens,
            "cache_creation_tokens": counts.cache_creation_tokens,
        })
    excluded = sum(1 for r in rows if int(r.get("cache_unsplit_tokens") or 0))
    unpriced = len(rows) - len(drawn) - excluded

    drawn.sort(key=lambda r: r["ratio"])
    ratios = [r["ratio"] for r in drawn]
    n = len(ratios)
    total_correct = sum(r["four_class_usd"] for r in drawn)
    total_naive = sum(r["two_class_usd"] for r in drawn)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.4),
                                      gridspec_kw={"width_ratios": [2, 1]})

    colours = [C_WRONG if r > 1 else C_CORRECT for r in ratios]
    left.bar(range(n), ratios, width=1.0, color=colours, linewidth=0)
    left.axhline(1.0, color="black", linewidth=1.0)
    left.set_xlabel(f"recorded run, sorted by error   (n = {n})")
    left.set_ylabel("two-class cost\n÷ four-class cost")
    left.set_title("Per-run pricing error", loc="left")
    left.text(0.02, 0.95,
              "above 1.0: overstated (cheap cache reads billed as fresh input)\n"
              "below 1.0: understated (cache writes carry a premium)",
              transform=left.transAxes, va="top", fontsize=7.5, color=C_NEUTRAL)

    right.bar(["four-class\n(correct)", "two-class\n(input+output)"],
              [total_correct, total_naive], color=[C_CORRECT, C_WRONG], width=0.6)
    for index, value in enumerate([total_correct, total_naive]):
        right.text(index, value, f" ${value:,.0f}", ha="center", va="bottom",
                   fontsize=8)
    right.set_ylabel("total USD")
    right.set_ylim(0, max(total_correct, total_naive) * 1.25)
    factor = total_naive / total_correct if total_correct else float("nan")
    right.set_title(f"Aggregate over {n} runs: {factor:.2f}x", loc="left")

    worst = max(ratios) if ratios else float("nan")
    over = sum(1 for r in ratios if r > 1.05)
    under = sum(1 for r in ratios if r < 0.95)
    fig.suptitle(
        f"Per-run error reaches {worst:.1f}x in both directions, while the "
        f"aggregate cancels to {factor:.2f}x — an aggregate check would have "
        f"missed this",
        fontsize=10, y=1.05,
    )
    caption = (f"{over} runs overstated by >5%, {under} understated by >5%; "
               f"excluded: {excluded} runs recorded only a merged cache total, "
               f"{unpriced} had no rate for their model")
    fig.text(0.5, -0.10, caption, ha="center", fontsize=7.5, color=C_NEUTRAL)
    _save(fig, "pricing_error", drawn, list(drawn[0]) if drawn else [])
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3 — retries were counted but never costed
# ---------------------------------------------------------------------------


def fig_retry_visibility() -> None:
    """A negative result about the old telemetry, which is the argument for PR6.

    Retries were recorded as an integer and nothing else. This figure shows how often
    they happened and that not one of those attempts has recoverable token counts —
    so the honest y-axis is *visibility*, not lost dollars. Fabricating a token count
    for those attempts would be inventing the very number the fix exists to produce.
    """
    plt = _require_matplotlib()
    # Read the wider retry CSV, not runs.csv: a cell that retried and then failed
    # carries no token counts, so filtering to token-bearing rows would drop most of
    # the retries and understate how often this happens.
    with (DATA / "retries.csv").open(newline="") as handle:
        rows = [{**row, "retries": int(row["retries"] or 0)}
                for row in csv.DictReader(handle)]

    with_field = rows
    retried = [r for r in with_field if r["retries"] > 0]
    attempts = int(sum(r["retries"] for r in retried))
    # A recorded retry whose tokens are recoverable would need per-attempt counts in
    # the row. No source in this corpus has them — that absence *is* the finding.
    costed = 0

    drawn = [{
        "run_id": r["run_id"], "source": r["source"], "model": r.get("model", ""),
        "retries": int(r["retries"]),
        "retry_tokens_recoverable": "no",
    } for r in retried]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.2),
                                      gridspec_kw={"width_ratios": [1, 1]})

    n_field = len(with_field)
    left.bar(["retried\nat least once", "clean"],
             [len(retried), n_field - len(retried)],
             color=[C_WRONG, C_NEUTRAL], width=0.55)
    for index, value in enumerate([len(retried), n_field - len(retried)]):
        share = value / n_field * 100 if n_field else 0
        left.text(index, value, f" {value} ({share:.0f}%)", ha="center", va="bottom",
                  fontsize=8)
    left.set_ylabel("runs")
    left.set_ylim(0, n_field * 1.2 if n_field else 1)
    left.set_title(f"Retry incidence   (n = {n_field} runs with the field)",
                   loc="left")

    right.bar(["retried attempts\nrecorded", "attempts with\nrecoverable tokens"],
              [attempts, costed], color=[C_WRONG, C_CORRECT], width=0.55)
    for index, value in enumerate([attempts, costed]):
        right.text(index, value, f" {value}", ha="center", va="bottom", fontsize=8)
    right.set_ylabel("attempts")
    right.set_ylim(0, max(attempts, 1) * 1.25)
    right.set_title("Every retried attempt was billed; none was attributable",
                    loc="left")

    fig.suptitle(
        "Retries were logged as a count, never as a cost — so the spend they "
        "caused cannot be recovered after the fact",
        fontsize=10, y=1.05,
    )
    _save(fig, "retry_visibility", drawn,
          ["run_id", "source", "model", "retries", "retry_tokens_recoverable"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4 — independent reimplementation count (gate G2)
# ---------------------------------------------------------------------------


def fig_reimplementation_count() -> None:
    """How many times each capability was built independently, and by whom."""
    plt = _require_matplotlib()
    with (DATA / "reimplementations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_capability: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_capability[row["capability"]].append(row)
    order = sorted(by_capability, key=lambda k: (-len(by_capability[k]), k))
    repos = sorted({row["repo"] for row in rows})
    palette = {repo: colour for repo, colour in
               zip(repos, [C_CORRECT, C_ACCENT, C_WRONG, C_NEUTRAL, "#805ad5"])}

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    for index, capability in enumerate(order):
        left = 0
        counted = Counter(row["repo"] for row in by_capability[capability])
        for repo in repos:
            width = counted.get(repo, 0)
            if not width:
                continue
            ax.barh(index, width, left=left, color=palette[repo], height=0.6,
                    label=repo if index == 0 or repo not in ax.get_legend_handles_labels()[1] else None)
            ax.text(left + width / 2, index, repo.replace("oscar-merlin", "merlin"),
                    ha="center", va="center", fontsize=7, color="white")
            left += width
        ax.text(left + 0.06, index, str(left), va="center", fontsize=8)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("independent implementations of the same behaviour")
    widest = max(len(v) for v in by_capability.values())
    ax.set_xlim(0, widest + 0.8)
    ax.set_xticks(range(widest + 1))
    ax.set_title(
        "Gate G2: what four repos built separately because the framework had no seam",
        loc="left", fontsize=10,
    )
    _save(fig, "reimplementation_count", rows,
          ["capability", "repo", "path", "note"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5 — deletion ratio (gate G3)
# ---------------------------------------------------------------------------


def fig_deletion_ratio() -> None:
    """Downstream lines deleted per upstream source line added.

    Deliberately unflattering where it should be: three of these PRs delete nothing
    downstream, because no workaround was possible from outside chia. A gate that
    only ever agrees with you is not a gate.
    """
    plt = _require_matplotlib()
    with (DATA / "deletion_ratio.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    labels = [row["pr"] for row in rows]
    added = [int(row["upstream_src_added"]) for row in rows]
    deleted = [int(row["downstream_deleted"]) for row in rows]
    ratios = [d / a if a else 0.0 for d, a in zip(deleted, added)]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1]})

    positions = range(len(labels))
    width = 0.38
    top.bar([p - width / 2 for p in positions], added, width=width,
            color=C_NEUTRAL, label="upstream source lines added")
    top.bar([p + width / 2 for p in positions], deleted, width=width,
            color=C_ACCENT, label="downstream workaround lines deleted")
    top.set_ylabel("lines")
    top.set_yscale("symlog", linthresh=10)
    top.legend(frameon=False, fontsize=8)
    top.set_title("Gate G3: what each change costs upstream and buys downstream",
                  loc="left", fontsize=10)

    colours = [C_ACCENT if r >= 1 else (C_WRONG if r > 0 else C_NEUTRAL)
               for r in ratios]
    bottom.bar(positions, ratios, width=0.6, color=colours)
    bottom.axhline(1.0, color="black", linewidth=1.0)
    for index, ratio in enumerate(ratios):
        bottom.text(index, ratio, f" {ratio:.2f}", ha="center", va="bottom",
                    fontsize=8)
    bottom.set_ylabel("deleted ÷ added")
    bottom.set_xticks(list(positions))
    bottom.set_xticklabels(labels)
    bottom.set_ylim(0, max(ratios) * 1.25 if any(ratios) else 1)
    fig.text(0.5, -0.02,
             "ratio 0 means no downstream workaround existed — the gap could not be "
             "worked around from outside chia,\nso those PRs have to be justified by "
             "G4 (a recorded number is wrong) instead",
             ha="center", va="top", fontsize=7.5, color=C_NEUTRAL)

    _save(fig, "deletion_ratio", rows, list(rows[0]))
    plt.close(fig)


FIGURE_FUNCS: Dict[str, Callable[[], None]] = {
    "cache_share": fig_cache_share,
    "pricing_error": fig_pricing_error,
    "retry_visibility": fig_retry_visibility,
    "reimplementation_count": fig_reimplementation_count,
    "deletion_ratio": fig_deletion_ratio,
}


def main(argv: Optional[List[str]] = None) -> int:
    # `choices` is read at call time, not at module definition time, so a figure
    # registered further down the file (harness_effect) is still selectable.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figures", nargs="*", choices=sorted(FIGURE_FUNCS) or None,
                       help="figures to draw (default: all)")
    args = parser.parse_args(argv)
    for name in args.figures or sorted(FIGURE_FUNCS):
        FIGURE_FUNCS[name]()
    return 0


# ---------------------------------------------------------------------------
# 6 — the harness effect at fixed model
# ---------------------------------------------------------------------------


def fig_harness_effect() -> None:
    """Cost and pass rate per harness, at a fixed model. The plot that justifies — or
    fails to justify — the Converse proxy.

    Reads ``data/harness_grid.csv`` (produced by ``grid.py``). Skipped cells are drawn
    as absent rather than as zeros: an empty ``cli_native`` column for a Converse-only
    model is the hole the proxy exists to fill, and drawing it as a zero would read as
    "free and never passes".
    """
    plt = _require_matplotlib()
    path = DATA / "harness_grid.csv"
    if not path.is_file():
        print(f"skipping harness_effect: {path} not found (run grid.py first)")
        return
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    # Three categories, and conflating any two would misreport the comparison:
    #   live    — the harness ran and answered; a quality datum.
    #   skipped — structurally impossible (no route from that harness to that model).
    #   failed  — the harness never produced an answer (auth, config, provider outage).
    # A failed harness plotted as 0/n would report an environment problem as a quality
    # result, which is precisely the misleading claim this study exists to avoid.
    failed = [r for r in rows if r.get("harness_failed") == "1"]
    live = [r for r in rows
            if not r["skipped_reason"] and r.get("harness_failed") != "1"]
    skipped = [r for r in rows if r["skipped_reason"]]
    if not live:
        print("skipping harness_effect: no live rows")
        return

    harness_order = [h for h in ("converse", "cli_native", "cli_proxy", "opencode")
                     if any(r["harness"] == h for r in live)]
    model_order = sorted({r["model"] for r in live})
    palette = {"converse": C_CORRECT, "cli_native": C_NEUTRAL,
               "cli_proxy": C_ACCENT, "opencode": "#805ad5"}

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.5),
                                      gridspec_kw={"wspace": 0.3})

    width = 0.8 / max(len(harness_order), 1)
    drawn = []
    for h_index, harness in enumerate(harness_order):
        xs, ys, labels = [], [], []
        for m_index, model in enumerate(model_order):
            cells = [r for r in live
                     if r["harness"] == harness and r["model"] == model
                     and r["cost_usd"]]
            if not cells:
                continue
            costs = [float(r["cost_usd"]) for r in cells]
            mean_cost = sum(costs) / len(costs)
            xs.append(m_index + h_index * width - 0.4 + width / 2)
            ys.append(mean_cost)
            passed = sum(1 for r in cells if r["passed"] == "1")
            labels.append(f"{passed}/{len(cells)}")
            drawn.append({"harness": harness, "model": model, "n": len(cells),
                          "passed": passed, "mean_cost_usd": round(mean_cost, 6)})
        if not xs:
            continue
        left.bar(xs, ys, width=width, color=palette.get(harness, C_NEUTRAL),
                 label=harness)
        for x, y, label in zip(xs, ys, labels):
            left.text(x, y, f" {label}", ha="center", va="bottom", fontsize=6.5,
                      rotation=90)

    left.set_xticks(range(len(model_order)))
    left.set_xticklabels(model_order)
    left.set_yscale("log")
    left.set_ylabel("mean $ per task (log)")
    left.set_xlabel("model (bars annotated pass/n)")
    left.set_title("Cost per task at fixed model, by harness", loc="left",
                   fontsize=9.5)
    left.legend(frameon=False, fontsize=7.5, ncol=2)

    # The prompt each harness sends, which is what the cost difference is made of.
    for h_index, harness in enumerate(harness_order):
        cells = [r for r in live if r["harness"] == harness]
        if not cells:
            continue
        billed_in = [int(r["input_tokens"] or 0) + int(r["cache_read_tokens"] or 0)
                     + int(r["cache_creation_tokens"] or 0) for r in cells]
        right.bar(h_index, sum(billed_in) / len(billed_in),
                  color=palette.get(harness, C_NEUTRAL), width=0.6)
    right.set_xticks(range(len(harness_order)))
    right.set_xticklabels(harness_order, rotation=20, ha="right", fontsize=8)
    right.set_yscale("log")
    right.set_ylabel("mean billed input tokens (log)")
    right.set_title("What each harness sends per task", loc="left", fontsize=9.5)

    n_live = len(live)
    fig.suptitle(
        "At the same model, the harness — not the model — dominates cost per task",
        fontsize=10, y=1.04,
    )
    failed_note = ""
    if failed:
        harnesses = sorted({r["harness"] for r in failed})
        failed_note = (f"; {len(failed)} cell(s) excluded because the harness itself "
                       f"never answered ({', '.join(harnesses)} — see the error column, "
                       f"an environment failure, NOT a wrong answer)")
    fig.text(0.5, -0.18,
             f"{n_live} live cell(s), {len(skipped)} structurally impossible and drawn "
             f"as absent rather than as zero{failed_note}. Every rate is over the n in "
             f"its own bar.",
             ha="center", fontsize=7.5, color=C_NEUTRAL, wrap=True)
    _save(fig, "harness_effect", drawn,
          ["harness", "model", "n", "passed", "mean_cost_usd"])
    plt.close(fig)


FIGURE_FUNCS["harness_effect"] = fig_harness_effect


if __name__ == "__main__":
    sys.exit(main())
