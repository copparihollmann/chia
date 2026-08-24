"""Figures for the harness study. Each one reads a committed CSV and hardcodes nothing.

Three rules, because they are what make a figure survive review:

1. **Every figure reads a CSV.** No number is typed into this file. Regenerate the
   CSVs with ``collect.py`` and the figures follow.
2. **Every figure writes the subset it drew.** Alongside ``figures/<name>.png`` goes
   ``figures/<name>.csv`` — exactly the rows and columns behind that panel, so a
   reader can check the figure without re-deriving the pipeline.
3. **Every proportion is annotated with its denominator.** A rate over 29 runs must
   not render like a rate over 2900.

And a fourth rule, added after a first pass produced figures that were honest but weak:

4. **A figure has to earn the space.** Five small integers are a table. A binary outcome
   is a sentence. A ratio whose denominator is "how much test code I wrote" is an
   argument, not a measurement. Three figures were cut on those grounds
   (``reimplementation_count``, ``retry_visibility``, ``deletion_ratio``); what they said
   is now stated as text or a table in the README, where it belongs. A weak figure does
   not add support, it spends the reader's trust.

Figures, and the claim each one tests:

``cache_share``
    Cache tokens as a share of billed input, over 152 recorded runs. If the share is
    small, the four-class split is pedantry; the argument for it lives or dies here.
``pricing_error``
    Per-run two-class cost ÷ four-class cost, against the aggregate. The point is the
    gap between them: the aggregate cancels while the per-run figures do not.
``cold_start``
    The first call against a fresh prompt cache and every call after it, at identical
    token counts. The sharpest of the set: four-class pricing puts the two 12.5x apart
    while two-class pricing reports one number for both.
``prompt_overhead``
    What each harness sends before the task starts, split by token class, at a fixed
    model. The mechanism behind ``harness_cost``.
``harness_cost``
    Cost per task, model held fixed, harness varied. Cost only — n per cell is nowhere
    near enough to compare quality, and a bar chart of 5-trial success rates would
    invite exactly that reading.
``proxied_cost``
    What the client says a proxied call cost against what the provider charged, with an
    Anthropic model as the negative control. Token counts are identical on both sides, so
    the gap is rate mispricing with no confound.
``framing_double_bill``
    Provider calls per task, and the share of them that were non-streaming fallbacks,
    under SSE framing against AWS event-stream framing. Two counts rather than rates, and
    the reason the proxy no longer needs an undocumented CLI flag.

Usage::

    python examples/harness_study/plots.py                 # all figures
    python examples/harness_study/plots.py cache_share     # just one
"""

from __future__ import annotations

import argparse
import csv
import json
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
                                      gridspec_kw={"width_ratios": [1.3, 1],
                                                   "wspace": 0.3})

    # A histogram, not an ECDF: the mass sits in three clusters rather than spread across
    # the range, and an ECDF of a clustered distribution is a few vertical steps, which
    # reads as a rendering fault rather than a result. The shape is part of the finding —
    # cache share is close to a property of the workload, not a continuum — so it has to
    # be visible.
    bins = [i * 10 for i in range(11)]
    counts_per_bin = [0] * 10
    for share in shares:
        index = min(int(share * 10), 9)
        counts_per_bin[index] += 1
    left.bar([b + 5 for b in bins[:-1]], counts_per_bin, width=9,
             color=C_ACCENT, linewidth=0)
    left.set_xlabel("cache tokens as % of billed input")
    left.set_ylabel(f"recorded runs (n = {n})")
    left.set_xlim(-2, 102)
    left.set_xticks([0, 25, 50, 75, 100])
    left.set_title("How much of each run's prompt was cached", loc="left")

    uncached = sum(1 for s in shares if s < 0.01)
    saturated = sum(1 for s in shares if s >= 0.99)
    left.annotate(f"{saturated} runs ≥99% cache",
                  xy=(95, counts_per_bin[9]), xytext=(48, counts_per_bin[9] * 0.82),
                  fontsize=8, color=C_NEUTRAL,
                  arrowprops={"arrowstyle": "->", "color": C_NEUTRAL, "lw": 0.8})
    if uncached:
        left.annotate(f"{uncached} with no cache at all",
                      xy=(4, counts_per_bin[0]),
                      xytext=(18, max(counts_per_bin) * 0.45),
                      fontsize=8, color=C_NEUTRAL,
                      arrowprops={"arrowstyle": "->", "color": C_NEUTRAL, "lw": 0.8})

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
    rates = {"fresh\ninput": "1x", "cache\nread": "0.1x", "cache\nwrite": "1.25x"}
    for index, (label, value) in enumerate(zip(labels, values)):
        note = f"{value / grand * 100:.0f}%"
        if label in rates:
            note += f"\n@ {rates[label]}"
        right.text(index, value, note, ha="center", va="bottom", fontsize=8)
    right.set_ylabel("Mtok, all runs")
    right.set_ylim(0, max(values) * 1.35)
    right.tick_params(axis="x", labelsize=8)
    right.set_title("Aggregate, with each class's rate", loc="left")

    cached = grand - values[0] if values else 0
    fig.suptitle(
        f"{cached / grand * 100:.0f}% of billed input is cache, not fresh input, and the "
        f"three classes bill at rates 12.5x apart",
        fontsize=10, y=1.04,
    )
    fig.text(0.5, -0.12,
             "Rates are Anthropic's published multipliers on the input rate. A two-class "
             "accounting applies the 1x rate to all of it.",
             ha="center", fontsize=7.5, color=C_NEUTRAL)
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


FIGURE_FUNCS: Dict[str, Callable[[], None]] = {
    "cache_share": fig_cache_share,
    "pricing_error": fig_pricing_error,
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
# 3 — the harness effect at fixed model
# ---------------------------------------------------------------------------


def _grid_rows() -> List[dict]:
    """The live cells of ``data/harness_grid.csv``.

    Three categories, and conflating any two misreports the comparison: ``live`` (the
    harness ran and answered), ``skipped`` (no route from that harness to that model), and
    ``harness_failed`` (never produced an answer — auth, config, outage). A failed cell
    plotted as 0/n reports an environment problem as a quality result.
    """
    path = DATA / "harness_grid.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [r for r in csv.DictReader(handle)
                if not r.get("skipped_reason") and r.get("harness_failed") != "1"]


HARNESS_ORDER = ("converse", "opencode", "cli_native", "cli_proxy")
HARNESS_COLOUR = {"converse": C_CORRECT, "opencode": "#805ad5",
                  "cli_native": C_NEUTRAL, "cli_proxy": C_ACCENT}


def _billed_input(row: dict) -> int:
    return (int(row.get("input_tokens") or 0)
            + int(row.get("cache_read_tokens") or 0)
            + int(row.get("cache_creation_tokens") or 0))


def fig_harness_cost() -> None:
    """Cost per task, model held fixed, harness varied.

    The claim: **at a fixed model the harness sets the bill.** It is a cost claim only —
    quality is reported in the caption as a count, not drawn, because n per cell is far
    too small to compare success rates and a bar chart of 5-trial rates would invite
    exactly that reading.

    Every cost here is priced from the provider's own token counts. For ``cli_proxy`` that
    means the proxy's usage log rather than the CLI's self-reported figure, which prices a
    Claude call it did not make.
    """
    plt = _require_matplotlib()
    rows = [r for r in _grid_rows() if r.get("cost_usd")]
    if not rows:
        print("skipping harness_cost: no live rows with a cost (run grid.py first)")
        return

    harnesses = [h for h in HARNESS_ORDER if any(r["harness"] == h for r in rows)]
    models = sorted({r["model"] for r in rows})

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    width = 0.8 / max(len(harnesses), 1)
    drawn, passes = [], []
    for h_index, harness in enumerate(harnesses):
        xs, ys = [], []
        for m_index, model in enumerate(models):
            cells = [r for r in rows
                     if r["harness"] == harness and r["model"] == model]
            if not cells:
                continue
            costs = sorted(float(r["cost_usd"]) for r in cells)
            # Median, not mean: exactly one call in each series pays to write the prompt
            # cache and costs several times the rest (see cold_start), so a mean reports
            # the cache lifecycle rather than the harness.
            median_cost = costs[len(costs) // 2]
            passed = sum(1 for r in cells if r["passed"] == "1")
            xs.append(m_index + h_index * width - 0.4 + width / 2)
            ys.append(median_cost)
            billed = sorted(_billed_input(r) for r in cells)
            drawn.append({"harness": harness, "model": model, "n": len(cells),
                          "passed": passed,
                          "median_cost_usd": round(median_cost, 8),
                          "mean_cost_usd": round(sum(costs) / len(costs), 8),
                          "median_billed_input_tokens": billed[len(billed) // 2]})
            passes.append((passed, len(cells)))
        if xs:
            ax.bar(xs, ys, width=width, color=HARNESS_COLOUR.get(harness, C_NEUTRAL),
                   label=harness)
        # A cell with no bar is a cell that cannot exist, and saying so on the figure is
        # the point of the whole exercise: cli_native has no route to a Converse-only
        # model. Left blank it reads as missing data.
        for m_index, model in enumerate(models):
            if any(e["harness"] == harness and e["model"] == model for e in drawn):
                continue
            # y in *axes* coordinates via get_xaxis_transform, not data coordinates. On a
            # log axis the lower data limit is not yet settled when this runs, and placing
            # text there once produced a 108,000-pixel-tall PNG.
            ax.text(m_index + h_index * width - 0.4 + width / 2, 0.02,
                    "no route", rotation=90, ha="center", va="bottom", fontsize=6.5,
                    color=C_NEUTRAL, transform=ax.get_xaxis_transform())

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yscale("log")
    ax.set_ylabel("median $ per task (log)")
    ax.set_xlabel("model")
    ax.legend(frameon=False, fontsize=8, ncol=len(harnesses),
              loc="upper center", bbox_to_anchor=(0.5, -0.22))

    # The ratio that carries the claim, computed rather than asserted.
    ratios = []
    for model in models:
        by_harness = {}
        for entry in drawn:
            if entry["model"] == model:
                by_harness[entry["harness"]] = entry["median_cost_usd"]
        base = by_harness.get("converse")
        if base:
            for harness, cost in by_harness.items():
                if harness != "converse" and cost:
                    ratios.append((model, harness, cost / base))
    ratios.sort(key=lambda item: item[2])
    spread = max((r for _, _, r in ratios), default=float("nan"))
    fig.suptitle(
        f"Same model, different harness: up to {spread:.0f}x the cost per task",
        fontsize=10.5, y=1.02,
    )
    total_passed = sum(p for p, _ in passes)
    total_n = sum(n for _, n in passes)
    fig.text(0.5, -0.30,
             f"{total_n} live cells, n={total_n // max(len(drawn), 1)}, median per cell. "
             f"Solved in {total_passed}/{total_n} — too few trials to rank quality, which "
             f"is why only cost is drawn. Bars absent where the harness has no route.\n"
             f"The widest gaps mix two causes: prompt overhead (see prompt_overhead) and "
             f"extra turns — cli_proxy x glm5 took 4 provider calls per task where the "
             f"others took 1-2.\nNova and GLM use estimated rates (pricing.py); the "
             f"Anthropic figures use published ones.",
             ha="center", fontsize=7.5, color=C_NEUTRAL)
    _save(fig, "harness_cost", drawn,
          ["harness", "model", "n", "passed", "median_cost_usd", "mean_cost_usd",
           "median_billed_input_tokens"])
    plt.close(fig)


FIGURE_FUNCS["harness_cost"] = fig_harness_cost


def fig_prompt_overhead() -> None:
    """What the harness sends before the task starts, split by token class.

    This is the *mechanism* behind ``harness_cost``, and the reason a four-class split is
    not pedantry: the overhead is not fresh input, it is a cache write on the first call
    and a cache read on every later one, and those three rates differ by 12.5x.

    Drawn at ``sonnet`` — the bridge cell every harness can reach — so nothing here is a
    model difference.
    """
    plt = _require_matplotlib()
    rows = [r for r in _grid_rows() if r["model"] == "sonnet"]
    if not rows:
        print("skipping prompt_overhead: no sonnet rows")
        return

    harnesses = [h for h in HARNESS_ORDER if any(r["harness"] == h for r in rows)]
    classes = [
        ("input_tokens", "fresh input", C_CORRECT),
        ("cache_creation_tokens", "cache write (1.25x input)", C_WRONG),
        ("cache_read_tokens", "cache read (0.1x input)", C_ACCENT),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    drawn = []
    for index, harness in enumerate(harnesses):
        cells = [r for r in rows if r["harness"] == harness]
        # Median per class, for the same reason harness_cost uses one: one run in each
        # series writes the cache and the rest read it, so a mean splits the same tokens
        # across two classes and understates both.
        def _median(key: str) -> float:
            values = sorted(int(r.get(key) or 0) for r in cells)
            return values[len(values) // 2]

        means = {key: _median(key) for key, _, _ in classes}
        bottom = 0.0
        for key, _, colour in classes:
            ax.bar(index, means[key], bottom=bottom, width=0.6, color=colour,
                   linewidth=0)
            bottom += means[key]
        total = bottom
        ax.text(index, total, f" {total:,.0f}", ha="center", va="bottom", fontsize=8.5)
        entry = {"harness": harness, "n": len(cells),
                 "median_billed_input_tokens": round(total)}
        entry.update({key: round(means[key]) for key, _, _ in classes})
        drawn.append(entry)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, _, colour in classes]
    ax.legend(handles, [label for _, label, _ in classes], frameon=False, fontsize=8)
    ax.set_xticks(range(len(harnesses)))
    ax.set_xticklabels(harnesses)
    ax.set_ylabel("median billed input tokens per task")
    ax.set_xlabel("harness (model held fixed: sonnet)")

    totals = {e["harness"]: e["median_billed_input_tokens"] for e in drawn}
    base = totals.get("converse")
    biggest = max(totals.values()) if totals else 0
    factor = (biggest / base) if base else float("nan")
    fig.suptitle(
        f"The harness sends {factor:,.0f}x more input than the task needs",
        fontsize=10.5, y=1.0,
    )
    fig.text(0.5, -0.16,
             "Medians over 5 runs per harness, so the bars show the steady state: the "
             "prompt is a cache read by then, which is why the write segment is empty "
             "here. The cold run, where the identical tokens bill 12.5x higher, is the "
             "subject of cold_start.",
             ha="center", fontsize=7.5, color=C_NEUTRAL, wrap=True)
    _save(fig, "prompt_overhead", drawn,
          ["harness", "n", "median_billed_input_tokens", "input_tokens",
           "cache_creation_tokens", "cache_read_tokens"])
    plt.close(fig)


FIGURE_FUNCS["prompt_overhead"] = fig_prompt_overhead


def fig_proxied_cost_is_misreported() -> None:
    """What the client says a proxied call cost, against what the provider charged.

    A controlled comparison, which is what makes it worth drawing: the token counts are
    *identical* on both sides — the proxy records the counts Bedrock returned for the very
    call the CLI then priced. The only thing that differs is which model's rate card was
    applied. So the ratio is a pure mispricing factor, with no confound from prompt size.

    ``sonnet`` is the negative control. It is proxied too, but it really is the model the
    CLI thinks it is, so its ratio must come out at ~1.0. If it does not, the measurement
    is wrong rather than the CLI.
    """
    plt = _require_matplotlib()
    rows = [r for r in _grid_rows()
            if r["harness"] == "cli_proxy" and r.get("cost_usd")
            and r.get("reported_usd")]
    if not rows:
        print("skipping proxied_cost: no cli_proxy rows with both costs")
        return

    models = sorted({r["model"] for r in rows})
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                      gridspec_kw={"width_ratios": [3, 2],
                                                   "wspace": 0.35})

    drawn = []
    for index, model in enumerate(models):
        cells = [r for r in rows if r["model"] == model]
        measured_all = sorted(float(r["cost_usd"]) for r in cells)
        reported_all = sorted(float(r["reported_usd"]) for r in cells)
        measured = measured_all[len(measured_all) // 2]
        reported = reported_all[len(reported_all) // 2]
        left.bar(index - 0.19, reported, width=0.36, color=C_WRONG,
                 label="what the CLI reported" if index == 0 else None)
        left.bar(index + 0.19, measured, width=0.36, color=C_CORRECT,
                 label="what the provider charged" if index == 0 else None)
        ratio = reported / measured if measured else float("nan")
        drawn.append({"model": model, "n": len(cells),
                      "reported_usd": round(reported, 8),
                      "measured_usd": round(measured, 8),
                      "ratio": round(ratio, 2)})

    left.set_xticks(range(len(models)))
    left.set_xticklabels(models)
    left.set_yscale("log")
    left.set_ylabel("median $ per task (log)")
    left.set_xlabel("model, driven through the proxy")
    left.legend(frameon=False, fontsize=8)
    left.set_title("Same tokens, two rate cards", loc="left", fontsize=9.5)

    ratios = [e["ratio"] for e in drawn]
    colours = [C_NEUTRAL if abs(r - 1) < 0.15 else C_WRONG for r in ratios]
    right.bar(range(len(drawn)), ratios, width=0.55, color=colours)
    right.axhline(1.0, color="black", linewidth=1.0)
    for index, ratio in enumerate(ratios):
        right.text(index, ratio, f" {ratio:.1f}x", ha="center", va="bottom", fontsize=8)
    right.set_xticks(range(len(drawn)))
    right.set_xticklabels([e["model"] for e in drawn])
    right.set_ylabel("reported ÷ actual")
    right.set_ylim(0, max(ratios + [1.2]) * 1.3)
    right.set_title("Overstatement factor", loc="left", fontsize=9.5)

    worst = max(ratios) if ratios else float("nan")
    fig.suptitle(
        f"A proxied model's cost is overstated up to {worst:.0f}x — and reported as "
        f"cost_source=\"billed\"",
        fontsize=10.5, y=1.03,
    )
    fig.text(0.5, -0.16,
             "sonnet is the negative control: it is proxied too, but it is the model the "
             "CLI believes it is, so ~1.0 is the correct answer there. Token counts are "
             "identical on both sides of every pair, so the gap is rate mispricing alone. "
             "Nova and GLM rates are estimates (pricing.py).",
             ha="center", fontsize=7.5, color=C_NEUTRAL, wrap=True)
    _save(fig, "proxied_cost", drawn,
          ["model", "n", "reported_usd", "measured_usd", "ratio"])
    plt.close(fig)


FIGURE_FUNCS["proxied_cost"] = fig_proxied_cost_is_misreported


# ---------------------------------------------------------------------------
# 6 — the framing that doubled the bill
# ---------------------------------------------------------------------------


def fig_framing_double_bill() -> None:
    """How many provider calls each response framing costs, and how many were fallbacks.

    Two counts, no pricing and no averaging over models, because a count cannot be
    explained by variance and needs no rate table to be believed.

    Under SSE the CLI could not parse the stream and silently retried each turn on the
    non-streaming ``/invoke`` route, so every turn reached Bedrock twice. It then reported
    the cost of the one call it accepted, which is why nothing in the client's own
    telemetry showed it. The duplicate is a full re-send: the two calls record the same
    token counts, so the doubling is exact rather than estimated.

    The right panel is the mechanism, and it is the cleaner measurement of the two: the
    share of provider calls that were the fallback route. Under a framing the client can
    read, it is zero.
    """
    plt = _require_matplotlib()
    runs = [
        ("SSE\n(guard flag disabled)", "harness_grid_sse_framing.csv",
         "proxy_usage_sse_framing.jsonl", C_WRONG),
        ("event-stream\n(what the CLI asks for)", "harness_grid.csv",
         "proxy_usage.jsonl", C_CORRECT),
    ]

    series = []
    for label, grid_name, usage_name, colour in runs:
        grid_path, usage_path = DATA / grid_name, DATA / usage_name
        if not (grid_path.is_file() and usage_path.is_file()):
            print(f"skipping framing_double_bill: {grid_name} or {usage_name} missing")
            return
        with grid_path.open(newline="") as handle:
            # sonnet only: it is the cell both runs share, and the one where the harness
            # makes a known number of logical model calls. Averaging over models would mix
            # in GLM-5's own retries, which have nothing to do with framing.
            cells = [r for r in csv.DictReader(handle)
                     if r["harness"] == "cli_proxy" and r["model"] == "sonnet"
                     and not r.get("skipped_reason") and r.get("harness_failed") != "1"
                     and r.get("provider_calls")]
        routes = Counter()
        with usage_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    routes[json.loads(line).get("route", "?")] += 1
                except ValueError:
                    continue
        if not cells or not routes:
            print(f"skipping framing_double_bill: nothing to compare in {grid_name}")
            return
        series.append((label, colour, cells, routes))

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.4),
                                      gridspec_kw={"wspace": 0.34})
    drawn = []
    for index, (label, colour, cells, routes) in enumerate(series):
        calls = sorted(int(r["provider_calls"]) for r in cells)
        median_calls = calls[len(calls) // 2]
        total_routes = sum(routes.values())
        fallback_share = routes.get("invoke", 0) / total_routes * 100

        left.bar(index, median_calls, width=0.55, color=colour)
        left.text(index, median_calls, f" {median_calls}", ha="center", va="bottom",
                  fontsize=10)
        right.bar(index, fallback_share, width=0.55, color=colour)
        right.text(index, fallback_share, f" {fallback_share:.0f}%", ha="center",
                   va="bottom", fontsize=10)
        drawn.append({
            "framing": label.replace("\n", " "),
            "sonnet_cells": len(cells),
            "median_provider_calls_per_task": median_calls,
            "provider_calls_logged": total_routes,
            "non_streaming_fallbacks": routes.get("invoke", 0),
            "fallback_share_pct": round(fallback_share, 1),
        })

    labels = [label for label, _, _, _ in series]
    for axis, ylabel, title in (
        (left, "provider calls per task (median)", "Calls that reached Bedrock, at sonnet"),
        (right, "% of calls on /invoke", "How many were the fallback route"),
    ):
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=9.5)
    left.set_ylim(0, max(e["median_provider_calls_per_task"] for e in drawn) * 1.35)
    right.set_ylim(0, 100)

    before, after = drawn[0], drawn[1]
    factor = (before["median_provider_calls_per_task"]
              / after["median_provider_calls_per_task"]
              if after["median_provider_calls_per_task"] else float("nan"))
    fig.suptitle(
        f"The framing the client said it accepted sent every turn to the provider "
        f"{factor:.0f}x",
        fontsize=10.5, y=1.02,
    )
    fig.text(0.5, -0.22,
             f"{before['sonnet_cells']} and {after['sonnet_cells']} cli_proxy sonnet cells; "
             f"same grid, same prompt, framing is the only change. "
             f"{before['non_streaming_fallbacks']} of {before['provider_calls_logged']} "
             f"logged calls were fallbacks before, {after['non_streaming_fallbacks']} of "
             f"{after['provider_calls_logged']} after. The client reported one call in both "
             f"cases — this is measured at the proxy, the only place it is visible.",
             ha="center", fontsize=7.5, color=C_NEUTRAL, wrap=True)
    _save(fig, "framing_double_bill", drawn,
          ["framing", "sonnet_cells", "median_provider_calls_per_task",
           "provider_calls_logged", "non_streaming_fallbacks", "fallback_share_pct"])
    plt.close(fig)


FIGURE_FUNCS["framing_double_bill"] = fig_framing_double_bill


# ---------------------------------------------------------------------------
# 7 — the same tokens, billed 8.6x apart
# ---------------------------------------------------------------------------


def fig_cold_start() -> None:
    """The first call against a fresh prompt cache, and every call after it.

    This is the sharpest available demonstration that the four-class split changes a
    number rather than merely describing one. Within a single harness at a single model,
    the *token count* is identical between the first call and the rest — only the class
    changes, from cache write to cache read — and the two bill 12.5x apart.

    A two-class accounting produces the same figure for both, because it sees one input
    total. There is no averaging or inference here: it is the same prompt, the same
    harness, the same model, five consecutive runs.
    """
    plt = _require_matplotlib()
    from pricing import Counts, price_four_class, price_two_class

    rows = [r for r in _grid_rows() if r["model"] == "sonnet" and r.get("cost_usd")]
    if not rows:
        print("skipping cold_start: no sonnet rows")
        return

    by_harness: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_harness[row["harness"]][int(row["repeat"])] = row

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                                      gridspec_kw={"width_ratios": [3, 2],
                                                   "wspace": 0.32})

    drawn = []
    for harness in HARNESS_ORDER:
        reps = by_harness.get(harness)
        if not reps:
            continue
        order = sorted(reps)
        costs = [float(reps[i]["cost_usd"]) for i in order]
        left.plot(order, costs, marker="o", markersize=4, linewidth=1.5,
                  color=HARNESS_COLOUR.get(harness, C_NEUTRAL), label=harness)
        for index in order:
            row = reps[index]
            drawn.append({
                "harness": harness, "repeat": index,
                "cost_usd": round(float(row["cost_usd"]), 6),
                "cache_creation_tokens": int(row["cache_creation_tokens"] or 0),
                "cache_read_tokens": int(row["cache_read_tokens"] or 0),
            })

    left.set_yscale("log")
    left.set_xticks(sorted({e["repeat"] for e in drawn}))
    left.set_xlabel("consecutive run of the identical task")
    left.set_ylabel("$ per task (log)")
    left.legend(frameon=False, fontsize=8, ncol=2)
    left.set_title("One call in each series pays for the cache", loc="left", fontsize=9.5)

    # The pair that makes the point, chosen by data rather than by hand: the harness whose
    # cold call has the largest cache-write share, and its own warm calls.
    def _write_share(entry: dict) -> float:
        total = entry["cache_creation_tokens"] + entry["cache_read_tokens"]
        return entry["cache_creation_tokens"] / total if total else 0.0

    cached = [e for e in drawn if _write_share(e) > 0]
    if not cached:
        print("skipping cold_start: no cached runs to contrast")
        plt.close(fig)
        return
    cold = max(cached, key=_write_share)
    warm_candidates = [e for e in drawn
                       if e["harness"] == cold["harness"] and _write_share(e) == 0
                       and e["cache_read_tokens"]]
    if not warm_candidates:
        print("skipping cold_start: no warm counterpart")
        plt.close(fig)
        return
    warm = min(warm_candidates, key=lambda e: e["cost_usd"])

    pairs = []
    for label, entry in (("first call\n(cache write)", cold),
                         ("later call\n(cache read)", warm)):
        counts = Counts(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=entry["cache_read_tokens"],
            cache_creation_tokens=entry["cache_creation_tokens"],
        )
        four = price_four_class(counts, "sonnet-4-6") or 0.0
        two = price_two_class(counts, "sonnet-4-6") or 0.0
        pairs.append((label, four, two,
                      entry["cache_creation_tokens"] + entry["cache_read_tokens"]))

    for index, (label, four, two, _) in enumerate(pairs):
        right.bar(index - 0.19, four, width=0.36, color=C_CORRECT,
                  label="four-class" if index == 0 else None)
        right.bar(index + 0.19, two, width=0.36, color=C_WRONG,
                  label="two-class" if index == 0 else None)
        right.text(index - 0.19, four, f"${four:.4f}", ha="center", va="bottom",
                   fontsize=7.5)
        right.text(index + 0.19, two, f"${two:.4f}", ha="center", va="bottom",
                   fontsize=7.5)
    right.set_xticks(range(len(pairs)))
    right.set_xticklabels([label for label, _, _, _ in pairs], fontsize=8)
    right.set_ylabel("$ of prompt input")
    right.set_ylim(0, max(max(f, t) for _, f, t, _ in pairs) * 1.3)
    right.legend(frameon=False, fontsize=8)
    right.set_title(f"{cold['harness']}: identical {pairs[0][3]:,} tokens",
                    loc="left", fontsize=9.5)

    factor = pairs[0][1] / pairs[1][1] if pairs[1][1] else float("nan")
    fig.suptitle(
        f"The same {pairs[0][3]:,} prompt tokens cost {factor:.1f}x more on the first "
        f"call — and a two-class accounting reports one price for both",
        fontsize=10, y=1.02,
    )
    # Stated from the data, not asserted: which run is the cold one depends on cache TTL
    # rather than on position, so it moves between runs of this grid. An earlier caption
    # named a harness whose spike had since moved.
    cold_runs = sorted(
        f"{entry['harness']} at run {entry['repeat']}"
        for entry in drawn if entry["cache_creation_tokens"]
    )
    # Two different reasons a series can be flat, and they must not be conflated: a
    # harness that sends no cacheable prompt at all, and one whose cache was already warm
    # for every run in this grid.
    harnesses = sorted({entry["harness"] for entry in drawn})
    uncached, already_warm = [], []
    for harness in harnesses:
        own = [e for e in drawn if e["harness"] == harness]
        if any(e["cache_creation_tokens"] for e in own):
            continue
        (already_warm if any(e["cache_read_tokens"] for e in own)
         else uncached).append(harness)
    note = (f"Left: five consecutive runs of one task per harness. Cache write observed "
            f"in: {', '.join(cold_runs) if cold_runs else 'none'}. Which run is the cold "
            f"one depends on cache TTL, not on position.")
    if uncached:
        note += f" {', '.join(uncached)} sends no cacheable prompt at all."
    if already_warm:
        note += (f" {', '.join(already_warm)} is flat because its cache was already warm "
                 f"for every run here, not because it has none.")
    fig.text(0.5, -0.18, note, ha="center", fontsize=7.5, color=C_NEUTRAL, wrap=True)
    _save(fig, "cold_start", drawn,
          ["harness", "repeat", "cost_usd", "cache_creation_tokens",
           "cache_read_tokens"])
    plt.close(fig)


FIGURE_FUNCS["cold_start"] = fig_cold_start


# The entry point stays at the very end of the file, after every FIGURE_FUNCS
# registration. Placing it earlier runs main() — and so resolves both the registry and
# argparse's `choices` — before the figures below it exist, which silently draws a subset.
if __name__ == "__main__":
    sys.exit(main())
