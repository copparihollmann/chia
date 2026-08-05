# harness_study — measuring what chia's telemetry gets wrong, and what it costs

A self-contained measurement directory. It exists to turn claims about chia's
accounting and its missing seams into numbers a reader can check, including the
numbers that came out against the claim.

Nothing here calls a provider. Every figure in this first set is computed from
telemetry that was **already recorded** by earlier experiments, so the whole set
costs $0 to reproduce.

```
pricing.py      the rate table, and the two ways of pricing a call that this compares
collect.py      readers: chia profiler log / aet run dirs / spec results.jsonl -> tidy CSV
plots.py        the figures; each reads a CSV and hardcodes nothing
data/           the committed CSVs
figures/        the generated PNGs, each beside the exact subset of rows it drew
```

## Reproducing

```bash
# Regenerate the CSVs from recorded telemetry (paths are yours to supply):
python examples/harness_study/collect.py \
    --aet-runs /path/to/aet/run/roots \
    --results-jsonl /path/to/results.jsonl \
    --chia-profile /tmp/ray/<job>/ChiaProfileCollector.log

# Draw everything, or one figure by name:
python examples/harness_study/plots.py
python examples/harness_study/plots.py pricing_error
```

`plots.py` needs `matplotlib`; nothing else here does. `collect.py`'s chia-profile
reader imports `chia.trace.aet_sink`, so run it from a checkout.

## Three rules the figures follow

1. **Every figure reads a CSV.** No number is typed into `plots.py`.
2. **Every figure ships its own source data.** `figures/<name>.csv` is exactly the
   rows and columns behind `figures/<name>.png`.
3. **Every proportion is annotated with its denominator.** A rate over 11 runs must
   not render like a rate over 1100.

## What the current figures show

Corpus: 157 recorded runs — 96 aet run directories plus 61 experiment cells from a
`results.jsonl` grid — carrying $163 of four-class-priced spend across 27.8 Mtok of
billed input.

### `cache_share` — the premise

65% of all billed input across the corpus is prompt cache rather than fresh input,
and 91 of 152 runs are **≥99% cache**. Those runs report single-digit
`input_tokens` and tens of thousands of cache tokens.

This is the figure the rest depends on. If the cache share were small, splitting the
input classes would be pedantry.

### `pricing_error` — and the result that qualifies the claim

Per run, pricing every input token at the fresh-input rate is wrong by **up to
10x**, in both directions: cache reads cost a tenth of fresh input (so a read-heavy
run is overstated) while cache writes carry a 25% premium (so a write-heavy run is
understated).

But over the whole corpus those errors largely cancel: the aggregate is only
**1.08x**. That is a weaker headline than "2.4x off" and it is the honest one. It
also sharpens the argument rather than dissolving it — a program-level spend check
would have looked fine while every *per-run* comparison was wrong by a factor of
several. A cost–quality frontier, or any claim of the form "model A is cheaper per
task than model B", is exactly a per-run comparison.

7 runs are excluded because they recorded only a merged cache total, which cannot be
split-priced at all; 14 more had no rate for their model. Both are stated on the
figure.

### `retry_visibility` — a negative result about the old telemetry

11 of 332 recorded cells retried at least once, 22 attempts in total. For **none** of
them are the tokens those attempts burned recoverable from the record: a retry was
an integer, and every backend cleared its per-attempt metadata before the next
attempt.

The y-axis is deliberately *visibility*, not lost dollars. Putting a dollar figure
on those attempts would mean inventing the number the fix exists to produce.

### `reimplementation_count` — gate G2

Counting independent implementations of the same behaviour across chia, aet,
oscar-merlin and spec: the per-call agent sandbox exists **3 times**, the token price
table **3 times**, the Bedrock model registry **3 times**, usage-key normalization
**3 times**, the Converse tool loop and opencode stream parsing **twice** each.

The sandbox case is the strongest: spec's copy carries a drift test that loads aet's
module *by path* and asserts both builders emit identical argv — a downstream project
defending itself against divergence from another downstream project, because neither
could rely on the framework.

### `deletion_ratio` — gate G3, unflatteringly

Downstream workaround lines deleted per upstream source line added:

| PR | ratio | why |
|---|---|---|
| PR2 batched `get()` | 2.30 | deletes a 23-line re-unwrapping shim for a 10-line fix |
| PR1 `register_backend()` | 0.45 | deletes a monkeypatch of a private table |
| PR3 nested-session env | 0.10 | deletes a hand-rolled env scrub |
| PR5 `QueryResult.usage` | 0.04 | deletes a private-attribute reach-in, but adds a lot |
| PR4, PR6, PR7, PR8 | 0.00 | **no downstream workaround existed** |

The four zeros are the interesting rows. They are not cheap wins — they are gaps that
could not be worked around from outside chia at all, so they have to be justified by
G4 (a recorded number is wrong) rather than by lines saved. PR6 is the clearest case:
no amount of downstream code can recover tokens the framework already discarded.

A gate that only ever agrees with you is not a gate, so both the 1.08x aggregate and
these zeros are reported as they came out.

## Provenance and honesty notes

- The rate table in `pricing.py` is duplicated rather than imported from aet, so a
  figure cannot change under a dependency update. Each entry's source is in the
  comment beside it. Nova's cache rates are an explicit approximation.
- `collect.py` reads aet's scalars under aet's own last-occurrence-wins rule, because
  that is the rule the numbers were written under. Reading them any other way would
  measure the reader.
- The committed CSVs carry identity, token counts, cost and retries only. Free-text
  failure detail, prompt paths and target names are dropped: the figures do not need
  them, and a committed CSV should carry no more of an experiment than the claim it
  supports.
- `runs.csv` mixes two sources with different fidelity. Rows from a trajectory
  artifact that recorded only a merged cache figure carry it in
  `cache_unsplit_tokens` rather than being guessed into one of the two classes, and
  the figures that need the split exclude them and say so.

## The harness × model study (`grid.py`, live)

Run 2026-08-05, 24 cells (2 repeats × 3 models × 4 harnesses), **$0.54 of metered
spend** against a `--cap-usd 15` ceiling. `check_budget` ran before dispatch and, on a
second invocation, correctly reported `~$0.1189/call from 16 recorded` — the projection
learning from the run's own history, which is the whole design.

| harness | model | n | pass | $/task | median wall |
|---|---|---:|---:|---:|---:|
| `converse` | nova-lite | 2 | 2/2 | $0.00004 | 4.4 s |
| `converse` | glm5 | 2 | 2/2 | $0.0004 | 24.1 s |
| `converse` | sonnet | 2 | 2/2 | $0.0027 | 9.4 s |
| `cli_native` | sonnet | 2 | 2/2 | $0.0096 | 5.5 s |
| `cli_proxy` | sonnet | 2 | 2/2 | $0.0096 | 8.3 s |
| `cli_proxy` | nova-lite | 2 | 2/2 | $0.0856\* | 9.9 s |
| `cli_proxy` | glm5 | 2 | 2/2 | $0.1605\* | 15.8 s |
| `cli_native` | nova-lite, glm5 | — | — | — | structurally impossible |
| `opencode` | sonnet | 2 | — | — | harness failed on this host |

### What it shows

**The proxy works, live.** `cli_proxy` × `glm5` and `cli_proxy` × `nova-lite` are 2/2 on
a graded task, driven by the real Claude Code CLI. Those are the two cells `cli_native`
cannot reach at all — the skip rows are not missing data, they *are* the finding.

**The proxy costs nothing at the same model.** `cli_native` and `cli_proxy` both come to
$0.0096/task on sonnet. Translation is free; what is not free is the harness.

**The harness dominates cost at fixed model.** Sonnet costs $0.0027/task through raw
Converse and $0.0096 through the CLI — **3.6×** — because the CLI sends a ~23 KB system
prompt plus ten tool definitions before the task even starts. The right-hand panel is
that difference: ~275 billed input tokens for Converse against ~23,000 for the CLI. In
the very first (cold-cache) call the CLI cost **$0.089**, nine times its warm figure,
because those 23 k tokens were cache *writes* at a 25% premium; every later call reads
them at a tenth of the fresh rate. A two-class accounting cannot see either effect,
which is the point of `chia.base.usage`.

**So: why not just use opencode?** This run cannot answer that. opencode is installed on
this host but `opencode run` exits 1 with *"Unexpected server error"* and an empty
response, so its two cells are recorded as `harness_failed`, excluded from the figure, and
**not** reported as 0/2. A harness that never answered has not answered wrongly; plotting
an environment failure as a quality result is exactly the misleading claim this study
exists to avoid. The comparison against opencode remains open.

### Two caveats that are part of the result

\* **The CLI's self-reported cost for a proxied non-Anthropic model is wrong, and it is
marked `billed`.** The CLI prices what it thinks is a Claude call, so it reports $0.16 for
a GLM-5 task that raw Converse shows really costs $0.0004 — an overstatement of roughly
**400×**. `cost_source="billed"` then marks that figure as authoritative. For a proxied
model the honest cost has to come from the proxy's own Converse usage priced against the
real model's rates, not from the CLI. The `converse` rows are the trustworthy ones for
non-Anthropic models, and until that is fixed the starred figures should be read as
"Anthropic-equivalent", not as spend.

**`cli_proxy` reports no token counts.** Its billed-input bar is empty because the
proxy's SSE re-framing carries the CLI's cost but not its token breakdown back into
`QueryResult.usage`. The bar means "not recorded", not "sends nothing" — the request is
byte-identical to `cli_native`'s, whose 23 k tokens are measured.

### What was not run

The plan's confirmatory tier (spec's nl2spec grid, replayed from cassettes) and the
tier-mix ablation were not run. n=2 per cell is enough to establish the harness *cost*
effect, which is large and mechanical, and nowhere near enough for a quality comparison —
every proportion above is over 2 trials and is labelled as such.
