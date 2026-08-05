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
