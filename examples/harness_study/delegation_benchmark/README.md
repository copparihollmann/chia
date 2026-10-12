# Matched delegation task pack

This is the prepared, offline portion of the Claude Code versus OpenCode delegation study. It does **not** report a quality or provider-cost comparison. No paid model calls run from these scripts.

`tasks.json` freezes five real Chia repair tasks on commits before their fixes. Each prompt requires an explorer delegation and a code change. Held-out oracles check the outcome; they are outside the source snapshot shown to the agent. A paid run must sandbox each agent from the evaluator directory to preserve that holdout. The tasks concern CLI accounting and profiling, so a future result applies to this class of repository work, not every agentic workload.

Prepare and validate from the integrated Chia worktree:

```bash
export TMPDIR=/scratch/agustin/research/chia/tasks/chia-results-integration-20261012/.tmp
.venv/bin/python examples/harness_study/delegation_benchmark/benchmark.py verify
.venv/bin/python examples/harness_study/delegation_benchmark/benchmark.py schedule --seed 20261012 --repeats 5 --cap-usd 25
.venv/bin/python examples/harness_study/delegation_benchmark/benchmark.py prepare claude_usage
```

`verify` archives each pinned starting commit into `$TMPDIR`, confirms that its oracle fails, then confirms that the same oracle passes against the integrated solution. It creates no clone or Git worktree. `schedule` writes a fixed, randomized, 25-pair/50-arm order with oracle hashes. Both commands make no model request.

`relay.py` is the accounting component for a future paid run. Point both harnesses at separate loopback relay instances, one per arm. The relay forwards OpenAI-compatible Chat Completions to the selected provider and writes one privacy-safe JSONL row for **every** upstream request, including OpenCode's automatic title request. It records model, timestamps, HTTP status, fresh/cache-read/cache-write/output tokens and whether the provider supplied a complete usage block; it stores no prompts, responses or keys. A local fake-upstream test covers two ordinary requests and a streamed child request. An arm with missing provider usage cannot support a full-cost comparison.

`run_arms.py` now connects each scheduled arm to the relay. It archives a fresh
baseline, gives each CLI its own home and writable workspace, and runs the CLI
inside bubblewrap with its network namespace disconnected. A single Unix
socket forwards only the model endpoint to the host relay. The agent cannot
read the evaluator directory or reach the public fork; the oracle runs after
the agent exits. OpenCode's generated config is placed inside the bound
workspace. The runner captures the patch, answer, oracle outcome, public Chia
usage, the default Ray profiler log, and every provider request. A conservative
per-request reservation checks a supplied dollar cap before sending upstream.
If usage is incomplete, the schedule stops so unknown spending is not treated
as zero.

The runner passed a **no-spend installed-CLI smoke** for both harnesses in this
sandbox. Both made strong → weak → strong calls. The fake provider led to zero
code changes, so both oracle grades correctly failed. Claude's three provider
requests reconciled exactly with its public 30-input/9-output counters.
OpenCode made four requests; the extra 10-input/3-output title request was
visible in provider logs and absent from its public exported-session counters.
The default Chia profiler recorded Claude's child lifecycle and mixed-model
request, and OpenCode's three attributed turns plus a tool interval. These
are transport/accounting checks, **not** task-quality or real-cost results.

For a paid run, supply a JSON price table keyed by the exact upstream model
IDs, each with `input`, `read`, `write`, and `output` USD per million tokens.
Choose a provider that implements OpenAI-compatible Chat Completions and
returns complete usage in streamed and ordinary responses. The `run` command
requires an explicit provider credential environment variable, source schedule,
rate table, models, and dollar cap; start with `--max-arms 2` for one paired
pilot. This host currently has no configured paid provider key, and **no paid
arm has run**. See the task's `deliverables/REMAINING_STUDIES.md` for the
comparison and cache-TTL analysis plan.

An optional dated OpenAI candidate is
[`prices_openai_gpt41_2026-10-12.json`](prices_openai_gpt41_2026-10-12.json):
`gpt-4.1-2025-04-14` for the lead and `gpt-4.1-mini-2025-04-14` for the
explorer. Both pinned snapshots support Chat Completions, tool calling, and
streaming according to the [GPT-4.1 model page](https://developers.openai.com/api/docs/models/gpt-4.1)
and [GPT-4.1 mini model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
The table uses their listed standard input, cached-input, and output prices;
cache writes use the base input rate because [OpenAI's caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
says earlier models have no additional write charge. Verify availability and
prices again before a paid run. This candidate is not a measured harness result
or an approved budget.

After a complete run, `analyze.py <run-directory>` checks the 25 matched
pairs. It counts an outcome as delegated success only if the held-out oracle
passes **and** the weak model was called. It reports a paired success
difference and task-clustered bootstrap interval, then compares cost computed
from complete provider usage and the supplied current rates, and CLI
wall time on pairs where both succeeded. Incomplete arms or usage prevent a
full-study ranking. The local fixture is not a quality result.
