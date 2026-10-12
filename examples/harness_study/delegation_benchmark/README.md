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

The remaining execution work is to connect the two installed CLIs to this relay for each prepared snapshot, define a provider/model pair and current price table, enforce an approved budget, then run and grade the 50 scheduled arms. The current host has no configured AWS profile or paid OpenCode provider key. See the task's `deliverables/REMAINING_STUDIES.md` for the outcome, cost, and cache-TTL analysis plan.
