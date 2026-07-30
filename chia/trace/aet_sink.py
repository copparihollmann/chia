"""Guarded bridge from chia's per-run usage metadata to the aet eval harness.

chia's profiler records per-run telemetry (token counts, cost, turns) into an
ephemeral JSONL. This module additionally forwards that same normalized usage
into `aet <https://…>`'s :class:`~aet.tracking.run_logger.EvalRunLogger` so a run
is queryable alongside other aet evals.

Two invariants keep chia standalone:

* **aet is optional.** It is imported lazily; if it is not installed the sink is
  a silent no-op.
* **Opt-in.** The sink does nothing unless ``CHIA_AET_SINK=1`` (or an explicit
  ``enabled=True`` is passed), so default chia behavior is unchanged.

The run context (directory / project / suite / ids) comes from the caller when
available, otherwise from ``CHIA_AET_*`` environment variables. When no run
directory can be resolved there is nowhere to write, so the sink no-ops.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("chia.aet_sink")

# Env var that opts the sink in, plus the run-context fallbacks.
_ENABLE_ENV = "CHIA_AET_SINK"
_RUN_DIR_ENV = "CHIA_AET_RUN_DIR"
_RUN_ID_ENV = "CHIA_AET_RUN_ID"
_PROJECT_ENV = "CHIA_AET_PROJECT"
_SUITE_ENV = "CHIA_AET_SUITE"
_TARGET_ENV = "CHIA_AET_TARGET"
_METHOD_ENV = "CHIA_AET_METHOD"
_SEED_ENV = "CHIA_AET_SEED"


def is_enabled(enabled: Optional[bool] = None) -> bool:
    """Whether the sink should act. Explicit *enabled* overrides the env flag."""
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(_ENABLE_ENV, "").strip() in ("1", "true", "True", "yes")


def _has_usage(usage: dict) -> bool:
    """True when *usage* carries any token counts worth recording."""
    return any(
        isinstance(usage.get(k), (int, float)) and usage.get(k)
        for k in ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens")
    )


def record_run_usage(
    usage: dict,
    *,
    model: str = "",
    run_id: str = "",
    run_dir: Optional[os.PathLike | str] = None,
    project: str = "",
    suite: str = "",
    target: str = "",
    method: str = "",
    seed: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Record one completed run's normalized *usage* into aet.

    *usage* is a canonical usage dict (see :mod:`chia.models.usage`): the keys
    ``input_tokens``, ``output_tokens``, ``cache_read_input_tokens``,
    ``cache_creation_input_tokens``, ``reasoning_tokens``, ``cost_usd`` and
    ``num_turns`` are consumed; anything else is ignored.

    Returns ``True`` when metrics were written, ``False`` on any no-op (disabled,
    aet missing, no run directory, or no token counts). Never raises — a
    telemetry hiccup must not fail a run.
    """
    if not is_enabled(enabled):
        return False
    if not isinstance(usage, dict) or not _has_usage(usage):
        return False

    resolved_dir = run_dir or os.environ.get(_RUN_DIR_ENV) or ""
    if not resolved_dir:
        logger.debug("aet sink enabled but no run dir (set %s); skipping.", _RUN_DIR_ENV)
        return False

    try:
        from aet.tracking.run_logger import EvalRunLogger
    except Exception:
        logger.debug("aet not importable; aet sink is a no-op.")
        return False

    model = model or str(usage.get("model", "") or "")
    run_id = run_id or os.environ.get(_RUN_ID_ENV, "") or "chia_run"
    project = project or os.environ.get(_PROJECT_ENV, "") or "chia"
    suite = suite or os.environ.get(_SUITE_ENV, "") or "default"
    target = target or os.environ.get(_TARGET_ENV, "") or "chia"
    method = method or os.environ.get(_METHOD_ENV, "") or "chia"
    if seed is None:
        try:
            seed = int(os.environ.get(_SEED_ENV, "0") or "0")
        except ValueError:
            seed = 0

    def _as_int(key: str) -> int:
        val = usage.get(key, 0)
        return int(val) if isinstance(val, (int, float)) else 0

    try:
        run_path = Path(resolved_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        run_logger = EvalRunLogger.start(
            project=project,
            suite=suite,
            target=target,
            method=method,
            seed=seed,
            run_id=run_id,
            run_path=run_path,
            tracking_mode="local",
        )

        input_tokens = _as_int("input_tokens")
        output_tokens = _as_int("output_tokens")
        cache_read = _as_int("cache_read_input_tokens")
        cache_creation = _as_int("cache_creation_input_tokens")

        run_logger.log_token_usage(
            input_tokens,
            output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            model=model,
        )

        cost = usage.get("cost_usd")
        if isinstance(cost, (int, float)):
            run_logger.log_cost(float(cost), model=model)

        num_turns = _as_int("num_turns")
        if num_turns:
            run_logger.log_agent_turns(num_turns)

        # Per-model breakdown, when the class is available. Cost defaults to 0.0
        # here only for the structured record shape; the authoritative unknown-cost
        # handling already happened above (cost omitted entirely when unknown).
        try:
            from aet.tracking.claude_stream import ModelUsage

            run_logger.log_model_usage(ModelUsage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
                cost_usd=float(cost) if isinstance(cost, (int, float)) else 0.0,
            ))
        except Exception:
            pass

        run_logger.finish("completed")
        return True
    except Exception as exc:  # never let telemetry break a run
        logger.debug("aet sink failed (ignored): %s", exc)
        return False
