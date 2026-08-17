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

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

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

        # Persist the model where aet's cross-experiment `aet spend` rollup discovers a run's identity
        # (run_record/trajectory are not written by this token-only sink), so per-model attribution is
        # correct instead of bucketing the run under "(unknown)".
        if model:
            run_logger.log_params({"gen_ai.response.model": model})

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


# ---------------------------------------------------------------------------
# Streaming recorder — the per-turn / per-tool / per-attempt contract.
#
# ``record_run_usage`` above is the *aggregate* post-hoc path: one call at the
# end with summed usage. That loses the shape of a run (which turn spent what,
# which tools ran, which attempt failed and still burned tokens). The recorder
# below is fed incrementally as the Codex stream arrives, so a run is durable
# even if the process later dies, and every attempt's usage survives a retry.
#
# Two invariants, same as the aggregate path:
#   * AET is OPTIONAL — imported lazily, a no-op when absent.
#   * FAIL-OPEN — no recorder method ever raises into the run; a telemetry
#     hiccup is swallowed to a debug log.
# The durable local JSONL (usage/tools/attempts) is written whenever a run_dir
# is available and the recorder is enabled, independent of whether AET imports —
# that local stream is the replayable source of truth; AET is a mirror.
# ---------------------------------------------------------------------------


def _as_record(obj: Any) -> dict:
    """Coerce a typed record (``.as_dict()``) or a plain dict into a dict."""
    if isinstance(obj, dict):
        return obj
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        try:
            return as_dict()
        except Exception:
            return {}
    return {}


class CodexAetRecorder:
    """A streaming sink for one Codex run: per-turn, per-tool, per-attempt.

    Construct once per logical Codex call, feed it as events arrive
    (:meth:`record_turn` / :meth:`record_tool` / :meth:`record_attempt`), then
    :meth:`finish` with the final :class:`CodexRunResult`. Every method is
    fail-open: it returns ``False`` on any no-op or error and never raises.
    """

    def __init__(
        self,
        *,
        run_dir: Optional[os.PathLike | str] = None,
        run_id: str = "",
        model: str = "",
        project: str = "",
        suite: str = "",
        target: str = "",
        method: str = "",
        seed: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.enabled = is_enabled(enabled)
        self.model = model
        self.run_id = run_id or os.environ.get(_RUN_ID_ENV, "") or "chia_codex_run"
        self.project = project or os.environ.get(_PROJECT_ENV, "") or "chia"
        self.suite = suite or os.environ.get(_SUITE_ENV, "") or "default"
        self.target = target or os.environ.get(_TARGET_ENV, "") or "chia"
        self.method = method or os.environ.get(_METHOD_ENV, "") or "chia"
        if seed is None:
            try:
                seed = int(os.environ.get(_SEED_ENV, "0") or "0")
            except ValueError:
                seed = 0
        self.seed = seed
        resolved = run_dir or os.environ.get(_RUN_DIR_ENV) or ""
        self.run_dir = Path(resolved) if resolved else None
        self._agent_dir: Optional[Path] = None
        self._run_logger = None
        if self.enabled and self.run_dir is not None:
            self._agent_dir = self.run_dir / "agent"
            try:
                self._agent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.debug("codex recorder could not create agent dir (ignored): %s", exc)
                self._agent_dir = None
            self._run_logger = self._start_run_logger()

    # -- durable local JSONL ------------------------------------------------

    def _append(self, name: str, record: dict) -> bool:
        if self._agent_dir is None:
            return False
        try:
            with open(self._agent_dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            return True
        except Exception as exc:
            logger.debug("codex recorder local write failed (ignored): %s", exc)
            return False

    # -- optional AET mirror ------------------------------------------------

    def _start_run_logger(self):
        try:
            from aet.tracking.run_logger import EvalRunLogger
        except Exception:
            logger.debug("aet not importable; codex recorder AET mirror is a no-op.")
            return None
        try:
            run_logger = EvalRunLogger.start(
                project=self.project,
                suite=self.suite,
                target=self.target,
                method=self.method,
                seed=self.seed,
                run_id=self.run_id,
                run_path=self.run_dir,
                tracking_mode="local",
            )
            if self.model:
                run_logger.log_params({"gen_ai.response.model": self.model})
            return run_logger
        except Exception as exc:
            logger.debug("codex recorder could not start EvalRunLogger (ignored): %s", exc)
            return None

    def _safe(self, fn) -> bool:
        if self._run_logger is None:
            return False
        try:
            fn(self._run_logger)
            return True
        except Exception as exc:
            logger.debug("codex recorder AET call failed (ignored): %s", exc)
            return False

    # -- streaming API ------------------------------------------------------

    def record_thread(self, thread_id: str) -> bool:
        if not self.enabled or not thread_id:
            return False
        wrote = self._append("session.jsonl", {"thread_id": thread_id})
        self._safe(lambda rl: rl.log_params({"gen_ai.conversation.id": thread_id}))
        return wrote

    def record_turn(self, turn: Any) -> bool:
        """Record one completed turn's usage. Unreported turns are skipped."""
        if not self.enabled:
            return False
        rec = _as_record(turn)
        if not rec.get("reported"):
            return False  # unknown usage: nothing to bill, keep it out of totals
        wrote = self._append("usage.jsonl", rec)

        def _log(rl):
            rl.log_token_usage(
                int(rec.get("input_tokens") or 0),
                int(rec.get("output_tokens") or 0),
                cache_creation_tokens=int(rec.get("cache_write_input_tokens") or 0),
                cache_read_tokens=int(rec.get("cached_input_tokens") or 0),
                model=self.model,
            )
        mirrored = self._safe(_log)
        return wrote or mirrored

    def record_tool(self, tool: Any) -> bool:
        if not self.enabled:
            return False
        return self._append("tools.jsonl", _as_record(tool))

    def record_attempt(self, attempt: Any) -> bool:
        if not self.enabled:
            return False
        return self._append("attempts.jsonl", _as_record(attempt))

    def finish(self, run_result: Any = None, status: str = "completed") -> bool:
        if not self.enabled:
            return False
        cost = None
        wrote = False
        if run_result is not None:
            rec = _as_record(run_result)
            wrote = self._append("run_result.json", rec)
            cost = rec.get("cost_usd")
        if isinstance(cost, (int, float)):
            self._safe(lambda rl: rl.log_cost(float(cost), model=self.model))
        finished = self._safe(lambda rl: rl.finish(status))
        return wrote or finished
