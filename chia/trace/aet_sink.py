"""Bridge one chia run's recorded telemetry into the aet eval harness.

`aet <https://github.com/ucb-bar/aet>`_ stores an eval run as a directory of
canonical logs, and its ``aet spend`` / ``aet runs`` / ``aet plot`` commands read
those. This module turns the events chia's profiler already collected into one
such run, so a chia experiment is queryable and plottable alongside other evals.

**A chia run is the unit, not a call.** aet's readers treat each metric as a final
scalar and take the *last* occurrence of a name — so N per-call writes of
``gen_ai.usage.input_tokens`` do not accumulate, they overwrite, and an N-call run
reports roughly 1/N of what it consumed. Everything here therefore folds the whole
run first (:func:`collect_run_usage`) and writes once (:func:`record_run`).

**It runs on the driver, reading the collector.** The profiler's collector actor
(:class:`~chia.trace.profiler.ProfileCollectorActor`) already holds every call's
event, and aet's local backend appends to files under one run directory. Writing
from Ray workers instead would mean concurrent read-modify-write on the same
``params.json`` from several processes — and on a multi-node cluster, several run
directories that silently diverge. Reading the collector needs neither lock nor
protocol: the events are already there, in order.

Two invariants keep chia standalone and predictable:

* **aet is optional.** It is imported lazily; without it the sink is a no-op. Install
  ``chia[aet]`` to enable it.
* **Opt-in.** Nothing happens unless ``CHIA_AET_SINK=1`` (or an explicit
  ``enabled=True``), so default chia behavior is unchanged.

Run identity (directory, ids, suite) comes from the caller, else from ``CHIA_AET_*``
environment variables. With no run directory there is nowhere to write, so the sink
no-ops rather than inventing a path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chia.base.usage import BILLING_MODES, TokenUsage, sum_usages

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

#: Profiler event types whose ``extra`` can carry a call's usage metadata — the
#: remote-task and local-call completion events respectively.
USAGE_EVENT_TYPES = frozenset({"complete", "local_end"})

#: Event type emitted for each retried attempt (see
#: :meth:`chia.base.llm_call.LLMCallBase.note_retry`).
RETRY_EVENT_TYPE = "llm_retry"

#: Metric name for the dollar-equivalent of subscription quota consumed.
#:
#: Deliberately **not** ``aet.agent.cost_usd``, which aet's spend rollup treats as
#: money and sums into a bill. A seat-authenticated CLI reports what a call would
#: have cost on the metered API; adding that to real charges yields a figure that
#: is neither spend nor quota. Keeping it under its own name means ``aet spend``
#: reports the metered total correctly and the quota consumption stays visible
#: without corrupting it.
SUBSCRIPTION_COST_METRIC = "chia.subscription.cost_equivalent_usd"

#: Metric names for the run-shape counters chia contributes beyond aet's own.
LLM_CALLS_METRIC = "chia.llm_calls"
RETRIES_METRIC = "chia.retries"


def is_enabled(enabled: Optional[bool] = None) -> bool:
    """Whether the sink should act. Explicit *enabled* overrides the env flag.

    :param enabled: ``True``/``False`` to force, ``None`` to read the environment.
    :type enabled: Optional[bool]
    :rtype: bool
    """
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(_ENABLE_ENV, "").strip() in ("1", "true", "True", "yes")


# ---------------------------------------------------------------------------
# Folding events into a run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallUsage:
    """One LLM call's contribution to a run.

    :param call_id: The profiler's call id, used to de-duplicate.
    :param func: Qualified name of the ``@ChiaFunction`` that made the call.
    :param ts: Wall-clock completion timestamp, for ordering the trajectory.
    :param usage: What the call consumed.
    :type call_id: str
    :type func: str
    :type ts: float
    :type usage: TokenUsage
    """

    call_id: str
    func: str
    ts: float
    usage: TokenUsage


@dataclass(frozen=True)
class RunUsage:
    """Everything one chia run consumed, and the per-call series behind it.

    :param calls: Per-call samples, ordered by completion time.
    :param retries: Number of retried attempts recorded during the run.
    :type calls: Tuple[CallUsage, ...]
    :type retries: int

    Totals are exposed per :data:`~chia.base.usage.BILLING_MODES` rather than as one
    number, because metered spend and subscription quota are different units and
    :meth:`~chia.base.usage.TokenUsage.__add__` refuses to combine them.
    """

    calls: Tuple[CallUsage, ...] = ()
    retries: int = 0

    @property
    def by_billing_mode(self) -> Dict[str, TokenUsage]:
        """Totals bucketed by billing mode. Only non-empty buckets appear.

        :rtype: Dict[str, TokenUsage]
        """
        buckets: Dict[str, List[TokenUsage]] = {}
        for call in self.calls:
            buckets.setdefault(call.usage.billing_mode, []).append(call.usage)
        return {mode: sum_usages(items) for mode, items in buckets.items()}

    @property
    def metered(self) -> TokenUsage:
        """The per-token-billed total — the one that may be summed into a bill."""
        return self.by_billing_mode.get("per_token", TokenUsage())

    @property
    def subscription(self) -> TokenUsage:
        """The subscription-quota total, in dollar-equivalent terms."""
        return self.by_billing_mode.get(
            "subscription", TokenUsage(billing_mode="subscription")
        )

    def per_model(self, mode: str = "per_token") -> Dict[str, TokenUsage]:
        """Totals split by model id within one billing mode.

        :param mode: One of :data:`~chia.base.usage.BILLING_MODES`.
        :type mode: str
        :rtype: Dict[str, TokenUsage]

        The split exists so a run that delegated to a cheaper model contributes to
        *each* model's spend rather than only to its identity model. It is scoped to
        one billing mode for the same reason the totals are: a model driven both
        through a seat and through a metered endpoint has two figures, not one sum.
        """
        buckets: Dict[str, List[TokenUsage]] = {}
        for call in self.calls:
            if call.usage.model and call.usage.billing_mode == mode:
                buckets.setdefault(call.usage.model, []).append(call.usage)
        return {model: sum_usages(items) for model, items in buckets.items()}

    @property
    def dominant_model(self) -> str:
        """The model that consumed the most tokens, or ``""`` when there were none.

        Used as the run's identity model where aet wants a single value; the full
        split is still recorded via :meth:`per_model`. Counted directly rather than
        via the per-mode totals, so the busiest model is found even on a run that
        mixed billing modes.
        """
        totals: Dict[str, int] = {}
        for call in self.calls:
            if call.usage.model:
                totals[call.usage.model] = (
                    totals.get(call.usage.model, 0) + call.usage.total_tokens
                )
        if not totals:
            return ""
        return max(totals.items(), key=lambda kv: kv[1])[0]

    def trajectory(self, mode: str = "per_token") -> List[dict]:
        """Cumulative per-call samples for one billing mode, oldest first.

        :param mode: One of :data:`~chia.base.usage.BILLING_MODES`.
        :type mode: str
        :returns: One dict per call, with ``t_s`` relative to the first call and
            cumulative token/cost figures.
        :rtype: List[dict]

        This is the series aet's ``log_trajectory_point`` consumes, and emitting it
        is what makes ``aet plot`` work for a chia run at all — a scalars-only run
        has no curve to draw.
        """
        calls = [c for c in self.calls if c.usage.billing_mode == mode]
        if not calls:
            return []
        t0 = calls[0].ts
        # Seeded from the first call rather than from an empty TokenUsage: an empty
        # one carries cost_usd=None, which would poison every running cost (see
        # TokenUsage.__add__).
        running: Optional[TokenUsage] = None
        points: List[dict] = []
        for index, call in enumerate(calls):
            running = call.usage if running is None else running + call.usage
            points.append({
                "index": index,
                "t_s": max(0.0, call.ts - t0),
                "cum_input": running.input_tokens,
                "cum_output": running.output_tokens,
                "cum_cache": (running.cache_read_input_tokens
                              + running.cache_creation_input_tokens),
                # An unknown running cost is reported as 0.0 with the point flagged
                # provisional, because the curve needs a number; `provisional_cost`
                # is how aet already marks a point whose cost is not authoritative.
                "cum_cost": running.cost_usd if running.cost_usd is not None else 0.0,
                "provisional_cost": running.cost_source != "billed",
            })
        return points


def _usage_from_event(event: dict) -> Optional[TokenUsage]:
    """Build a :class:`TokenUsage` from one profiler event, or ``None``.

    ``None`` means the event is not an LLM call — most profiler events are ordinary
    ``@ChiaFunction`` completions with no usage metadata attached.
    """
    if event.get("type") not in USAGE_EVENT_TYPES:
        return None
    extra = event.get("extra")
    if not isinstance(extra, dict):
        return None
    usage = TokenUsage.from_metadata(
        extra,
        model=str(extra.get("model", "") or ""),
        billing_mode=(
            extra["billing_mode"]
            if extra.get("billing_mode") in BILLING_MODES
            else "per_token"
        ),
    )
    if not usage.total_tokens and usage.cost_usd is None:
        return None
    # A cost_source recorded by the backend is authoritative: it knows whether the
    # provider reported the figure. Re-deriving it here from a flat dict would
    # downgrade a billed cost to "estimated".
    if extra.get("cost_source") in ("billed", "estimated", "unavailable"):
        usage = usage.replace(cost_source=extra["cost_source"])
    return usage


def collect_run_usage(events: Iterable[dict]) -> RunUsage:
    """Fold profiler *events* into one run's usage.

    :param events: Profiler events, as recorded by the collector actor.
    :type events: Iterable[dict]
    :rtype: RunUsage

    Pure: no aet, no Ray, no filesystem — so the accumulation can be tested against
    a literal event list, which is where the per-call-vs-per-run bug lived.

    Events without usage metadata are skipped, and a repeated ``call_id`` is counted
    once (an event stream may legitimately be read more than once).
    """
    calls: List[CallUsage] = []
    seen: set = set()
    retries = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == RETRY_EVENT_TYPE:
            retries += 1
            continue
        usage = _usage_from_event(event)
        if usage is None:
            continue
        call_id = str(event.get("call_id", "") or f"_{len(calls)}")
        if call_id in seen:
            continue
        seen.add(call_id)
        calls.append(CallUsage(
            call_id=call_id,
            func=str(event.get("func", "") or ""),
            ts=float(event.get("ts", 0.0) or 0.0),
            usage=usage,
        ))

    calls.sort(key=lambda c: c.ts)
    return RunUsage(calls=tuple(calls), retries=retries)


# ---------------------------------------------------------------------------
# Writing the run
# ---------------------------------------------------------------------------


def _collector_events(namespace: Optional[str] = None) -> Optional[List[dict]]:
    """Read every event out of the profiler's collector actor, or ``None``."""
    try:
        import ray

        from chia.trace.profiler import get_collector

        collector = get_collector(namespace=namespace)
        if collector is None:
            return None
        return list(ray.get(collector.get_events.remote()))
    except Exception as exc:
        logger.debug("could not read the profile collector (%s); aet sink skipped.", exc)
        return None


def _default_run_id() -> str:
    """A run id that does not collide across chia jobs.

    The previous default (``"chia_run"``) meant every run wrote the same id, so
    ``aet spend`` / ``aet runs`` could not tell two experiments apart. The Ray job
    id is unique per job and is the natural scope for a chia run.
    """
    try:
        import ray

        job_id = ray.get_runtime_context().get_job_id()
        if job_id:
            return f"chia_{job_id}"
    except Exception:
        pass
    return f"chia_{os.getpid()}"


def record_run(
    events: Optional[Sequence[dict]] = None,
    *,
    run_dir: Optional[os.PathLike | str] = None,
    run_id: str = "",
    project: str = "",
    suite: str = "",
    target: str = "",
    method: str = "",
    seed: Optional[int] = None,
    model: str = "",
    extra: Optional[dict] = None,
    enabled: Optional[bool] = None,
    namespace: Optional[str] = None,
) -> bool:
    """Write one chia run into aet. Call once, from the driver, after the work.

    :param events: Profiler events to fold. Defaults to reading the collector actor.
    :param run_dir: Destination run directory. Defaults to ``$CHIA_AET_RUN_DIR``.
    :param run_id: Run identity. Defaults to ``$CHIA_AET_RUN_ID``, else a
        job-scoped id.
    :param project: aet project name. Defaults to ``$CHIA_AET_PROJECT``, else ``"chia"``.
    :param suite: aet suite name. Defaults to ``$CHIA_AET_SUITE``, else ``"default"``.
    :param target: What was being built/evaluated. Defaults to ``$CHIA_AET_TARGET``.
    :param method: The method/agent under test. Defaults to ``$CHIA_AET_METHOD``.
    :param seed: Run seed. Defaults to ``$CHIA_AET_SEED``, else ``0``.
    :param model: Identity model for the run. Defaults to whichever model consumed
        the most tokens.
    :param extra: Extra fields merged into ``run_record.json``.
    :param enabled: Force the sink on/off, bypassing ``CHIA_AET_SINK``.
    :param namespace: Ray namespace for the collector lookup.
    :type events: Optional[Sequence[dict]]
    :type run_dir: Optional[os.PathLike | str]
    :type run_id: str
    :type project: str
    :type suite: str
    :type target: str
    :type method: str
    :type seed: Optional[int]
    :type model: str
    :type extra: Optional[dict]
    :type enabled: Optional[bool]
    :type namespace: Optional[str]
    :returns: ``True`` when a run was written; ``False`` on any no-op (disabled, aet
        missing, no run directory, no collector, or nothing recorded).
    :rtype: bool

    Never raises — a telemetry failure must not fail an experiment that already
    completed.
    """
    if not is_enabled(enabled):
        return False

    resolved_dir = run_dir or os.environ.get(_RUN_DIR_ENV) or ""
    if not resolved_dir:
        logger.debug("aet sink enabled but no run dir (set %s); skipping.", _RUN_DIR_ENV)
        return False

    if events is None:
        events = _collector_events(namespace)
        if events is None:
            return False

    run = collect_run_usage(events)
    if not run.calls:
        logger.debug("aet sink found no LLM-call usage in %d events; skipping.",
                     len(events))
        return False

    try:
        from aet.tracking.run_logger import EvalRunLogger
    except Exception:
        logger.debug("aet not importable; aet sink is a no-op. Install chia[aet].")
        return False

    run_id = run_id or os.environ.get(_RUN_ID_ENV, "") or _default_run_id()
    project = project or os.environ.get(_PROJECT_ENV, "") or "chia"
    suite = suite or os.environ.get(_SUITE_ENV, "") or "default"
    target = target or os.environ.get(_TARGET_ENV, "") or "chia"
    method = method or os.environ.get(_METHOD_ENV, "") or "chia"
    if seed is None:
        try:
            seed = int(os.environ.get(_SEED_ENV, "0") or "0")
        except ValueError:
            seed = 0
    model = model or run.dominant_model

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
        _write_run(run_logger, run, model=model, extra=extra)
        run_logger.finish("completed")
        return True
    except Exception as exc:  # never let telemetry break a run
        logger.debug("aet sink failed (ignored): %s", exc)
        return False


def _write_run(run_logger, run: RunUsage, *, model: str, extra: Optional[dict]) -> None:
    """Emit one folded :class:`RunUsage` through an already-started aet logger."""
    metered = run.metered
    subscription = run.subscription

    # run_record.json carries the run's identity. Without it aet's readers fall back
    # to inferring run_id and suite from directory names, which is how two chia runs
    # ended up indistinguishable.
    run_logger.write_run_record(extra={
        "source": "chia",
        "llm_calls": len(run.calls),
        "retries": run.retries,
        "billing_modes": sorted(run.by_billing_mode),
        **(extra or {}),
    })

    if model:
        run_logger.log_params({"gen_ai.response.model": model})
    run_logger.log_params({
        "chia.billing_mode": ",".join(sorted(run.by_billing_mode)),
        "chia.cost_source": metered.cost_source,
    })

    run_logger.log_token_usage(
        metered.input_tokens,
        metered.output_tokens,
        cache_creation_tokens=metered.cache_creation_input_tokens,
        cache_read_tokens=metered.cache_read_input_tokens,
        model=model,
    )
    # Only a known cost is recorded. An unpriced run is left for aet to count in
    # `unpriced_runs`; writing 0.0 would deflate a program-wide total silently.
    if metered.cost_usd is not None:
        run_logger.log_cost(float(metered.cost_usd), model=model)
    if subscription.cost_usd is not None:
        run_logger.log_metric(SUBSCRIPTION_COST_METRIC, float(subscription.cost_usd))

    turns = metered.num_turns + subscription.num_turns
    if turns:
        run_logger.log_agent_turns(turns)
    run_logger.log_metric(LLM_CALLS_METRIC, len(run.calls))
    run_logger.log_metric(RETRIES_METRIC, run.retries)

    _write_per_model(run_logger, run)
    _write_trajectory(run_logger, run)


def _write_per_model(run_logger, run: RunUsage) -> None:
    """Record the per-model split aet's spend rollup reads."""
    try:
        from aet.tracking.claude_stream import ModelUsage
    except Exception:
        return
    for model_id, usage in run.per_model().items():
        try:
            run_logger.log_model_usage(ModelUsage(
                model=model_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                cost_usd=float(usage.cost_usd) if usage.cost_usd is not None else 0.0,
            ))
        except Exception as exc:
            logger.debug("per-model usage for %s not recorded: %s", model_id, exc)


def _write_trajectory(run_logger, run: RunUsage) -> None:
    """Record the cumulative per-call series, which is what ``aet plot`` draws."""
    for point in run.trajectory():
        run_logger.log_trajectory_point(
            index=point["index"],
            t_s=point["t_s"],
            cum_input=point["cum_input"],
            cum_output=point["cum_output"],
            cum_cache=point["cum_cache"],
            cum_cost=point["cum_cost"],
            provisional_cost=point["provisional_cost"],
        )
