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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from chia.base.usage import BILLING_MODES, TokenUsage, sum_usages
from chia.trace.stream_telemetry import METADATA_KEY as STREAM_TELEMETRY_KEY

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
    # Wall-clock length of the call, used to place this call's activity bands on the
    # run-wide clock: a stream's own t=0 is the call's start, not the run's.
    duration_s: float = 0.0
    # The per-call summary derived on the worker by chia.trace.stream_telemetry — activity
    # bands, tool counts, per-model usage. Empty when the aet extra is absent or when the
    # backend is not the Claude Code CLI.
    telemetry: Dict[str, Any] = field(default_factory=dict)


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

    def clock_origin(self, mode: str = "per_token") -> Optional[float]:
        """Wall-clock zero for this run's series: the first call's **start**.

        Not its completion. A call's completion timestamp is the only one the profiler
        records, so the origin used to be the first *completion* — which put t=0 after the
        first call's work was already done, and left that call's tool activity at negative
        time. A run is described as starting when it started.

        Falls back to the first completion when no call duration was recorded, so a run
        collected before per-call durations existed reconstructs exactly as it did before.
        """
        calls = [c for c in self.calls if c.usage.billing_mode == mode]
        if not calls:
            return None
        return calls[0].ts - float(calls[0].duration_s or 0.0)

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

        Cache tokens are reported three ways: the ``cum_cache`` sum aet's cost model
        bills as one class, plus the read and creation halves separately. The halves
        are not redundant — a read is billed at 0.1× the input rate and a write at
        1.25×, so two runs with the same ``cum_cache`` can differ 12.5-fold in what
        that cache cost, and the sum alone cannot tell a warm run from a cold one.
        """
        calls = [c for c in self.calls if c.usage.billing_mode == mode]
        if not calls:
            return []
        t0 = self.clock_origin(mode)
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
                "cum_cache_read": running.cache_read_input_tokens,
                "cum_cache_creation": running.cache_creation_input_tokens,
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
        extra = event.get("extra") or {}
        telemetry = extra.get(STREAM_TELEMETRY_KEY)
        calls.append(CallUsage(
            call_id=call_id,
            func=str(event.get("func", "") or ""),
            ts=float(event.get("ts", 0.0) or 0.0),
            usage=usage,
            duration_s=float(extra.get("duration_s", 0.0) or 0.0),
            telemetry=telemetry if isinstance(telemetry, dict) else {},
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
        _write_run(run_logger, run, model=model, extra=extra, run_id=run_id,
                   run_path=run_path)
        export_profile_jsonl(events, run_path / "agent" / "chia_profile.jsonl")
        _materialize_structured_profile(run_path, run_id=run_id)
        run_logger.finish("completed")
        return True
    except Exception as exc:  # never let telemetry break a run
        logger.debug("aet sink failed (ignored): %s", exc)
        return False


def export_profile_jsonl(events: Sequence[dict], path: os.PathLike | str) -> int:
    """Write only stable privacy-safe profile records for an external profiler.

    The exporter intentionally rejects ordinary profiler ``extra`` metadata:
    callers may put arbitrary domain data there, while ``chia.agent_profile``
    records have a closed schema that excludes prompt/tool payloads.
    """
    selected = [event for event in events if event.get("schema") == "chia.agent_profile"]
    if not selected:
        return 0
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".partial")
    temporary.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in selected))
    temporary.replace(out)
    return len(selected)


def _materialize_structured_profile(run_path: Path, *, run_id: str) -> bool:
    """Let a compatible AET enrich its trajectory from the exported JSONL."""
    profile = run_path / "agent" / "chia_profile.jsonl"
    if not profile.is_file():
        return False
    try:
        from aet.trajectory.importers.chia import import_chia

        import_chia(profile, run_id=run_id).to_json(run_path / "metrics" / "trajectory.json")
        return True
    except Exception as exc:
        logger.debug("structured AET profile not materialized (ignored): %s", exc)
        return False


def _write_run(run_logger, run: RunUsage, *, model: str, extra: Optional[dict],
               run_id: str = "", run_path=None) -> None:
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
    _write_tool_metrics(run_logger, run)
    _write_trajectory(run_logger, run, model=model, run_id=run_id, run_path=run_path)


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


def _write_trajectory(run_logger, run: RunUsage, *, model: str = "",
                      run_id: str = "", run_path=None) -> None:
    """Record the cumulative per-call series, which is what ``aet plot`` draws.

    Three things go out, not one, because aet's reader needs all three to reconstruct a
    trajectory that renders truthfully:

    * the per-call points, including the cache read/creation split;
    * one ``aet.traj.round`` boundary spanning the run. Without a round, the reader's
      ``duration_s`` and ``num_rounds`` both fall back to 0 and the figure titles itself
      "0 rounds · $X · 0 min" — a run that plainly took minutes described as taking none;
    * the ``aet.traj.summary`` param holding the run-level totals, which is where the
      reader looks before falling back to the last point.

    A chia run is emitted as a single round: the unit of work chia schedules is the run,
    and inventing per-call rounds would draw dividers that correspond to nothing an agent
    did. The verdict fields are left unset rather than defaulted to a pass — chia does not
    run the oracle, and ``n_passed=0`` would render as a failing run.
    """
    points = run.trajectory()
    if not points:
        return

    for point in points:
        _log_point(run_logger, point)

    metered = run.metered
    last = points[-1]
    summary = {
        "run_id": run_id,
        "source": "chia",
        "model": model,
        "duration_s": last["t_s"],
        "num_rounds": 1,
        "final_cost_usd": last["cum_cost"],
        "final_input_tokens": last["cum_input"],
        "final_output_tokens": last["cum_output"],
        "final_cache_tokens": last["cum_cache"],
        "final_cache_read_tokens": last["cum_cache_read"],
        "final_cache_creation_tokens": last["cum_cache_creation"],
    }
    try:
        from aet.trajectory.model import RoundBoundary

        run_logger.log_round_boundary(RoundBoundary(
            index=0,
            t_start_s=0.0,
            t_end_s=last["t_s"],
            cost_usd=last["cum_cost"],
            input_tokens=int(metered.input_tokens),
            output_tokens=int(metered.output_tokens),
            cache_tokens=int(last["cum_cache"]),
        ))
    except Exception as exc:
        logger.debug("aet round boundary not recorded: %s", exc)
    try:
        run_logger.log_param("aet.traj.summary", summary)
    except Exception as exc:
        logger.debug("aet trajectory summary not recorded: %s", exc)

    # The fast-path artifact, written last because it is the only one that can carry the
    # activity bands: aet's logs/ reconstruction restores points, rounds and milestones but
    # documents bands as out of reach ("they belong to tool events"). Everything above is
    # still written when this is skipped, so a run without the telemetry loses the activity
    # view and nothing else.
    if run_path is not None:
        _write_trajectory_json(run_path, run, summary, points, _activity_bands(run))


def _activity_bands(run: RunUsage, mode: str = "per_token") -> List[dict]:
    """Every call's activity bands, placed on the run-wide clock.

    Each call's stream numbers its tool calls from that call's own t=0, so a run of eight
    prompts would otherwise stack eight overlapping sets of bands at the start of the
    figure. The offset is the call's start — its completion timestamp minus its own
    duration — measured from the first call in the run.
    """
    calls = [c for c in run.calls if c.usage.billing_mode == mode]
    if not calls:
        return []
    # The SAME origin the points use (see RunUsage.clock_origin). Bands and the spend curve
    # are drawn on one x-axis, so they must share a zero — placing bands on a start-time clock
    # while the curve was on a completion-time clock would slide the activity view relative to
    # the spend it exists to explain, and would put the first call's tools at negative time.
    t0 = run.clock_origin(mode)
    bands: List[dict] = []
    for call in calls:
        raw = call.telemetry.get("bands") if call.telemetry else None
        if not raw:
            continue
        # A call with no recorded duration cannot be placed better than at its completion.
        start = (call.ts - call.duration_s) - t0
        for band in raw:
            try:
                t_start = float(band["t0_s"]) + start
                t_end = float(band["t1_s"]) + start
            except (KeyError, TypeError, ValueError):
                continue
            if t_end <= 0.0:
                # Entirely before the run clock's zero. Dropped rather than squashed to
                # [0, 0]: a zero-length band at the origin is a claim that something
                # happened there instantaneously, which is not what was measured.
                continue
            bands.append({**band, "t0_s": max(0.0, t_start), "t1_s": t_end})
    bands.sort(key=lambda b: b["t0_s"])
    return bands


def _write_trajectory_json(run_path, run: RunUsage, summary: dict, points: List[dict],
                           bands: List[dict]) -> bool:
    """Write ``metrics/trajectory.json`` — the only run-dir artifact that carries bands.

    ``RunTrajectory.from_run_dir`` prefers this file and falls back to replaying ``logs/``.
    That fallback is documented in aet as unable to restore activity bands ("they belong to
    tool events"), so a sink that writes only ``logs/`` can never produce an activity view
    no matter what it records. Writing the fast path is what puts the bands in reach.

    Returns False, quietly, when aet is absent or the write fails: a missing figure must
    not fail a run, and ``logs/`` has already been written by this point either way.
    """
    if not points:
        return False
    try:
        from aet.trajectory.model import ActivityBand, RoundBoundary, RunTrajectory, TrajectoryPoint
    except Exception:
        return False
    try:
        traj = RunTrajectory(
            run_id=summary.get("run_id", ""),
            duration_s=float(summary.get("duration_s", 0.0)),
            num_rounds=int(summary.get("num_rounds", 1)),
            points=[TrajectoryPoint(
                t_s=float(p["t_s"]),
                cum_input_tokens=float(p["cum_input"]),
                cum_output_tokens=float(p["cum_output"]),
                cum_cache_tokens=float(p["cum_cache"]),
                cum_cache_read_tokens=float(p["cum_cache_read"]),
                cum_cache_creation_tokens=float(p["cum_cache_creation"]),
                cum_cost_usd=float(p["cum_cost"]),
                provisional_cost=bool(p.get("provisional_cost", False)),
            ) for p in points],
            bands=[ActivityBand(
                t0_s=float(b["t0_s"]),
                t1_s=float(b["t1_s"]),
                category=str(b.get("category", "bash")),
                tool_name=str(b.get("tool_name", "")),
                weight=float(b.get("weight", 1.0)),
                is_error=bool(b.get("is_error", False)),
            ) for b in bands],
            # One round for the whole run, matching what logs/ records. The verdict fields
            # stay unset: chia runs no oracle, and a 0-of-N would render as a failing run.
            rounds=[RoundBoundary(index=0, t_start_s=0.0,
                                  t_end_s=float(summary.get("duration_s", 0.0)))],
            final_input_tokens=float(summary.get("final_input_tokens", 0.0)),
            final_output_tokens=float(summary.get("final_output_tokens", 0.0)),
            final_cache_tokens=float(summary.get("final_cache_tokens", 0.0)),
            final_cost_usd=float(summary.get("final_cost_usd", 0.0)),
        )
        out = Path(run_path) / "metrics" / "trajectory.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.partial")
        tmp.write_text(json.dumps(traj.to_dict(), indent=2, sort_keys=True))
        tmp.replace(out)                       # write-then-rename: never a half-read figure
        return True
    except Exception as exc:
        logger.debug("aet trajectory.json not written: %s", exc)
        return False


def _write_tool_metrics(run_logger, run: RunUsage) -> None:
    """Run-level tool counts, so "how much tool use" is answerable without the figure."""
    totals = {"tool_calls": 0, "tool_errors": 0}
    tools: set = set()
    seen_any = False
    for call in run.calls:
        t = call.telemetry
        if not t:
            continue
        seen_any = True
        totals["tool_calls"] += int(t.get("tool_calls", 0) or 0)
        totals["tool_errors"] += int(t.get("tool_errors", 0) or 0)
        tools.update(t.get("tools_used") or ())
    if not seen_any:
        # No stream telemetry at all is different from a run that used no tools, and
        # recording 0 would erase that difference.
        return
    try:
        run_logger.log_metric("chia.tool.calls", float(totals["tool_calls"]))
        run_logger.log_metric("chia.tool.errors", float(totals["tool_errors"]))
        run_logger.log_param("chia.tool.names", sorted(tools))
    except Exception as exc:
        logger.debug("tool metrics not recorded: %s", exc)


def _log_point(run_logger, point: dict) -> None:
    """One trajectory point, degrading to the pre-split signature on an older aet.

    ``chia[aet]`` tracks aet's default branch rather than a release, so the installed
    copy may predate the cache-split parameters. Losing the split is acceptable — the
    sum is still recorded and the curve still draws — but losing the whole trajectory to
    a ``TypeError`` is not.
    """
    common = dict(
        index=point["index"],
        t_s=point["t_s"],
        cum_input=point["cum_input"],
        cum_output=point["cum_output"],
        cum_cache=point["cum_cache"],
        cum_cost=point["cum_cost"],
        provisional_cost=point["provisional_cost"],
    )
    try:
        run_logger.log_trajectory_point(
            **common,
            cum_cache_read=point["cum_cache_read"],
            cum_cache_creation=point["cum_cache_creation"],
        )
    except TypeError:
        run_logger.log_trajectory_point(**common)


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
        raw = _as_record(tool)
        safe = {key: raw.get(key) for key in (
            "item_id", "item_type", "kind", "status", "exit_code",
            "started_at", "completed_at",
        ) if raw.get(key) is not None}
        return self._append("tools.jsonl", safe)

    def record_attempt(self, attempt: Any) -> bool:
        if not self.enabled:
            return False
        raw = _as_record(attempt)
        turns = raw.get("turns") if isinstance(raw.get("turns"), list) else []
        safe = {key: raw.get(key) for key in (
            "index", "thread_id", "started_at", "ended_at", "exit_code",
            "signal", "timeout", "retry_reason", "failure_class",
        ) if raw.get(key) is not None}
        safe["usage"] = {
            key: sum(int(turn.get(key) or 0) for turn in turns if isinstance(turn, dict))
            for key in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                        "output_tokens", "reasoning_output_tokens")
        }
        return self._append("attempts.jsonl", safe)

    def finish(self, run_result: Any = None, status: str = "completed") -> bool:
        if not self.enabled:
            return False
        cost = None
        wrote = False
        if run_result is not None:
            rec = _as_record(run_result)
            safe = {key: rec.get(key) for key in (
                "thread_id", "resolved_model", "requested_model", "agent_version",
                "status", "active_wall_s", "price_snapshot_id", "total_usage",
                "usage_complete",
            ) if rec.get(key) is not None}
            wrote = self._append("run_result.json", safe)
            cost = rec.get("cost_usd")
        if isinstance(cost, (int, float)):
            self._safe(lambda rl: rl.log_cost(float(cost), model=self.model))
        finished = self._safe(lambda rl: rl.finish(status))
        return wrote or finished
