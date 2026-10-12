"""Tests for the chia -> aet run sink (:mod:`chia.trace.aet_sink`).

Two layers, split by what they need:

* **Folding** — :func:`collect_run_usage` against literal profiler-event lists. Pure,
  no aet, no Ray, no filesystem. This is where the per-call-versus-per-run bug lived,
  so it is tested against data rather than through a writer.
* **Writing** — :func:`record_run` against a real ``EvalRunLogger``, reading the
  files back. Skipped without aet installed (``chia[aet]``).

The headline assertion is ``test_multi_call_run_records_the_sum_not_the_last_call``:
aet's readers take the *last* occurrence of each metric name, so a sink that fired
per call reported roughly 1/N of an N-call run.
"""

from __future__ import annotations

import json

import pytest

from chia.base.usage import TokenUsage
from chia.trace import aet_sink


# ---------------------------------------------------------------------------
# Event fixtures — the shape ProfileCollectorActor actually stores
# ---------------------------------------------------------------------------


def _call_event(call_id, ts, *, input_tokens=0, output_tokens=0, cache_read=0,
                cache_creation=0, cost_usd=None, cost_source=None, num_turns=1,
                model="anthropic.claude-sonnet-4-6", billing_mode="per_token",
                event_type="complete"):
    extra = {
        "model": model,
        "num_turns": num_turns,
        "billing_mode": billing_mode,
        # `tools` is always present in a real event and is not usage; it must pass
        # through the fold untouched.
        "tools": [{"name": "bash"}],
    }
    for key, value in (("input_tokens", input_tokens),
                       ("output_tokens", output_tokens),
                       ("cache_read_input_tokens", cache_read),
                       ("cache_creation_input_tokens", cache_creation)):
        if value:
            extra[key] = value
    if cost_usd is not None:
        extra["cost_usd"] = cost_usd
    if cost_source is not None:
        extra["cost_source"] = cost_source
    return {"type": event_type, "call_id": call_id, "func": "ClaudeCodeLLM.prompt",
            "ts": ts, "extra": extra}


def _plain_event(call_id, ts):
    """An ordinary @ChiaFunction completion — no usage metadata at all."""
    return {"type": "complete", "call_id": call_id, "func": "build_target",
            "ts": ts, "exec_time_s": 1.5}


def _retry_event(ts, attempt=1):
    return {"type": aet_sink.RETRY_EVENT_TYPE, "ts": ts, "attempt": attempt,
            "error_type": "ServerError", "backoff_s": 5.0}


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


def test_collect_sums_across_calls():
    run = aet_sink.collect_run_usage([
        _call_event("c1", 100.0, input_tokens=10, output_tokens=1, cost_usd=0.01,
                    cost_source="billed"),
        _call_event("c2", 101.0, input_tokens=20, output_tokens=2, cost_usd=0.02,
                    cost_source="billed"),
        _call_event("c3", 102.0, input_tokens=30, output_tokens=3, cost_usd=0.03,
                    cost_source="billed"),
    ])

    assert len(run.calls) == 3
    assert run.metered.input_tokens == 60
    assert run.metered.output_tokens == 6
    assert run.metered.cost_usd == pytest.approx(0.06)
    assert run.metered.num_turns == 3


def test_collect_ignores_events_without_usage():
    """Most profiler events are ordinary function completions; they must not be
    mistaken for zero-token LLM calls, which would pad the trajectory."""
    run = aet_sink.collect_run_usage([
        _plain_event("p1", 100.0),
        _call_event("c1", 101.0, input_tokens=10),
        _plain_event("p2", 102.0),
        {"type": "dispatch", "call_id": "c1", "ts": 100.5},
        "not even a dict",
    ])

    assert [c.call_id for c in run.calls] == ["c1"]


def test_collect_deduplicates_a_repeated_call_id():
    """The collector's event list may legitimately be read more than once; folding
    it twice must not double a run's spend."""
    event = _call_event("c1", 100.0, input_tokens=10, cost_usd=0.01)

    run = aet_sink.collect_run_usage([event, event])

    assert len(run.calls) == 1
    assert run.metered.input_tokens == 10


def test_collect_orders_calls_by_timestamp():
    """Workers report out of order; the trajectory has to be chronological."""
    run = aet_sink.collect_run_usage([
        _call_event("late", 300.0, input_tokens=1),
        _call_event("early", 100.0, input_tokens=1),
        _call_event("mid", 200.0, input_tokens=1),
    ])

    assert [c.call_id for c in run.calls] == ["early", "mid", "late"]


def test_collect_counts_retries():
    run = aet_sink.collect_run_usage([
        _retry_event(100.0), _retry_event(101.0),
        _call_event("c1", 102.0, input_tokens=10),
    ])

    assert run.retries == 2
    assert len(run.calls) == 1


def test_collect_buckets_billing_modes_separately():
    """Metered spend and subscription quota are different units, so a single total
    would be neither. TokenUsage refuses to add them; the fold must not try."""
    run = aet_sink.collect_run_usage([
        _call_event("m1", 100.0, input_tokens=10, cost_usd=0.01,
                    cost_source="billed", billing_mode="per_token"),
        _call_event("s1", 101.0, input_tokens=20, cost_usd=0.50,
                    cost_source="billed", billing_mode="subscription"),
    ])

    assert set(run.by_billing_mode) == {"per_token", "subscription"}
    assert run.metered.input_tokens == 10
    assert run.metered.cost_usd == pytest.approx(0.01)
    assert run.subscription.input_tokens == 20
    assert run.subscription.cost_usd == pytest.approx(0.50)


def test_collect_splits_per_model():
    """A run that delegated to a cheaper model must contribute to each model's
    spend, not only to its identity model."""
    run = aet_sink.collect_run_usage([
        _call_event("c1", 100.0, input_tokens=100, model="zai.glm-5"),
        _call_event("c2", 101.0, input_tokens=10, model="amazon.nova-pro-v1:0"),
        _call_event("c3", 102.0, input_tokens=10, model="amazon.nova-pro-v1:0"),
    ])

    per_model = run.per_model()
    assert set(per_model) == {"zai.glm-5", "amazon.nova-pro-v1:0"}
    assert per_model["amazon.nova-pro-v1:0"].input_tokens == 20
    # Identity model = the one that consumed the most, not the first seen.
    assert run.dominant_model == "zai.glm-5"


def test_collect_preserves_a_backend_reported_cost_source():
    """The backend knows whether the provider reported the figure; re-deriving it
    from a flat dict here would downgrade a billed cost to an estimate."""
    run = aet_sink.collect_run_usage([
        _call_event("c1", 100.0, input_tokens=10, cost_usd=0.01, cost_source="billed"),
    ])

    assert run.metered.cost_source == "billed"


def test_collect_keeps_an_unpriced_run_unpriced():
    run = aet_sink.collect_run_usage([
        _call_event("c1", 100.0, input_tokens=10,
                    model="chia-test-model-with-no-price-9e3f"),
    ])

    assert run.metered.cost_usd is None
    assert run.metered.cost_source == "unavailable"


def test_collect_empty_is_an_empty_run():
    run = aet_sink.collect_run_usage([])

    assert run.calls == ()
    assert run.metered == TokenUsage()
    assert run.trajectory() == []
    assert run.dominant_model == ""


# ---------------------------------------------------------------------------
# The trajectory series
# ---------------------------------------------------------------------------


def test_trajectory_is_cumulative_and_time_relative():
    run = aet_sink.collect_run_usage([
        _call_event("c1", 100.0, input_tokens=10, output_tokens=1, cache_read=5,
                    cost_usd=0.01, cost_source="billed"),
        _call_event("c2", 105.5, input_tokens=20, output_tokens=2, cache_read=5,
                    cost_usd=0.02, cost_source="billed"),
    ])

    points = run.trajectory()

    assert [p["index"] for p in points] == [0, 1]
    assert points[0]["t_s"] == pytest.approx(0.0)
    assert points[1]["t_s"] == pytest.approx(5.5)
    assert [p["cum_input"] for p in points] == [10, 30]
    assert [p["cum_cache"] for p in points] == [5, 10]
    assert points[1]["cum_cost"] == pytest.approx(0.03)
    assert points[1]["provisional_cost"] is False


def test_trajectory_reports_the_cache_split_not_only_the_sum():
    """A cache read is billed at 0.1x the input rate and a write at 1.25x -- 12.5x
    apart -- so two runs with the same ``cum_cache`` can differ that much in what the
    cache actually cost. The cold/warm shape below is the case: the first call writes
    the cache, the second reads it.
    """
    run = aet_sink.collect_run_usage([
        _call_event("cold", 100.0, input_tokens=10, cache_creation=4000),
        _call_event("warm", 101.0, input_tokens=10, cache_read=4000),
    ])

    points = run.trajectory()

    assert [p["cum_cache_creation"] for p in points] == [4000, 4000]
    assert [p["cum_cache_read"] for p in points] == [0, 4000]
    assert [p["cum_cache"] for p in points] == [4000, 8000]
    # the split is structure, not the sum duplicated
    assert points[1]["cum_cache_read"] != points[1]["cum_cache"]


def test_trajectory_flags_an_estimated_cost_as_provisional():
    """aet already has a provisional_cost flag for a point whose cost is not
    authoritative; an estimate is exactly that."""
    run = aet_sink.collect_run_usage([
        _call_event("c1", 100.0, input_tokens=10, cost_usd=0.01,
                    cost_source="estimated"),
    ])

    assert run.trajectory()[0]["provisional_cost"] is True


def test_trajectory_is_per_billing_mode():
    """Mixing quota and spend on one curve would draw a line through two units."""
    run = aet_sink.collect_run_usage([
        _call_event("m1", 100.0, input_tokens=10, billing_mode="per_token"),
        _call_event("s1", 101.0, input_tokens=20, billing_mode="subscription"),
    ])

    assert len(run.trajectory("per_token")) == 1
    assert len(run.trajectory("subscription")) == 1


# ---------------------------------------------------------------------------
# No-op paths (no aet needed)
# ---------------------------------------------------------------------------


def test_record_run_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("CHIA_AET_SINK", raising=False)

    assert aet_sink.record_run([_call_event("c1", 1.0, input_tokens=10)],
                               run_dir=tmp_path) is False
    assert not (tmp_path / "logs").exists()


def test_record_run_noop_without_a_run_dir(monkeypatch):
    monkeypatch.delenv("CHIA_AET_RUN_DIR", raising=False)

    assert aet_sink.record_run([_call_event("c1", 1.0, input_tokens=10)],
                               enabled=True) is False


def test_record_run_noop_when_nothing_was_recorded(tmp_path):
    assert aet_sink.record_run([_plain_event("p1", 1.0)], run_dir=tmp_path,
                               enabled=True) is False


def test_is_enabled_reads_the_env_flag(monkeypatch):
    monkeypatch.setenv("CHIA_AET_SINK", "1")
    assert aet_sink.is_enabled() is True
    monkeypatch.setenv("CHIA_AET_SINK", "0")
    assert aet_sink.is_enabled() is False
    # An explicit argument always wins.
    assert aet_sink.is_enabled(True) is True


# ---------------------------------------------------------------------------
# Writing — needs aet
# ---------------------------------------------------------------------------

aet = pytest.importorskip("aet", reason="aet not installed; install chia[aet]")


def _final_metrics(run_dir) -> dict:
    """Metric name -> value, applying aet's last-occurrence-wins rule.

    This mirrors ``aet.trajectory.rollup._read_metrics`` deliberately: the whole
    point is that the sink must be correct under *that* reader, not under a
    hypothetical accumulating one.
    """
    path = run_dir / "logs" / "metrics.jsonl"
    final: dict = {}
    for line in path.read_text().strip().splitlines():
        record = json.loads(line)
        if record.get("step") is None:  # step-metrics are the trajectory, not scalars
            final[record["name"]] = record["value"]
    return final


def _step_series(run_dir) -> dict:
    """Step-metric name -> values ordered by step. The trajectory as aet reads it back."""
    path = run_dir / "logs" / "metrics.jsonl"
    rows: dict = {}
    for line in path.read_text().strip().splitlines():
        record = json.loads(line)
        if record.get("step") is not None:
            rows.setdefault(record["name"], []).append((record["step"], record["value"]))
    return {name: [v for _, v in sorted(pairs)] for name, pairs in rows.items()}


def _step_metric_names(run_dir) -> set:
    path = run_dir / "logs" / "metrics.jsonl"
    return {
        json.loads(line)["name"]
        for line in path.read_text().strip().splitlines()
        if json.loads(line).get("step") is not None
    }


def test_multi_call_run_records_the_sum_not_the_last_call(tmp_path):
    """THE regression. aet takes the last occurrence of each metric name, so a sink
    that fired per call reported the final call's tokens as the run's total — an
    N-call run under-reported by roughly N times."""
    events = [
        _call_event("c1", 100.0, input_tokens=1000, output_tokens=100,
                    cache_read=5000, cost_usd=0.10, cost_source="billed"),
        _call_event("c2", 101.0, input_tokens=2000, output_tokens=200,
                    cache_read=6000, cost_usd=0.20, cost_source="billed"),
        _call_event("c3", 102.0, input_tokens=7, output_tokens=1,
                    cache_read=3, cost_usd=0.001, cost_source="billed"),
    ]

    assert aet_sink.record_run(events, run_dir=tmp_path, run_id="r1", suite="s",
                               enabled=True) is True

    final = _final_metrics(tmp_path)
    assert final["gen_ai.usage.input_tokens"] == 3007          # not 7
    assert final["gen_ai.usage.output_tokens"] == 301          # not 1
    assert final["gen_ai.usage.cache_read.input_tokens"] == 11_003  # not 3
    assert final["aet.agent.cost_usd"] == pytest.approx(0.301)  # not 0.001
    assert final[aet_sink.LLM_CALLS_METRIC] == 3


def test_run_record_is_written_so_the_run_has_an_identity(tmp_path):
    """Without run_record.json, aet infers run_id and suite from directory names —
    which is how two chia runs became indistinguishable."""
    assert aet_sink.record_run([_call_event("c1", 1.0, input_tokens=10)],
                               run_dir=tmp_path, run_id="run-42", suite="nl2spec",
                               project="proj", enabled=True) is True

    record = json.loads((tmp_path / "run_record.json").read_text())
    assert record["run_id"] == "run-42"
    assert record["suite"] == "nl2spec"
    assert record["project"] == "proj"
    assert record["source"] == "chia"
    assert record["llm_calls"] == 1


def test_trajectory_points_are_written(tmp_path):
    """A scalars-only run has no curve, so none of aet's plot kinds work for it.
    Emitting the step-metric family is the single change that unlocks them."""
    events = [
        _call_event("c1", 100.0, input_tokens=10, output_tokens=1, cost_usd=0.01,
                    cost_source="billed"),
        _call_event("c2", 102.0, input_tokens=20, output_tokens=2, cost_usd=0.02,
                    cost_source="billed"),
    ]

    aet_sink.record_run(events, run_dir=tmp_path, enabled=True)

    names = _step_metric_names(tmp_path)
    assert "aet.traj.cum_input_tokens" in names
    assert "aet.traj.cum_cost_usd" in names
    assert "aet.traj.t_s" in names


def test_the_cache_split_reaches_the_step_metrics(tmp_path):
    """``aet plot --split-cache`` reads these two families; without them it draws two
    lines flat at zero for a run that plainly used the cache.

    Asserted on the *values*, not on the metric names: aet's logger emits both families
    unconditionally now that its parameters default to 0.0, so a name-only assertion
    would pass against a sink that never sent the split.
    """
    events = [
        _call_event("cold", 100.0, input_tokens=10, cache_creation=4000,
                    cost_usd=0.01, cost_source="billed"),
        _call_event("warm", 102.0, input_tokens=10, cache_read=4000,
                    cost_usd=0.002, cost_source="billed"),
    ]

    aet_sink.record_run(events, run_dir=tmp_path, enabled=True)

    series = _step_series(tmp_path)
    assert series["aet.traj.cum_cache_creation_tokens"] == [4000, 4000]
    assert series["aet.traj.cum_cache_read_tokens"] == [0, 4000]
    assert series["aet.traj.cum_cache_tokens"] == [4000, 8000]


def test_a_run_is_recorded_as_one_round_with_a_real_duration(tmp_path):
    """Without a round boundary and a summary param, aet's log reader reports
    ``num_rounds=0, duration_s=0`` and the figure titles itself "0 rounds ... 0 min"
    for a run that took as long as it took."""
    events = [
        _call_event("c1", 100.0, input_tokens=10, cost_usd=0.01, cost_source="billed"),
        _call_event("c2", 130.0, input_tokens=20, cost_usd=0.02, cost_source="billed"),
    ]

    aet_sink.record_run(events, run_dir=tmp_path, run_id="r-round", enabled=True)

    events_out = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text().strip().splitlines()
    ]
    rounds = [e for e in events_out if e.get("event") == "aet.traj.round"]
    assert len(rounds) == 1
    assert rounds[0]["payload"]["t_end_s"] == pytest.approx(30.0)
    # the verdict fields stay unset: chia does not run the oracle, and n_passed=0
    # would render as a failing run rather than as an unjudged one
    assert rounds[0]["payload"]["n_passed"] is None

    params = json.loads((tmp_path / "logs" / "params.json").read_text())
    summary = params["aet.traj.summary"]
    assert summary["run_id"] == "r-round"
    assert summary["num_rounds"] == 1
    assert summary["duration_s"] == pytest.approx(30.0)


def test_the_run_dir_reconstructs_to_the_same_totals_aet_sees(tmp_path):
    """The interchange contract, asserted rather than assumed.

    The claim the sink makes is that a chia run directory *is* an aet run: whatever
    ``collect_run_usage`` folded, ``RunTrajectory.from_run_dir`` must give back. This
    goes through the logs/ reconstruction path, because a chia run has no
    ``metrics/trajectory.json`` fast path.
    """
    from aet.trajectory.model import RunTrajectory

    events = [
        _call_event("cold", 100.0, input_tokens=1000, output_tokens=100,
                    cache_creation=4000, cost_usd=0.10, cost_source="billed"),
        _call_event("warm", 110.0, input_tokens=1500, output_tokens=200,
                    cache_read=4000, cost_usd=0.02, cost_source="billed"),
    ]
    run = aet_sink.collect_run_usage(events)
    assert aet_sink.record_run(events, run_dir=tmp_path, run_id="rt", enabled=True)

    back = RunTrajectory.from_run_dir(tmp_path)
    metered = run.metered

    assert len(back.points) == len(run.calls)
    last = back.points[-1]
    assert last.cum_input_tokens == metered.input_tokens
    assert last.cum_output_tokens == metered.output_tokens
    assert last.cum_cache_read_tokens == metered.cache_read_input_tokens
    assert last.cum_cache_creation_tokens == metered.cache_creation_input_tokens
    assert last.cum_cost_usd == pytest.approx(float(metered.cost_usd))
    assert back.num_rounds == 1
    assert back.duration_s == pytest.approx(10.0)


def test_a_trajectory_still_lands_on_an_aet_without_the_cache_split(tmp_path):
    """``chia[aet]`` tracks aet's default branch, not a release, so the installed copy
    may predate ``log_trajectory_point``'s split parameters. Degrade to the sum rather
    than lose the whole curve to a TypeError."""
    recorded = []

    class _OldLogger:
        def log_trajectory_point(self, index, t_s, cum_input, cum_output, cum_cache,
                                 cum_cost, round_index=0, provisional_cost=False):
            recorded.append((index, cum_cache))

    for point in aet_sink.collect_run_usage([
        _call_event("c1", 1.0, input_tokens=10, cache_read=50),
    ]).trajectory():
        aet_sink._log_point(_OldLogger(), point)

    assert recorded == [(0, 50)]


def test_subscription_cost_is_never_written_as_money(tmp_path):
    """aet's spend rollup sums aet.agent.cost_usd into a bill. Quota consumption
    reported in dollar-equivalent terms must not land there."""
    events = [
        _call_event("s1", 100.0, input_tokens=10, cost_usd=1.23,
                    cost_source="billed", billing_mode="subscription"),
    ]

    aet_sink.record_run(events, run_dir=tmp_path, enabled=True)

    final = _final_metrics(tmp_path)
    assert "aet.agent.cost_usd" not in final
    assert final[aet_sink.SUBSCRIPTION_COST_METRIC] == pytest.approx(1.23)


def test_mixed_billing_modes_keep_metered_spend_clean(tmp_path):
    events = [
        _call_event("m1", 100.0, input_tokens=10, cost_usd=0.05,
                    cost_source="billed", billing_mode="per_token"),
        _call_event("s1", 101.0, input_tokens=10, cost_usd=9.99,
                    cost_source="billed", billing_mode="subscription"),
    ]

    aet_sink.record_run(events, run_dir=tmp_path, enabled=True)

    final = _final_metrics(tmp_path)
    assert final["aet.agent.cost_usd"] == pytest.approx(0.05)
    assert final[aet_sink.SUBSCRIPTION_COST_METRIC] == pytest.approx(9.99)


def test_an_unpriced_run_writes_no_cost_at_all(tmp_path):
    """aet counts a cost-less run in unpriced_runs and never folds it in as $0;
    writing 0.0 here would silently deflate a program-wide total."""
    events = [_call_event("c1", 100.0, input_tokens=10,
                          model="chia-test-model-with-no-price-9e3f")]

    aet_sink.record_run(events, run_dir=tmp_path, enabled=True)

    final = _final_metrics(tmp_path)
    assert "aet.agent.cost_usd" not in final
    assert final["gen_ai.usage.input_tokens"] == 10


def test_per_model_breakdown_is_written(tmp_path):
    events = [
        _call_event("c1", 100.0, input_tokens=100, cost_usd=0.10,
                    cost_source="billed", model="zai.glm-5"),
        _call_event("c2", 101.0, input_tokens=10, cost_usd=0.01,
                    cost_source="billed", model="amazon.nova-pro-v1:0"),
    ]

    aet_sink.record_run(events, run_dir=tmp_path, enabled=True)

    names = set(_final_metrics(tmp_path))
    per_model = {n for n in names if n.startswith("per_model.")}
    assert any("glm" in n for n in per_model)
    assert any("nova" in n for n in per_model)


def test_retries_are_recorded(tmp_path):
    events = [_retry_event(99.0), _call_event("c1", 100.0, input_tokens=10)]

    aet_sink.record_run(events, run_dir=tmp_path, enabled=True)

    assert _final_metrics(tmp_path)[aet_sink.RETRIES_METRIC] == 1


def test_record_run_never_raises_on_a_broken_run_dir(tmp_path):
    """Telemetry must not fail an experiment that already completed."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")

    assert aet_sink.record_run([_call_event("c1", 1.0, input_tokens=10)],
                               run_dir=blocker, enabled=True) is False


# ---------------------------------------------------------------------------
# Reading the collector — needs a local Ray instance
# ---------------------------------------------------------------------------


@pytest.fixture
def local_collector():
    """A live ProfileCollectorActor on a local Ray instance, wired as the override.

    This exercises the part that replaced per-worker writes: the sink reads the
    events the collector already holds, on the driver, so there is no concurrent
    read-modify-write of one run directory from several processes.
    """
    ray = pytest.importorskip("ray")
    from chia.trace import profiler as profiler_mod

    started_here = False
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, namespace="chia", num_cpus=1,
                     include_dashboard=False)
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"could not start a local Ray instance: {exc}")
        started_here = True

    actor = ray.remote(profiler_mod.ProfileCollectorActor).options(
        name="chia_aet_sink_test_collector", num_cpus=0,
    ).remote()
    ray.get(actor.get_events.remote())
    previous = profiler_mod._collector_override
    profiler_mod._collector_override = actor
    try:
        yield actor
    finally:
        profiler_mod._collector_override = previous
        try:
            ray.kill(actor)
        except Exception:
            pass
        if started_here:
            ray.shutdown()


def test_record_run_reads_the_collector_when_given_no_events(tmp_path, local_collector):
    """The default path: the driver folds whatever the collector holds."""
    import ray

    for event in (_call_event("c1", 100.0, input_tokens=1000, cost_usd=0.10,
                              cost_source="billed"),
                  _call_event("c2", 101.0, input_tokens=2000, cost_usd=0.20,
                              cost_source="billed")):
        ray.get(local_collector.record.remote(event))

    assert aet_sink.record_run(run_dir=tmp_path, run_id="from-collector",
                               enabled=True) is True

    final = _final_metrics(tmp_path)
    assert final["gen_ai.usage.input_tokens"] == 3000
    assert final["aet.agent.cost_usd"] == pytest.approx(0.30)
