"""Outcome analysis checks the paired unit and provider-cost provenance."""

from examples.harness_study.delegation_benchmark.analyze import analyze


def _row(task, repeat, harness, *, grade, delegate, cost, wall, full=True):
    return {"arm": {"task": task, "repeat": repeat, "harness": harness},
            "grade_pass": grade, "weak_model_requests": delegate,
            "cost_usd": cost, "wall_seconds": wall, "full_usage": full}


def test_success_requires_delegation_and_cost_uses_both_successful_pairs():
    schedule = {"arms": [{"task": "a", "repeat": repeat, "harness": harness}
                         for repeat in (0, 1) for harness in ("claude", "opencode")]}
    rows = [
        _row("a", 0, "claude", grade=True, delegate=1, cost=0.30, wall=10),
        _row("a", 0, "opencode", grade=True, delegate=1, cost=0.20, wall=8),
        _row("a", 1, "claude", grade=True, delegate=0, cost=0.10, wall=7),
        _row("a", 1, "opencode", grade=False, delegate=1, cost=0.40, wall=12),
    ]
    result = analyze(rows, schedule)
    assert result["complete_pairs"] == 2
    assert result["delegated_success"]["claude"] == 1
    assert result["delegated_success"]["opencode"] == 1
    assert result["both_succeeded_fully_priced_pairs"] == 1
    assert abs(result["provider_cost_usd"][
        "median_paired_difference_claude_minus_opencode"] - 0.10) < 1e-9
    assert result["cli_wall_seconds"][
        "median_paired_difference_claude_minus_opencode"] == 2
