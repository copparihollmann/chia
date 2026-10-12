from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, filename: str):
    path = _ROOT / "examples" / "circt_issue_solver" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


benchmark = _load("isolation_benchmark", "isolation_benchmark.py")
analysis = _load("isolation_benchmark_analysis", "isolation_benchmark_analysis.py")


def test_frozen_issue_10568_input_matches_snapshot_metadata() -> None:
    benchmark_dir = _ROOT / "examples" / "circt_issue_solver" / "benchmarks"
    script = (benchmark_dir / "issue-10568-repro.sh").read_text()
    mlir = script.split("<<'MLIR'\n", 1)[1].split("\nMLIR\n", 1)[0] + "\n"
    metadata = json.loads(
        (benchmark_dir / "issue-10568-snapshot.json").read_text())
    assert hashlib.sha256(mlir.encode()).hexdigest() == metadata[
        "normalized_mlir_sha256"]
    assert metadata["source_body_sha256"] == (
        "c55f618f9f90a694c63bfaa35fbb53456083d9a88cd9178e965b7825906b763e")


def test_snapshot_prep_requires_dedicated_marked_directory(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    (rootfs / "workspace").mkdir(parents=True)
    (rootfs / "workspace" / "source").write_text("unchanged")
    (rootfs / "home/ray").mkdir(parents=True)
    (rootfs / "home/ray" / "settings").write_text("baseline")
    writable = tmp_path / "snapshots"

    first = benchmark._prepare_writable_snapshot(rootfs, writable)
    assert first["prep_seconds"] >= 0
    assert (writable / ".chia-isolation-benchmark-snapshots").is_file()
    (writable / "workspace" / "source").write_text("mutated")
    benchmark._prepare_writable_snapshot(rootfs, writable)
    assert (writable / "workspace" / "source").read_text() == "unchanged"
    assert (rootfs / "workspace" / "source").read_text() == "unchanged"


def test_snapshot_prep_refuses_unmarked_or_dangerous_roots(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    (rootfs / "workspace").mkdir(parents=True)
    (rootfs / "home/ray").mkdir(parents=True)
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    (unmarked / "keep").write_text("do not delete")

    with pytest.raises(ValueError, match="unmarked"):
        benchmark._prepare_writable_snapshot(rootfs, unmarked)
    assert (unmarked / "keep").is_file()
    with pytest.raises(ValueError, match="unsafe"):
        benchmark._prepare_writable_snapshot(rootfs, Path("/"))
    with pytest.raises(ValueError, match="unsafe"):
        benchmark._prepare_writable_snapshot(rootfs, rootfs / "nested")


def _record(pair: int, arm: str, representative: float, up: float = 1.0) -> dict:
    # Representative sum = up + dispatch + rebuild + down.
    remainder = representative - up - 2.0
    return {
        "pair": pair,
        "arm": arm,
        "up_seconds": up,
        "dispatch_seconds": 1.0,
        "down_seconds": 1.0,
        "up_returncode": 0,
        "down_returncode": 0,
        "dispatch_valid": True,
        "circt": {
            "warm_seconds": 0.0,
            "rebuild_seconds": remainder,
            "warm": {"success": True},
            "rebuild": {"success": True},
            "lit": {"success": True},
            "repro": {"returncode": 1},
        },
    }


def test_analysis_emits_go_when_threshold_and_ci_pass() -> None:
    records = [{
        "event": "manifest", "full_lit": True,
        "repro_script_sha256": "abc",
        "expected_repro_exit_code": 1,
    }]
    for pair in range(10):
        records.extend([
            _record(pair, "docker", 100 + pair / 10, up=10),
            _record(pair, "bwrap", 90 + pair / 10, up=8),
        ])
    summary = analysis.analyze(records, bootstrap_samples=500, seed=7)
    assert summary["decision"] == "go"
    assert summary["complete_pairs"] == 10
    assert summary["representative_end_to_end"]["improvement_percent"] > 5
    assert summary["representative_end_to_end"][
        "paired_bootstrap_95_ci_seconds"][0] > 0


def test_analysis_emits_no_go_for_insufficient_or_small_gain() -> None:
    records = []
    for pair in range(3):
        records.extend([
            _record(pair, "docker", 100),
            _record(pair, "bwrap", 98),
        ])
    summary = analysis.analyze(records, bootstrap_samples=100, seed=7)
    assert summary["decision"] == "no-go"
    assert not summary["criteria"]["enough_pairs"]
    assert not summary["criteria"]["threshold_met"]
