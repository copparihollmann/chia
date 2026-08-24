"""Verilator on plain SystemVerilog: three checkpoints, and what each one actually catches.

The distinction is the point. A generation-loop trajectory measures time-to-first-parse and
time-to-first-elaboration separately, and a run can sit at one for a long time — collapsing them
would make "the RTL does not parse" and "the testbench failed" the same result.

Run:
  pytest chia/sim/test/test_verilator_sv_node.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chia.sim.verilator_sv_node import (
    ALWAYS_FATAL,
    DEFAULT_SUPPRESS,
    VerilatorSvNode,
    VerilatorSvResult,
    VerilatorUnavailable,
    _classify,
)

_VERILATOR = (shutil.which("verilator")
              or "/scratch2/agustin/mvp-lhwir/modeling/third_party/verilator/bin/verilator")
_HAVE = Path(_VERILATOR).is_file()

GOOD = """\
module good (input logic clk, input logic rst, input logic [7:0] a, input logic [7:0] b,
             output logic [7:0] s);
  always_ff @(posedge clk) begin
    if (rst) s <= '0; else s <= a + b;
  end
endmodule
"""

UNDECLARED = """\
module broken (input logic clk, output logic [7:0] q);
  always_ff @(posedge clk) q <= undeclared_signal;
endmodule
"""

MISSING_CHILD = """\
module parent (input logic clk, output logic [7:0] o);
  child u (.clk(clk), .o(o));
endmodule
"""


def _node() -> VerilatorSvNode:
    return VerilatorSvNode(verilator=_VERILATOR)


class TestLogClassification:
    def test_errors_and_warnings_are_separated(self):
        errs, warns = _classify("%Error: bad thing\n%Warning-FOO: cosmetic\nplain line\n")
        assert errs == ["%Error: bad thing"]
        assert warns == ["%Warning-FOO: cosmetic"]

    def test_continuation_lines_join_their_message(self):
        """A bare '%Error: Cannot find file' without its next line does not say WHICH file."""
        errs, _ = _classify("%Error: Cannot find file\n    tried: /a/b.sv\n")
        assert errs == ["%Error: Cannot find file | tried: /a/b.sv"]

    def test_an_unindented_line_ends_the_message(self):
        errs, _ = _classify("%Error: one\nunrelated output\n    not a continuation\n")
        assert errs == ["%Error: one"]

    def test_an_empty_log_is_not_an_error(self):
        assert _classify("") == ([], [])


class TestSuppressionPolicy:
    def test_the_always_fatal_set_and_the_default_suppressions_do_not_overlap(self):
        """A warning cannot be both ignored by default and always fatal — one of the two would
        silently win, and which one would depend on argument order."""
        assert not (set(DEFAULT_SUPPRESS) & set(ALWAYS_FATAL))

    def test_behaviour_changing_warnings_are_never_suppressed(self):
        for w in ("LATCH", "BLKANDNBLK", "MULTIDRIVEN", "IMPLICIT"):
            assert w in ALWAYS_FATAL


class TestUnavailable:
    def test_a_bad_explicit_path_raises_the_structured_error(self, tmp_path):
        src = tmp_path / "x.sv"
        src.write_text(GOOD)
        with pytest.raises(VerilatorUnavailable):
            VerilatorSvNode(verilator="/definitely/not/verilator").lint(
                [src], "good", tmp_path / "wd")

    def test_run_sim_without_a_build_reports_rather_than_raises(self, tmp_path):
        """Nothing was misconfigured — the caller just skipped a step, and that is a result."""
        r = _node().run_sim(tmp_path, "good")
        assert not r.ok and "build first" in r.first_error()


@pytest.mark.skipif(not _HAVE, reason="verilator not available")
class TestEndToEnd:
    def test_a_clean_module_lints(self, tmp_path):
        src = tmp_path / "good.sv"
        src.write_text(GOOD)
        r = _node().lint([src], "good", tmp_path / "wd")
        assert r.ok and r.n_errors == 0
        assert r.stage == "lint" and r.runtime_s >= 0

    def test_an_undeclared_signal_fails_lint(self, tmp_path):
        src = tmp_path / "broken.sv"
        src.write_text(UNDECLARED)
        r = _node().lint([src], "broken", tmp_path / "wd")
        assert not r.ok and r.n_errors > 0

    def test_lint_already_catches_a_missing_submodule(self, tmp_path):
        """Pinned because the docstring used to claim the opposite. Lint resolves instantiations,
        so MODMISSING is NOT what distinguishes elaborate from lint."""
        src = tmp_path / "p.sv"
        src.write_text(MISSING_CHILD)
        r = _node().lint([src], "parent", tmp_path / "wd")
        assert not r.ok
        assert any("MODMISSING" in e for e in r.errors)

    def test_a_clean_module_elaborates(self, tmp_path):
        src = tmp_path / "good.sv"
        src.write_text(GOOD)
        r = _node().elaborate([src], "good", tmp_path / "wd")
        assert r.ok and r.n_errors == 0
        assert (tmp_path / "wd" / "obj_dir").is_dir(), "elaboration emitted a model"

    def test_elaboration_of_a_missing_child_fails(self, tmp_path):
        src = tmp_path / "p.sv"
        src.write_text(MISSING_CHILD)
        assert not _node().elaborate([src], "parent", tmp_path / "wd").ok

    def test_every_stage_writes_its_log(self, tmp_path):
        src = tmp_path / "good.sv"
        src.write_text(GOOD)
        n = _node()
        for stage, r in (("lint", n.lint([src], "good", tmp_path / "wd")),
                         ("elaborate", n.elaborate([src], "good", tmp_path / "wd"))):
            assert r.log_path and Path(r.log_path).is_file()
            assert Path(r.log_path).name == f"{stage}.log"

    def test_the_version_is_reported_for_the_run_manifest(self):
        """A toolchain version that is not recorded cannot be held constant across arms."""
        v = _node().version()
        assert v and "Verilator" in v
