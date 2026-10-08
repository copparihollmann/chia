"""Yosys synthesis node: parsing, validity gates, and the two false positives it had to fix.

Run:
  pytest chia/synth/test/test_yosys_node.py

Skips cleanly without yosys or a liberty library; the parsing tests run everywhere because they
operate on captured log text.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chia.synth.yosys_node import (
    YosysConfig,
    YosysNode,
    YosysResult,
    YosysUnavailable,
    _parse,
)

_LIBERTY = "/scratch/agustin/projects/asap7-energy/synth/libs/asap7_merged_RVT_TT.lib"
_HAVE = shutil.which("yosys") is not None or Path(
    "/scratch2/agustin/miniforge3/envs/autogenila/bin/yosys").is_file()
_HAVE_LIB = Path(_LIBERTY).is_file()

GOOD = """\
module good (input logic clk, input logic rst, input logic [7:0] a, input logic [7:0] b,
             output logic [7:0] s);
  always_ff @(posedge clk) begin
    if (rst) s <= '0; else s <= a + b;
  end
endmodule
"""

LATCHY = """\
module latchy (input logic en, input logic [7:0] d, output logic [7:0] q);
  always_comb begin
    if (en) q = d;
  end
endmodule
"""


def _node() -> YosysNode:
    y = shutil.which("yosys") or "/scratch2/agustin/miniforge3/envs/autogenila/bin/yosys"
    return YosysNode(yosys=y)


class TestConfig:
    def test_not_flattening_omits_the_pass(self):
        """`flatten -none` is not a yosys option — it is a syntax error that aborts the script."""
        assert "flatten" not in YosysConfig(liberty="l", top="t").passes()
        assert "flatten" in YosysConfig(liberty="l", top="t", flatten=True).passes()

    def test_the_liberty_is_read_as_a_cell_library_first(self):
        """Without this, `check` and `ltp` do not know the mapped cells' pin directions: every wire
        driven by a cell output looks undriven, so the warning count is inflated by one per bit and
        the reported logic depth is 0 on a design that plainly has logic. Both false, both readable
        as real defects."""
        pre = YosysConfig(liberty="/x/lib.lib", top="t").preamble()
        assert pre == ["read_liberty -lib /x/lib.lib"]

    def test_the_pass_list_is_explicit_so_two_runs_can_be_diffed(self):
        p = YosysConfig(liberty="l", top="t").passes()
        assert p[0] == "hierarchy -check -top t"
        assert any(x.startswith("dfflibmap") for x in p)
        assert any(x.startswith("stat") for x in p)

    def test_memory_policy_is_a_knob_not_a_default(self):
        assert "memory" in YosysConfig(liberty="l", top="t").passes()
        assert "memory -nomap" in YosysConfig(liberty="l", top="t", memory_policy="nomap").passes()


class TestParsing:
    def test_area_and_cell_counts(self):
        log = """\
   Number of cells:                 53
     DFFHQNx1_ASAP7_75t_R            8
     INVx1_ASAP7_75t_R              12
     NAND2xp33_ASAP7_75t_R          33

   Chip area for module '\\good': 6.065280
     of which used for sequential elements: 2.332800 (38.46%)
"""
        r = _parse(log, "good")
        assert r.mapped_cell_area == pytest.approx(6.06528)
        assert r.total_cell_count == 53
        assert r.sequential_cell_count == 8
        assert r.combinational_cell_count == 45

    def test_logic_depth_from_ltp(self):
        r = _parse("Longest topological path in good (length=8):\n", "good")
        assert r.logic_depth == 8

    def test_a_refused_latch_is_named_not_reported_as_a_flow_failure(self):
        """Yosys ERRORS on a latch inferred from always_comb rather than emitting a $_DLATCH_ cell,
        so the latch never reaches `stat`. Without parsing the error the result reads 'synthesis did
        not complete', which sends the reader to debug the flow instead of the design."""
        log = "ERROR: Latch inferred for signal `\\latchy.\\q' from always_comb process\n"
        r = _parse(log, "latchy")
        assert r.latches == 1
        assert r.errors and "Latch inferred" in r.errors[0]

    def test_unmapped_cells_are_counted(self):
        log = "   Number of cells:                 4\n     $_AND_                          4\n"
        r = _parse(log, "x")
        assert r.unmapped_cell_count == 4

    def test_warnings_are_taken_from_the_total(self):
        assert _parse("Warnings: 3 unique messages, 9 total\n", "x").warnings == 9


class TestValidityGates:
    def test_a_clean_result_is_reportable(self):
        r = YosysResult(top="t", ok=True, mapped_cell_area=1.0)
        assert r.valid and r.validity_failures == []

    @pytest.mark.parametrize("kw,fragment", [
        ({"latches": 1}, "latch"),
        ({"combinational_loops": 1}, "loop"),
        ({"unmapped_cell_count": 2}, "unmapped"),
        ({"black_boxes": 1}, "black box"),
    ])
    def test_each_gate_withholds_the_row(self, kw, fragment):
        """None of these mean 'slightly worse area' — they mean the area is not the area of the
        design you think you synthesized."""
        r = YosysResult(top="t", ok=True, mapped_cell_area=1.0, **kw)
        assert not r.valid
        assert any(fragment in f for f in r.validity_failures)

    def test_a_missing_area_is_a_failure_not_a_zero(self):
        r = YosysResult(top="t", ok=True, mapped_cell_area=None)
        assert not r.valid


class TestUnavailable:
    def test_a_missing_liberty_raises(self, tmp_path):
        src = tmp_path / "x.sv"
        src.write_text(GOOD)
        with pytest.raises(YosysUnavailable):
            _node().run([src], YosysConfig(liberty=str(tmp_path / "nope.lib"), top="good"),
                        tmp_path / "wd")

    def test_a_missing_source_raises(self, tmp_path):
        with pytest.raises(YosysUnavailable):
            _node().run([tmp_path / "nope.sv"], YosysConfig(liberty=_LIBERTY, top="good"),
                        tmp_path / "wd")

    def test_no_yosys_raises_rather_than_reporting_unavailable(self, tmp_path):
        """A skipped synthesis and a candidate with no area are the same empty cell downstream."""
        src = tmp_path / "x.sv"
        src.write_text(GOOD)
        with pytest.raises(YosysUnavailable):
            YosysNode(yosys="/definitely/not/yosys").run(
                [src], YosysConfig(liberty=_LIBERTY, top="good"), tmp_path / "wd")


@pytest.mark.skipif(not (_HAVE and _HAVE_LIB), reason="yosys or ASAP7 liberty unavailable")
class TestEndToEnd:
    def test_a_synchronous_design_synthesizes_cleanly(self, tmp_path):
        src = tmp_path / "good.sv"
        src.write_text(GOOD)
        r = _node().run([src], YosysConfig(liberty=_LIBERTY, top="good"), tmp_path / "wd")
        assert r.ok and r.valid, r.validity_failures
        assert r.mapped_cell_area and r.mapped_cell_area > 0
        assert r.sequential_cell_count == 8, "one flop per output bit"
        assert r.combinational_cell_count and r.combinational_cell_count > 0

    def test_the_liberty_preamble_removes_the_false_warnings_and_zero_depth(self, tmp_path):
        """Regression on the two false positives: without `read_liberty -lib` this design reported
        8 undriven-wire warnings and a logic depth of 0, both wrong."""
        src = tmp_path / "good.sv"
        src.write_text(GOOD)
        r = _node().run([src], YosysConfig(liberty=_LIBERTY, top="good"), tmp_path / "wd")
        assert r.warnings == 0
        assert r.logic_depth and r.logic_depth > 0

    def test_an_inferred_latch_is_caught_and_named(self, tmp_path):
        src = tmp_path / "latchy.sv"
        src.write_text(LATCHY)
        r = _node().run([src], YosysConfig(liberty=_LIBERTY, top="latchy"), tmp_path / "wd")
        assert not r.valid
        assert r.latches == 1
        assert any("latch" in f.lower() for f in r.validity_failures)

    def test_the_same_design_synthesizes_to_the_same_area(self, tmp_path):
        """An area comparison between designs is meaningless if one design's own area moves."""
        src = tmp_path / "good.sv"
        src.write_text(GOOD)
        cfg = YosysConfig(liberty=_LIBERTY, top="good")
        a = _node().run([src], cfg, tmp_path / "a")
        b = _node().run([src], cfg, tmp_path / "b")
        assert a.mapped_cell_area == b.mapped_cell_area
        assert a.total_cell_count == b.total_cell_count
