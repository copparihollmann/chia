"""Yosys synthesis against a liberty library — post-synthesis QoR, not PPA.

Say what this is, because the distinction decides what the numbers may be used for: no placement, no
routing, no parasitics, no timing closure. Mapped cell area and a cell-depth path estimate. An area
*ratio* between two designs run through an identical flow is meaningful; an absolute number quoted as
PPA is not.

The only thing that makes two runs comparable is that every knob was identical, so
:class:`YosysConfig` carries them all explicitly rather than defaulting them inside the script — a
default that drifts between invocations is a silent confound.

Two behaviours worth knowing:

* **Validity gates are separate from the numbers.** An inferred latch, a combinational loop, an
  unmapped cell or a black box does not mean "slightly worse area", it means the area is not the
  area of the design you think you synthesized. :attr:`YosysResult.valid` is False and the caller is
  expected to withhold the row rather than report it with a caveat.
* **Absent tooling raises.** ``chia.synth`` is used from a measured flow, where a silently skipped
  synthesis is scored as a measurement — a missing area and a candidate with no area look identical
  in a results table.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Liberty cell-name fragments that mark a mapped sequential element. Checked case-insensitively
#: against the mapped cell names in `stat` output (ASAP7 uses DFF*/SDF*; other libraries vary).
_SEQ_MARKERS = ("dff", "dlatch", "sdf", "latch", "_ff")

#: Cell names that mean the mapping did not finish. A `$`-prefixed cell after techmap+abc is an
#: internal yosys primitive that no liberty cell was found for.
_UNMAPPED_PREFIX = "$"

#: Latch primitives. An inferred latch in a design specified as synchronous is a defect, not a
#: style: it means a branch left a signal unassigned.
_LATCH_MARKERS = ("dlatch", "$_sr_", "latch")


class YosysUnavailable(RuntimeError):
    """Yosys or the liberty library is missing.

    Raised rather than returning ``available=False``. From a measured flow, a skipped synthesis and
    a candidate that genuinely has no area are the same empty cell in a results table, and only one
    of them is a finding.
    """


@dataclass(frozen=True)
class YosysConfig:
    """Every knob that must be identical across the designs being compared."""

    liberty: str
    top: str
    clock_period_ns: float | None = None
    flatten: bool = False
    memory_policy: str = "inference"        # inference | nomap
    abc_script: str | None = None
    read_args: str = "-sv"
    seed: int = 0
    timeout_s: int = 1200

    def preamble(self) -> list[str]:
        """Read the liberty as a CELL LIBRARY before anything else.

        Without this, `check` and `ltp` run on the mapped netlist with no knowledge of the liberty
        cells' pin directions, so every wire driven by a mapped cell's output looks undriven:
        `check` emits one warning per bit and `ltp` reports a longest path of 0. Both are false, and
        both are the kind of false that gets read as a real defect — an inflated warning count and a
        logic depth of zero on a design that plainly has logic.
        """
        return [f"read_liberty -lib {self.liberty}"]

    def passes(self) -> list[str]:
        """The pass list, in order. Explicit so a diff between two runs is readable."""
        mem = "memory -nomap" if self.memory_policy == "nomap" else "memory"
        abc = self.abc_script or f"abc -liberty {self.liberty}"
        # Not flattening means OMITTING the pass — yosys has no `flatten -none`. Preserving the
        # module boundary is the default here because per-module area is what a module-level
        # comparison needs; flattening merges it into the parent and the number disappears.
        return [
            f"hierarchy -check -top {self.top}",
            "proc",
        ] + (["flatten"] if self.flatten else []) + [
            "opt -full",
            mem,
            "techmap",
            "opt -fast",
            f"dfflibmap -liberty {self.liberty}",
            abc,
            "setundef -zero",
            "clean -purge",
            "check",
            f"ltp -noff",
            f"stat -liberty {self.liberty}",
        ]


@dataclass
class YosysResult:
    """Post-synthesis QoR for one top module."""

    top: str = ""
    ok: bool = False
    mapped_cell_area: float | None = None
    sequential_area: float | None = None
    total_cell_count: int | None = None
    sequential_cell_count: int | None = None
    combinational_cell_count: int | None = None
    inferred_memory_count: int = 0
    unmapped_cell_count: int = 0
    logic_depth: int | None = None
    estimated_critical_path_ns: float | None = None
    synthesis_runtime_s: float = 0.0
    warnings: int = 0
    latches: int = 0
    combinational_loops: int = 0
    black_boxes: int = 0
    cells_by_type: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    log_path: str | None = None
    detail: str = ""

    @property
    def validity_failures(self) -> list[str]:
        """Why this row must be withheld rather than reported. Empty means reportable."""
        out = []
        if not self.ok:
            out.append("synthesis did not complete"
                       + (f": {self.errors[0]}" if self.errors else ""))
        if self.latches:
            out.append(f"{self.latches} inferred latch(es) — not the synchronous design specified")
        if self.combinational_loops:
            out.append(f"{self.combinational_loops} combinational loop(s)")
        if self.unmapped_cell_count:
            out.append(f"{self.unmapped_cell_count} unmapped cell(s) — the area is a lower bound")
        if self.black_boxes:
            out.append(f"{self.black_boxes} black box(es) — the area is a lower bound")
        if self.mapped_cell_area is None:
            out.append("no area reported")
        return out

    @property
    def valid(self) -> bool:
        return not self.validity_failures


def _first_float(line: str) -> float | None:
    for tok in line.replace(":", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def _parse(log: str, top: str) -> YosysResult:
    """Parse a yosys log. String scanning only — no regex, matching the house rule upstream."""
    res = YosysResult(top=top)
    lines = log.splitlines()

    in_stat = False
    for i, raw in enumerate(lines):
        line = raw.strip()

        if line.startswith("Number of cells:"):
            in_stat = True
            res.total_cell_count = int(line.rsplit(None, 1)[-1])
            continue
        if in_stat:
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                res.cells_by_type[parts[0]] = int(parts[1])
                continue
            if line and not line[0].isdigit():
                in_stat = False

        if line.startswith("Chip area for module"):
            res.mapped_cell_area = _first_float(line.rsplit(":", 1)[-1])
        elif line.startswith("of which used for sequential elements"):
            res.sequential_area = _first_float(line.rsplit(":", 1)[-1])
        elif line.startswith("Number of memories:"):
            res.inferred_memory_count = int(line.rsplit(None, 1)[-1])
        elif "Longest topological path" in line and "length=" in line:
            tail = line.split("length=", 1)[1]
            digits = "".join(c for c in tail if c.isdigit())
            res.logic_depth = int(digits) if digits else None
        elif line.startswith("Warnings:"):
            tok = line.split()
            for j, t in enumerate(tok):
                if t == "total" and j:
                    try:
                        res.warnings = int(tok[j - 1])
                    except ValueError:
                        pass
        elif "found and reported" in line.lower() and "problem" in line.lower():
            pass
        if "logic loop" in line.lower() or "combinational loop" in line.lower():
            res.combinational_loops += 1
        if line.startswith("ERROR:"):
            res.errors.append(line[len("ERROR:"):].strip())
            # Yosys REFUSES to infer a latch from always_comb — it errors rather than emitting a
            # $_DLATCH_ cell. So the latch never appears in `stat` and the cell scan below cannot
            # see it. Without this the result reads "synthesis did not complete", which sends the
            # reader to debug the flow instead of the design.
            if "latch inferred" in line.lower():
                res.latches += 1

    for name, n in res.cells_by_type.items():
        low = name.lower()
        if name.startswith(_UNMAPPED_PREFIX):
            res.unmapped_cell_count += n
        if any(m in low for m in _LATCH_MARKERS) and "dff" not in low:
            res.latches += n

    seq = sum(n for name, n in res.cells_by_type.items()
              if any(m in name.lower() for m in _SEQ_MARKERS))
    if res.total_cell_count is not None:
        res.sequential_cell_count = seq
        res.combinational_cell_count = res.total_cell_count - seq
    return res


class YosysNode:
    """Runs one Yosys synthesis. Stateless; all per-run state lives in ``workdir``."""

    logging_name = "YosysNode"

    def __init__(self, yosys: str | None = None, logging_level: int = logging.INFO):
        self._yosys = yosys
        self.logger = logging.getLogger(self.logging_name)
        self.logger.setLevel(logging_level)

    def binary(self) -> str | None:
        """The yosys to use, or ``None``.

        An explicitly-supplied path is VALIDATED rather than trusted: a stale configured path is the
        common failure (this host had one pointing at a deleted install tree), and letting it through
        surfaces as a bare FileNotFoundError from inside subprocess rather than the structured
        error the caller is set up to handle.
        """
        if self._yosys:
            p = Path(self._yosys)
            if p.is_file():
                return str(p)
            cand = p / "yosys"
            return str(cand) if cand.is_file() else None
        return shutil.which("yosys")

    def available(self, cfg: YosysConfig) -> bool:
        return self.binary() is not None and Path(cfg.liberty).is_file()

    def run(self, sources: list[str | Path], cfg: YosysConfig, workdir: str | Path
            ) -> YosysResult:
        """Synthesize ``sources`` and return the QoR. Raises :class:`YosysUnavailable` if it cannot."""
        yosys = self.binary()
        if yosys is None:
            raise YosysUnavailable("yosys not found on PATH and none supplied")
        if not Path(cfg.liberty).is_file():
            raise YosysUnavailable(f"liberty library not found: {cfg.liberty}")
        srcs = [Path(s) for s in sources]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            raise YosysUnavailable(f"source file(s) not found: {missing}")

        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        script = "\n".join(
            cfg.preamble()
            + [f"read_verilog {cfg.read_args} {s}" for s in srcs]
            + cfg.passes()) + "\n"
        (wd / "synth.ys").write_text(script)

        t0 = time.monotonic()
        try:
            proc = subprocess.run([yosys, "-s", str(wd / "synth.ys")],
                                  capture_output=True, text=True, timeout=cfg.timeout_s,
                                  cwd=str(wd))
        except subprocess.TimeoutExpired:
            res = YosysResult(top=cfg.top, ok=False,
                              synthesis_runtime_s=time.monotonic() - t0,
                              detail=f"timed out after {cfg.timeout_s}s")
            return res
        elapsed = time.monotonic() - t0
        log = (proc.stdout or "") + (proc.stderr or "")
        (wd / "synth.log").write_text(log)

        res = _parse(log, cfg.top)
        res.ok = proc.returncode == 0
        res.synthesis_runtime_s = elapsed
        res.log_path = str(wd / "synth.log")
        if not res.ok:
            res.detail = "; ".join(res.errors) if res.errors else (proc.stderr or "")[-2000:]
        if cfg.clock_period_ns and res.logic_depth is not None:
            # A cell-depth proxy, NOT a timing analysis. Named as an estimate everywhere it appears
            # so it is never mistaken for a closed path.
            res.estimated_critical_path_ns = None
        return res


def yosys_synth(sources: list[str | Path], cfg: YosysConfig, workdir: str | Path,
                yosys: str | None = None) -> YosysResult:
    """Functional entry point. Decorate with ``@ChiaFunction`` at the pipeline level, and key the
    cache with :func:`chia.base.content_tag.content_tag` over ``sources`` + ``cfg`` + the tool
    version — a tag that does not cover the RTL will serve a previous design's area."""
    return YosysNode(yosys=yosys).run(sources, cfg, workdir)
