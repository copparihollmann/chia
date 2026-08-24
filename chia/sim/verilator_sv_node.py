"""Verilator on a plain SystemVerilog tree — lint, elaborate, build, run.

Three checkpoints, deliberately separate, because they fail for different reasons and a caller
usually wants the cheapest one that can answer its question:

    lint       does it parse and pass Verilator's static checks?   ~ms, no C++ toolchain
    elaborate  can Verilator emit a model of it?                   ~s
    build+run  does it simulate?                                   ~minutes, needs a testbench

Collapsing them would make "the RTL does not parse" and "the testbench failed" the same result,
which is exactly the distinction a generation-loop trajectory is built to measure: time-to-first-
parse and time-to-first-elaboration are different milestones and a run can sit at one for a long
time.

``--lint-only`` is not merely a fast build: it accepts designs a build would reject (unconnected
pins, missing timing constructs) and rejects things a build tolerates. Two checkpoints, not one
checkpoint at two speeds.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Verilator warnings that are style, not defects, and that a generated tree trips constantly.
#: Suppressed by default so the signal is not buried; every suppression is listed so a caller can
#: see exactly what was ignored rather than trusting a silent default.
DEFAULT_SUPPRESS = (
    "DECLFILENAME",   # module name != file name — a candidate may pack several modules per file
    "UNUSEDSIGNAL",
    "UNUSEDPARAM",
    "PINCONNECTEMPTY",
    "WIDTHTRUNC",     # kept out of lint; a real width bug shows up in simulation
    "WIDTHEXPAND",
)

#: Warnings that are NEVER suppressed regardless of caller preference: each is a genuine defect that
#: a generated design plausibly makes, and each changes behaviour rather than appearance.
ALWAYS_FATAL = (
    "LATCH",          # inferred latch in a design specified as synchronous
    "BLKANDNBLK",     # blocking and non-blocking assignment to one variable
    "COMBDLY",
    "MULTIDRIVEN",
    "IMPLICIT",       # implicitly declared wire — usually a typo'd signal name
    "SELRANGE",
    "CASEINCOMPLETE",
)


class VerilatorUnavailable(RuntimeError):
    """Verilator is not installed. Raised rather than reported, for the same reason
    :class:`chia.synth.yosys_node.YosysUnavailable` is: a skipped check and a passed check are the
    same empty cell downstream."""


@dataclass
class VerilatorSvResult:
    """One Verilator invocation."""

    stage: str = ""                 # lint | elaborate | build | run
    ok: bool = False
    returncode: int | None = None
    runtime_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    top: str = ""
    log_path: str | None = None
    stdout_tail: str = ""

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    def first_error(self) -> str | None:
        return self.errors[0] if self.errors else None


def _classify(log: str) -> tuple[list[str], list[str]]:
    """Split a Verilator log into errors and warnings.

    Verilator prefixes with ``%Error`` / ``%Warning``; continuation lines are indented and belong to
    the preceding message. Joining them matters — a bare ``%Error: Cannot find file`` without its
    following line does not say which file.
    """
    errors: list[str] = []
    warnings: list[str] = []
    current: list[str] | None = None
    for raw in log.splitlines():
        if raw.startswith("%Error"):
            errors.append(raw.strip())
            current = errors
        elif raw.startswith("%Warning"):
            warnings.append(raw.strip())
            current = warnings
        elif current is not None and raw.startswith((" ", "\t")) and raw.strip():
            current[-1] = current[-1] + " | " + raw.strip()
        else:
            current = None
    return errors, warnings


class VerilatorSvNode:
    """Runs Verilator against a standalone SystemVerilog tree. Stateless."""

    logging_name = "VerilatorSvNode"

    def __init__(self, verilator: str | None = None, logging_level: int = logging.INFO):
        self._verilator = verilator
        self.logger = logging.getLogger(self.logging_name)
        self.logger.setLevel(logging_level)

    def binary(self) -> str | None:
        """The verilator to use, or ``None``. An explicit path is validated, not trusted — see
        :meth:`chia.synth.yosys_node.YosysNode.binary`."""
        if self._verilator:
            p = Path(self._verilator)
            if p.is_file():
                return str(p)
            cand = p / "verilator"
            return str(cand) if cand.is_file() else None
        return shutil.which("verilator")

    def available(self) -> bool:
        return self.binary() is not None

    def version(self) -> str | None:
        v = self.binary()
        if v is None:
            return None
        try:
            out = subprocess.run([v, "--version"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return (out.stdout or "").strip() or None

    # ------------------------------------------------------------------ stages

    def _run(self, argv: list[str], stage: str, top: str, workdir: Path,
             timeout_s: int) -> VerilatorSvResult:
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout_s, cwd=str(workdir))
        except subprocess.TimeoutExpired:
            return VerilatorSvResult(stage=stage, ok=False, top=top,
                                     runtime_s=time.monotonic() - t0,
                                     errors=[f"timed out after {timeout_s}s"])
        log = (proc.stdout or "") + (proc.stderr or "")
        (workdir / f"{stage}.log").write_text(log)
        errors, warnings = _classify(log)
        return VerilatorSvResult(
            stage=stage, ok=proc.returncode == 0, returncode=proc.returncode,
            runtime_s=time.monotonic() - t0, errors=errors, warnings=warnings, top=top,
            log_path=str(workdir / f"{stage}.log"), stdout_tail=log[-2000:])

    def lint(self, sources: list[str | Path], top: str, workdir: str | Path,
             include_dirs: list[str | Path] = (), suppress: tuple[str, ...] = DEFAULT_SUPPRESS,
             timeout_s: int = 600) -> VerilatorSvResult:
        """Parse + static checks. The cheapest signal that the agent produced valid HDL at all."""
        v = self.binary()
        if v is None:
            raise VerilatorUnavailable("verilator not found on PATH and none supplied")
        argv = [v, "--lint-only", "-Wall", "--top-module", top]
        for w in suppress:
            if w not in ALWAYS_FATAL:
                argv.append(f"-Wno-{w}")
        for w in ALWAYS_FATAL:
            argv.append(f"-Werror-{w}")
        for d in include_dirs:
            argv += ["-I" + str(d)]
        argv += [str(s) for s in sources]
        return self._run(argv, "lint", top, Path(workdir), timeout_s)

    def elaborate(self, sources: list[str | Path], top: str, workdir: str | Path,
                  include_dirs: list[str | Path] = (), timeout_s: int = 1200
                  ) -> VerilatorSvResult:
        """Full hierarchy resolution and C++ emission, without compiling it.

        Measured difference from :meth:`lint`, which is narrower than one might assume: lint already
        catches a missing submodule (``%Error-MODMISSING``), so this is not "the first check that
        resolves instantiations". What it adds is everything that only shows up when Verilator
        actually has to *emit* a model — parameter resolution through the hierarchy, generate-block
        expansion, and constructs it can parse but not translate. It also costs seconds rather than
        milliseconds, which is why it is a separate checkpoint rather than the only one.
        """
        v = self.binary()
        if v is None:
            raise VerilatorUnavailable("verilator not found on PATH and none supplied")
        argv = [v, "--cc", "--top-module", top, "-Wno-fatal",
                "--Mdir", str(Path(workdir) / "obj_dir")]
        for d in include_dirs:
            argv += ["-I" + str(d)]
        argv += [str(s) for s in sources]
        return self._run(argv, "elaborate", top, Path(workdir), timeout_s)

    def build(self, sources: list[str | Path], top: str, workdir: str | Path,
              tb_cpp: str | Path | None = None, include_dirs: list[str | Path] = (),
              jobs: int = 4, timeout_s: int = 3600) -> VerilatorSvResult:
        """Compile a simulator. Needs a C++ testbench (or ``--main``) to link an executable."""
        v = self.binary()
        if v is None:
            raise VerilatorUnavailable("verilator not found on PATH and none supplied")
        wd = Path(workdir)
        argv = [v, "--cc", "--exe", "--build", f"-j{jobs}", "--top-module", top, "-Wno-fatal",
                "--Mdir", str(wd / "obj_dir")]
        if tb_cpp is None:
            argv += ["--main"]
        for d in include_dirs:
            argv += ["-I" + str(d)]
        argv += [str(s) for s in sources]
        if tb_cpp is not None:
            argv.append(str(tb_cpp))
        return self._run(argv, "build", top, wd, timeout_s)

    def run_sim(self, workdir: str | Path, top: str, args: list[str] = (),
                timeout_s: int = 3600) -> VerilatorSvResult:
        """Execute a simulator built by :meth:`build`."""
        exe = Path(workdir) / "obj_dir" / f"V{top}"
        if not exe.is_file():
            return VerilatorSvResult(stage="run", ok=False, top=top,
                                     errors=[f"no simulator at {exe} — build first"])
        return self._run([str(exe), *args], "run", top, Path(workdir), timeout_s)


def verilator_lint(sources: list[str | Path], top: str, workdir: str | Path,
                   verilator: str | None = None, **kw) -> VerilatorSvResult:
    """Functional entry point; decorate with ``@ChiaFunction`` at the pipeline level and key the
    cache with :func:`chia.base.content_tag.content_tag` over the sources and the tool version."""
    return VerilatorSvNode(verilator=verilator).lint(sources, top, workdir, **kw)


def verilator_elaborate(sources: list[str | Path], top: str, workdir: str | Path,
                        verilator: str | None = None, **kw) -> VerilatorSvResult:
    return VerilatorSvNode(verilator=verilator).elaborate(sources, top, workdir, **kw)
