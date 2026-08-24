"""The output-side leak scan: did anything leave the run carrying content that was withheld?

:mod:`chia.base.sandbox` closes the **filesystem** channel — an agent cannot read the answer
key because the answer key is not in its view. Nothing checks the **output** channel. An agent
that saw withheld content some other way still scores: a harness bug that pasted a golden
reference into the prompt, a tool whose output crossed the boundary, a resumed session
carrying a previous call's context, a backend that is not sandboxable at all
(``supports_sandbox = False`` on every raw-API backend, and honestly so). In each case the
sandbox is intact and the measurement is worthless, and nothing in chia would say so.

So this module asks the complementary question. The sandbox asks *can it reach the answer*;
this asks *did the answer come out*. Both are needed, because a run can pass either one alone.

The marker set comes from the sandbox spec
--------------------------------------------

A leak scan needs to know what was withheld, and that is exactly what a
:class:`~chia.base.sandbox.SandboxSpec` already declares: ``deny`` and ``mask_files`` **are**
the withheld set, ``allow`` / ``rw_binds`` / ``extra_binds`` **are** the granted set. So
:func:`markers_from_spec` reads the spec rather than taking a second list, and there is no way
for the scan's idea of the boundary to drift from the boundary the sandbox actually built. Move
a directory from ``deny`` to ``allow`` and the scan follows in the same commit.

Distinctiveness is computed, not assumed
----------------------------------------

A marker is a line of a withheld file that is long enough to be contentful **and occurs
nowhere in the granted corpus**. That second half is the part a naive implementation omits, and
it is what stops the scan convicting an agent for quoting the document it was handed: a
withheld ``MXU_SPEC.md`` and a granted ``SPEC.md`` restate each other's sentences, and a line
they share is evidence of nothing.

Two codes, not three
--------------------

``SUPPLIED``
    A prompt contained withheld content. A harness defect, and the more serious of the two,
    because it means the leak was *handed over* — no amount of model integrity could have
    avoided it, and every score in the run is uninterpretable.
``COPIED``
    A response, transcript, or workspace artifact contained withheld content. The agent
    reached it somehow.

The reference implementation this design follows (``specir.nl2spec.leakscan`` in a downstream
project, from which the granted-line subtraction and both thresholds are taken) has a third
code for "a candidate process reached a gold path", raised by a live filesystem probe. chia
does not need it: the sandbox *is* that control, and a path that is masked cannot be reached.
A code for a check this layer never performs would be a permanent zero in every report.

What this module does not do
----------------------------

It does not decide what a finding means. A hit voids a measurement in some study designs and
merely annotates it in others — an MLIR line echoed from a reference module is weak evidence
where an identical line of English prose is strong. :class:`ScanReport` reports; the caller
judges. There is no ``raise`` on the happy path and no global severity table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from chia.base.sandbox import SandboxSpec

PathLike = Union[str, os.PathLike]

#: Shortest line usable as a marker. ``"---"`` or ``"## Notes"`` occurs in a thousand
#: documents, and a scanner that fired on one would be switched off within a day.
MIN_MARKER_LEN = 24

#: Fewest whitespace-separated tokens a marker may have. **Set by a measurement, not a guess.**
#: The reference implementation's first run over recorded grids reported 15 hits, every one the
#: same shape: a withheld script enumerating a tier's documents, and a candidate enumerating
#: the same list from the file names it was *given*. The line is absent from the granted prose,
#: so subtracting granted lines does not remove it — yet nothing about it is secret; it is a
#: rearrangement of material the prompt already contained. Sentence-scale content is not
#: reconstructible that way, so a marker has to be a sentence rather than a fragment. Six
#: tokens excluded every observed false positive and kept every real one.
MIN_MARKER_WORDS = 6

#: Suffixes read when expanding a withheld or granted directory.
#:
#: **Source code belongs here, not just documentation.** The reference implementation shipped a
#: documentation-only list; one benchmark withheld its Chisel (``.scala``) and the scan silently
#: found zero files, produced zero markers, and reported "findings: none" — a run with no
#: output-side verification at all, wearing a clean bill of health. That is why
#: :attr:`ScanReport.skipped` exists and why :meth:`ScanReport.scanned` is not simply
#: ``not findings``: "clean" and "could not look" must never render the same.
TEXT_SUFFIXES = (
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".csv", ".toml", ".ini", ".cfg",
    ".py", ".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".go", ".java", ".sh", ".tcl", ".mk",
    ".scala", ".sv", ".v", ".vh", ".vhd", ".chisel", ".mlir", ".td", ".ll", ".s", ".S",
)

#: The two codes this module emits.
SUPPLIED = "SUPPLIED"
COPIED = "COPIED"
CODES = (SUPPLIED, COPIED)

#: Files larger than this are not read. A withheld build artifact can be gigabytes, and a
#: scanner that tried to line-split one would take the run down with it. Skipping is recorded.
MAX_FILE_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One withheld line found in one place it should not be.

    :param code: :data:`SUPPLIED` or :data:`COPIED`.
    :param where: Label of the text the marker was found in — an artifact path, or a name
        like ``"prompt"`` / ``"response"``.
    :param source: The withheld file the marker came from.
    :param marker: The offending line.
    :type code: str
    :type where: str
    :type source: str
    :type marker: str
    """

    code: str
    where: str
    source: str
    marker: str

    def to_dict(self) -> dict:
        """A JSON-safe dict. The marker is truncated: a findings file is often committed, and
        writing the whole withheld line into it would leak the thing being reported."""
        return {"code": self.code, "where": self.where, "source": self.source,
                "marker": self.marker[:120]}


@dataclass
class ScanReport:
    """What a scan looked at, what it found, and what it could not look at.

    :param findings: Every hit, in discovery order.
    :param markers_used: How many distinct withheld lines were distinctive enough to convict on.
    :param texts_scanned: How many texts/artifacts were examined.
    :param withheld_files: How many withheld files contributed markers.
    :param skipped: Why the scan could not be complete — an unreadable file, an oversized one,
        a withheld path that does not exist on this host.
    :type findings: List[Finding]
    :type markers_used: int
    :type texts_scanned: int
    :type withheld_files: int
    :type skipped: List[str]
    """

    findings: List[Finding] = field(default_factory=list)
    markers_used: int = 0
    texts_scanned: int = 0
    withheld_files: int = 0
    skipped: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """No findings **and** the scan was able to run.

        Not ``not self.findings``: a scan that derived zero markers finds nothing by
        construction, and reporting that as clean is how a study ends up with an unverified
        arm that looks verified.
        """
        return not self.findings and self.scanned

    @property
    def scanned(self) -> bool:
        """Whether the scan had anything to work with. Zero markers means it did not."""
        return self.markers_used > 0

    def codes(self) -> Dict[str, int]:
        """``code -> count``, for the codes that actually occurred."""
        out: Dict[str, int] = {}
        for finding in self.findings:
            out[finding.code] = out.get(finding.code, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "clean": self.clean,
            "scanned": self.scanned,
            "markers_used": self.markers_used,
            "texts_scanned": self.texts_scanned,
            "withheld_files": self.withheld_files,
            "codes": self.codes(),
            "findings": [f.to_dict() for f in self.findings],
            "skipped": list(self.skipped),
        }

    def extend(self, other: "ScanReport") -> "ScanReport":
        """Fold another report in — for a run whose output arrives in several pieces."""
        self.findings.extend(other.findings)
        self.texts_scanned += other.texts_scanned
        self.skipped.extend(other.skipped)
        self.markers_used = max(self.markers_used, other.markers_used)
        self.withheld_files = max(self.withheld_files, other.withheld_files)
        return self


# ---------------------------------------------------------------------------
# Building the marker set
# ---------------------------------------------------------------------------


def is_marker(line: str) -> bool:
    """Whether a stripped line is contentful enough to convict on.

    See :data:`MIN_MARKER_LEN` and :data:`MIN_MARKER_WORDS`.
    """
    return len(line) >= MIN_MARKER_LEN and len(line.split()) >= MIN_MARKER_WORDS


def _read(path: Path, skipped: Optional[List[str]] = None) -> str:
    """A text file's contents, or ``""`` — never raising, always recording why it gave up."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        if skipped is not None:
            skipped.append(f"{path}: {type(exc).__name__}")
        return ""
    if size > MAX_FILE_BYTES:
        if skipped is not None:
            skipped.append(f"{path}: {size} bytes exceeds MAX_FILE_BYTES")
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        if skipped is not None:
            skipped.append(f"{path}: {type(exc).__name__}")
        return ""


def long_lines(text: str, *, contentful: bool = True) -> List[str]:
    """Stripped lines of *text* worth considering.

    :param contentful: ``True`` for markers — what a finding convicts on. ``False`` for the
        granted-corpus subtrahend — what a finding is forgiven for.
    :type contentful: bool

    The asymmetry is deliberate. A *short* granted line should still cancel a short withheld
    one, so the subtrahend must not be narrowed by the rule that narrows the markers. Applying
    ``is_marker`` to both would let a line that is granted-but-unconvictable survive
    subtraction and then convict from the withheld side.
    """
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < MIN_MARKER_LEN:
            continue
        if contentful and not is_marker(line):
            continue
        out.append(line)
    return out


def _text_files(paths: Iterable[PathLike], skipped: Optional[List[str]] = None,
                ) -> List[Tuple[str, Path]]:
    """``(label, path)`` for every readable text file under *paths*, expanding directories.

    A declared path that does not exist on this host is recorded rather than ignored: a spec
    written for one node applied on another is exactly how a scan ends up with no markers.
    """
    out: List[Tuple[str, Path]] = []
    for entry in paths:
        path = Path(entry)
        if path.is_file():
            out.append((str(path), path))
        elif path.is_dir():
            for found in sorted(path.rglob("*")):
                if found.is_file() and found.suffix in TEXT_SUFFIXES:
                    out.append((str(found), found))
        elif skipped is not None:
            skipped.append(f"{path}: not present on this host")
    return out


def granted_lines(spec: SandboxSpec, skipped: Optional[List[str]] = None) -> Set[str]:
    """Every long line of everything the sandbox *granted*. The subtrahend.

    ``allow``, ``rw_binds`` and ``extra_binds`` together are what the agent was handed. A line
    it could legitimately have read is not evidence that it read something else.

    ``system_dirs`` is deliberately **not** included. It is ``/usr``, ``/etc``, ``/lib`` — tens
    of thousands of files whose contents have nothing to do with the answer, and walking them
    would cost minutes per scan to subtract lines no withheld document contains anyway.
    """
    out: Set[str] = set()
    for _, path in _text_files(
            list(spec.allow) + list(spec.rw_binds) + list(spec.extra_binds), skipped):
        out.update(long_lines(_read(path, skipped), contentful=False))
    return out


def markers_from_spec(spec: SandboxSpec, *, min_len: int = MIN_MARKER_LEN,
                      ) -> Tuple[Dict[str, str], ScanReport]:
    """``(markers, report)`` where *markers* maps a withheld line to the file it came from.

    :param spec: The spec whose ``deny`` / ``mask_files`` define the withheld set.
    :param min_len: Override the marker length floor. Raising it trades sensitivity for
        precision on a corpus of short lines.
    :type spec: chia.base.sandbox.SandboxSpec
    :type min_len: int
    :rtype: Tuple[Dict[str, str], ScanReport]

    The returned report carries ``markers_used``, ``withheld_files`` and any ``skipped``
    reasons, already populated — so a caller that gets zero markers learns *why* rather than
    concluding the run was clean.
    """
    report = ScanReport()
    granted = granted_lines(spec, report.skipped)

    markers: Dict[str, str] = {}
    withheld = _text_files(list(spec.deny) + list(spec.mask_files), report.skipped)
    for label, path in withheld:
        for line in long_lines(_read(path, report.skipped)):
            if len(line) < min_len:
                continue
            # First withheld file wins the attribution; the line is the same either way, and
            # reporting one source is more useful than reporting that several files share it.
            if line not in granted and line not in markers:
                markers[line] = label

    report.markers_used = len(markers)
    report.withheld_files = len(withheld)
    if not markers and withheld:
        report.skipped.append(
            f"{len(withheld)} withheld files yielded no distinctive lines "
            f"(every long line also occurs in the granted corpus)")
    return markers, report


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_text(text: str, markers: Mapping[str, str], *, where: str = "text",
              code: str = COPIED) -> List[Finding]:
    """Every withheld line that appears verbatim in *text*.

    Substring containment rather than line equality: a model that copies a withheld line
    *into* a sentence ("the reference says: ...") has copied it just the same.
    """
    if not text:
        return []
    return [Finding(code=code, where=where, source=source, marker=marker)
            for marker, source in markers.items() if marker in text]


def scan_texts(texts: Sequence[Tuple[str, str]], markers: Mapping[str, str], *,
               code: str = COPIED) -> ScanReport:
    """Scan ``(label, text)`` pairs in memory — before anything has been written down."""
    report = ScanReport(markers_used=len(markers))
    for label, text in texts:
        report.texts_scanned += 1
        report.findings.extend(scan_text(text, markers, where=label, code=code))
    return report


def scan_result(result, markers: Mapping[str, str], *, prompt: Optional[str] = None,
                ) -> ScanReport:
    """Scan a :class:`~chia.base.llm_call.QueryResult`, and the prompt that produced it.

    :param result: Any object with ``result`` / ``stream_result`` / ``stderr`` attributes.
    :param markers: From :func:`markers_from_spec`.
    :param prompt: The prompt text, scanned under :data:`SUPPLIED` when given.
    :type markers: Mapping[str, str]
    :type prompt: Optional[str]
    :rtype: ScanReport

    ``stream_result`` matters as much as ``result``: the final answer may be clean while the
    transcript shows the agent reading and discussing withheld content mid-run, and the
    transcript is what a reader of the recorded results will actually have. ``stderr`` is
    included because a tool that failed on a withheld path can print it.
    """
    pairs: List[Tuple[str, str]] = []
    if prompt:
        pairs.append(("prompt", prompt))
    report = scan_texts(pairs, markers, code=SUPPLIED) if pairs else ScanReport(
        markers_used=len(markers))

    outputs = [(name, getattr(result, name, "") or "")
               for name in ("result", "stream_result", "stderr")]
    report.extend(scan_texts([(n, t) for n, t in outputs if t], markers, code=COPIED))
    return report


def scan_workspace(workspace: PathLike, markers: Mapping[str, str], *,
                   code: str = COPIED) -> ScanReport:
    """Scan every text file an agent left in its workspace.

    The workspace is the sandbox's one writable directory, so it is where a copied answer
    would land if the agent wrote it to a file instead of saying it. Labels are relative to
    *workspace*, so a report does not carry the host's temp-directory path.
    """
    root = Path(workspace)
    report = ScanReport(markers_used=len(markers))
    if not root.is_dir():
        report.skipped.append(f"{root}: workspace not present")
        return report
    for _, path in _text_files([root], report.skipped):
        text = _read(path, report.skipped)
        if not text:
            continue
        report.texts_scanned += 1
        try:
            label = str(path.relative_to(root))
        except ValueError:  # pragma: no cover — rglob cannot escape its root
            label = str(path)
        report.findings.extend(scan_text(text, markers, where=label, code=code))
    return report


def scan_call(spec: SandboxSpec, result, *, prompt: Optional[str] = None,
              workspace: Optional[PathLike] = None) -> ScanReport:
    """The whole check for one sandboxed call: derive markers from *spec*, scan everything.

    :param spec: The spec the call ran under.
    :param result: The call's ``QueryResult``.
    :param prompt: The prompt, scanned under :data:`SUPPLIED`.
    :param workspace: Defaults to ``spec.workspace``.
    :rtype: ScanReport

    The convenience path, and the one worth calling: passing the same spec that built the
    sandbox is what guarantees the scan and the isolation describe the same boundary.
    """
    markers, report = markers_from_spec(spec)
    report.extend(scan_result(result, markers, prompt=prompt))
    target = workspace if workspace is not None else spec.workspace
    if target is not None:
        report.extend(scan_workspace(target, markers))
    report.markers_used = len(markers)
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(report: ScanReport, *, limit: int = 20) -> str:
    """One screen. A clean scan says so out loud rather than printing nothing.

    A scan that could not run says *that*, and says it first — the failure mode this guards
    against is a reader seeing an empty findings list and moving on.
    """
    lines = [
        "leak scan",
        f"  markers: {report.markers_used} distinctive withheld lines "
        f"from {report.withheld_files} files",
        f"  texts scanned: {report.texts_scanned}",
    ]
    if not report.scanned:
        lines.append("  NOT SCANNED: no markers were derived, so nothing could be found. "
                     "This is not a clean result.")
    elif report.findings:
        counts = ", ".join(f"{code}={n}" for code, n in sorted(report.codes().items()))
        lines.append(f"  FINDINGS: {len(report.findings)} ({counts})")
        for finding in report.findings[:limit]:
            lines.append(f"    {finding.code} {finding.where} <- {finding.source}: "
                         f"{finding.marker[:80]!r}")
        if len(report.findings) > limit:
            lines.append(f"    ... {len(report.findings) - limit} more")
    else:
        lines.append("  findings: none")
    for reason in report.skipped[:limit]:
        lines.append(f"  skipped: {reason}")
    if len(report.skipped) > limit:
        lines.append(f"  skipped: ... {len(report.skipped) - limit} more")
    return "\n".join(lines)
