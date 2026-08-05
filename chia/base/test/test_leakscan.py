"""Tests for :mod:`chia.base.leakscan`.

The important tests here are the ones about *not* firing. A scanner that convicts on every
shared sentence gets switched off within a day, and a switched-off scanner is worse than none
because the run still carries its name. So most of what follows pins the boundaries: what is
too short to convict on, what the granted corpus forgives, and what a scan that could not run
is required to say about itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chia.base.leakscan import (
    COPIED,
    MIN_MARKER_WORDS,
    SUPPLIED,
    Finding,
    ScanReport,
    format_report,
    is_marker,
    long_lines,
    markers_from_spec,
    scan_call,
    scan_result,
    scan_text,
    scan_workspace,
)
from chia.base.sandbox import SandboxSpec


#: A sentence long enough and wordy enough to convict on, and distinctive by construction.
SECRET = "The multiplier array rounds per step in k-order, never at the accumulator boundary."
SHARED = "This document describes the memory hierarchy of the accelerator under test."


def _spec(tmp_path, *, deny=(), allow=(), mask_files=(), rw_binds=(), extra_binds=()):
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return SandboxSpec(workspace=workspace, deny=list(deny), allow=list(allow),
                       mask_files=list(mask_files), rw_binds=list(rw_binds),
                       extra_binds=list(extra_binds))


def _write(path, *lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- markers
def test_a_marker_must_be_long_and_wordy():
    assert is_marker(SECRET)
    assert not is_marker("## Notes")                        # too short
    assert not is_marker("supercalifragilisticexpialidocious_identifier_name")  # one word
    # long enough on characters, one word short -- so this fails on the word rule alone,
    # which is the rule a measurement put there and a rewrite would be tempted to drop
    short_on_words = " ".join(["abcdefgh"] * (MIN_MARKER_WORDS - 1))
    assert len(short_on_words) >= 24
    assert not is_marker(short_on_words)
    assert is_marker(short_on_words + " abcdefgh")


def test_markers_come_from_the_denied_directory(tmp_path):
    gold = tmp_path / "gold"
    _write(gold / "ANSWERS.md", SECRET)
    markers, report = markers_from_spec(_spec(tmp_path, deny=[gold]))

    assert SECRET in markers
    assert markers[SECRET].endswith("ANSWERS.md")
    assert report.markers_used == 1
    assert report.withheld_files == 1


def test_a_masked_file_is_withheld_too(tmp_path):
    """``mask_files`` is the per-file half of the boundary; a scan keyed only off ``deny``
    would miss exactly the files a spec singles out as most sensitive."""
    answer = _write(tmp_path / "reference" / "golden.md", SECRET)
    markers, _ = markers_from_spec(_spec(tmp_path, mask_files=[answer]))
    assert SECRET in markers


def test_a_line_shared_with_an_allowed_file_is_not_a_marker(tmp_path):
    """THE test. Without granted-line subtraction, an agent gets convicted for quoting the
    document it was handed -- and the two document sets overlap in practice, because a
    withheld spec and a granted spec restate each other.
    """
    gold = tmp_path / "gold"
    granted = tmp_path / "granted"
    _write(gold / "ANSWERS.md", SHARED, SECRET)
    _write(granted / "SPEC.md", SHARED)

    markers, _ = markers_from_spec(_spec(tmp_path, deny=[gold], allow=[granted]))

    assert SECRET in markers      # the line only the withheld file has
    assert SHARED not in markers  # the line both have


@pytest.mark.parametrize("field", ["allow", "rw_binds", "extra_binds"])
def test_every_granted_channel_subtracts(tmp_path, field):
    """All three are things the agent was handed. A scan that only subtracted ``allow`` would
    convict on the toolchain's own README."""
    gold = tmp_path / "gold"
    granted = tmp_path / "granted"
    _write(gold / "ANSWERS.md", SHARED)
    _write(granted / "README.md", SHARED)

    markers, _ = markers_from_spec(_spec(tmp_path, deny=[gold], **{field: [granted]}))
    assert SHARED not in markers


def test_a_short_granted_line_still_cancels_a_short_withheld_one(tmp_path):
    """The subtrahend is deliberately wider than the marker rule.

    A line can be long enough to appear in the granted corpus but too short to be a marker;
    narrowing both sides by the same rule would let such a line survive subtraction.
    """
    short_shared = "The accelerator has four banks."      # >= MIN_MARKER_LEN, < MIN_MARKER_WORDS
    assert not is_marker(short_shared)
    assert short_shared in long_lines(short_shared, contentful=False)
    assert short_shared not in long_lines(short_shared, contentful=True)


def test_system_dirs_do_not_subtract(tmp_path):
    """``system_dirs`` is /usr, /etc, /lib -- tens of thousands of files whose contents have
    nothing to do with the answer. Walking them would cost minutes per scan to remove lines no
    withheld document contains, so they are deliberately not part of the subtrahend.

    Asserted directly: a line present in a directory named *only* under ``system_dirs`` stays a
    marker, where the same line under ``allow`` would not.
    """
    gold, fake_system = tmp_path / "gold", tmp_path / "fake_usr"
    _write(gold / "ANSWERS.md", SECRET)
    _write(fake_system / "share" / "doc.md", SECRET)

    as_system = SandboxSpec(workspace=tmp_path / "ws", deny=[gold],
                            system_dirs=[str(fake_system)])
    as_allowed = SandboxSpec(workspace=tmp_path / "ws", deny=[gold], allow=[fake_system])

    assert SECRET in markers_from_spec(as_system)[0]      # not subtracted
    assert SECRET not in markers_from_spec(as_allowed)[0]  # subtracted


# --------------------------------------------------------------------------- scanning
def test_a_copied_line_is_found_inside_a_sentence(tmp_path):
    """Containment, not line equality: quoting a withheld line into prose is still copying."""
    markers = {SECRET: "ANSWERS.md"}
    findings = scan_text(f"According to the reference, {SECRET} That explains the result.",
                         markers, where="response")
    assert len(findings) == 1
    assert findings[0].code == COPIED
    assert findings[0].source == "ANSWERS.md"


def test_a_prompt_hit_is_supplied_not_copied():
    """The distinction is the whole reason there are two codes: SUPPLIED means the harness
    handed the answer over, and no model behaviour could have avoided it."""
    result = SimpleNamespace(result="42", stream_result="", stderr="")
    report = scan_result(result, {SECRET: "ANSWERS.md"},
                         prompt=f"Here is context: {SECRET}")

    assert [f.code for f in report.findings] == [SUPPLIED]
    assert report.codes() == {SUPPLIED: 1}


def test_the_transcript_is_scanned_not_only_the_final_answer():
    """A clean final answer over a transcript that shows the agent reading withheld content
    is not a clean run, and the transcript is what a reader of the results will have."""
    result = SimpleNamespace(
        result="The answer is 42.",
        stream_result=f"[tool] cat gold/ANSWERS.md\n{SECRET}\n",
        stderr="")
    report = scan_result(result, {SECRET: "ANSWERS.md"})

    assert [f.where for f in report.findings] == ["stream_result"]


def test_stderr_is_scanned(tmp_path):
    """A tool that failed on a withheld path prints it, and stderr is recorded durably."""
    result = SimpleNamespace(result="", stream_result="",
                             stderr=f"error while reading: {SECRET}")
    report = scan_result(result, {SECRET: "ANSWERS.md"})
    assert [f.where for f in report.findings] == ["stderr"]


def test_a_clean_result_is_clean():
    result = SimpleNamespace(result="42", stream_result="thinking...", stderr="")
    report = scan_result(result, {SECRET: "ANSWERS.md"}, prompt="What is 6 times 7?")
    assert report.findings == []
    assert report.clean


# --------------------------------------------------------------------------- workspace
def test_a_written_artifact_is_found(tmp_path):
    """An agent that writes the answer to a file rather than saying it has still copied it."""
    workspace = tmp_path / "ws"
    _write(workspace / "notes" / "design.md", "My design:", SECRET)

    report = scan_workspace(workspace, {SECRET: "ANSWERS.md"})

    assert len(report.findings) == 1
    # labelled relative to the workspace, so a report carries no host temp path
    assert report.findings[0].where == "notes/design.md"


def test_a_binary_artifact_is_not_scanned(tmp_path):
    """Suffix filtering, so a scan never line-splits an ELF or a waveform dump."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "sim.vcd").write_bytes(b"\x00\x01\x02" + SECRET.encode() + b"\x00")

    report = scan_workspace(workspace, {SECRET: "ANSWERS.md"})
    assert report.findings == []
    assert report.texts_scanned == 0


def test_an_oversized_file_is_skipped_and_recorded(tmp_path, monkeypatch):
    """Skipped, and *recorded* as skipped: an unread file is a hole in the scan, not a pass."""
    from chia.base import leakscan

    monkeypatch.setattr(leakscan, "MAX_FILE_BYTES", 16)
    workspace = tmp_path / "ws"
    _write(workspace / "big.md", SECRET)

    report = scan_workspace(workspace, {SECRET: "ANSWERS.md"})
    assert report.findings == []
    assert any("MAX_FILE_BYTES" in reason for reason in report.skipped)


# --------------------------------------------------------------------------- honest reporting
def test_a_scan_with_no_markers_is_not_clean(tmp_path):
    """The failure that hides itself.

    A spec whose withheld paths do not exist on this host derives no markers, finds nothing,
    and would report a clean run -- leaving an arm with no output-side verification at all
    while wearing a clean bill of health.
    """
    spec = _spec(tmp_path, deny=[tmp_path / "does-not-exist"])
    markers, report = markers_from_spec(spec)

    assert markers == {}
    assert not report.scanned
    assert not report.clean          # <- not the same as "no findings"
    assert any("not present" in reason for reason in report.skipped)
    assert "NOT SCANNED" in format_report(report)


def test_all_withheld_lines_shared_is_reported_as_a_reason(tmp_path):
    """Zero markers from files that *were* read is a different hole, and says so."""
    gold, granted = tmp_path / "gold", tmp_path / "granted"
    _write(gold / "ANSWERS.md", SHARED)
    _write(granted / "SPEC.md", SHARED)

    _, report = markers_from_spec(_spec(tmp_path, deny=[gold], allow=[granted]))
    assert not report.scanned
    assert any("no distinctive lines" in reason for reason in report.skipped)


def test_format_report_states_a_clean_result_explicitly():
    report = ScanReport(markers_used=12, withheld_files=3, texts_scanned=4)
    text = format_report(report)
    assert "findings: none" in text
    assert "12 distinctive withheld lines" in text


def test_to_dict_truncates_the_marker():
    """A findings file is often committed. Writing the whole withheld line into it would leak
    the very thing being reported."""
    long_marker = SECRET + " " + "x" * 400
    payload = Finding(code=COPIED, where="response", source="ANSWERS.md",
                      marker=long_marker).to_dict()
    assert len(payload["marker"]) == 120
    assert long_marker not in payload["marker"]


# --------------------------------------------------------------------------- the whole call
def test_scan_call_derives_the_boundary_from_the_spec_that_built_the_sandbox(tmp_path):
    """The convenience path, and the reason it exists: one spec, so the scan and the isolation
    cannot describe different boundaries."""
    gold, granted = tmp_path / "gold", tmp_path / "granted"
    _write(gold / "ANSWERS.md", SHARED, SECRET)
    _write(granted / "SPEC.md", SHARED)
    workspace = tmp_path / "ws"
    _write(workspace / "answer.md", SECRET)

    spec = SandboxSpec(workspace=workspace, deny=[gold], allow=[granted])
    result = SimpleNamespace(result=f"I found that {SECRET}", stream_result="", stderr="")

    report = scan_call(spec, result, prompt=f"Context: {SHARED}")

    assert report.scanned
    assert not report.clean
    # the shared line in the prompt is forgiven; the withheld line is caught in both places
    assert report.codes() == {COPIED: 2}
    assert {f.where for f in report.findings} == {"result", "answer.md"}
