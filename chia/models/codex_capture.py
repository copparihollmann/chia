"""Streaming capture of a ``codex exec --json`` subprocess.

The single job of this module is to make a Codex run *replayable even when it
dies mid-turn*. It runs the CLI under :class:`subprocess.Popen`, tees every
stdout line to ``events.raw.jsonl`` the instant the line arrives, ``fsync``\\ s
periodically, and feeds each line to an optional live callback. A timeout or a
killed process therefore still leaves a byte-for-byte, line-complete JSONL file
that :func:`chia.models.codex_events.parse_stream_text` can parse offline.

Why not ``subprocess.run(capture_output=True)``: it buffers stdout in memory and
returns it only on exit, so a timeout or a crash loses exactly the events that a
billing/diagnosis reader needs most. See the frozen contract §7 (Phase-1
acceptance: "killing a run mid-turn leaves parseable raw JSONL").

The stream carries no timestamps of its own (contract §1 note 3), so this module
records the *reader's* arrival time per line into a sibling
``events.timestamped.jsonl`` when asked, which is the only honest source of
per-event timing an experiment has.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from chia.models.codex_events import parse_line

# Flush the OS page cache to disk every this many lines. A crash between fsyncs
# still leaves flushed lines on disk (Python's line flush already made them
# visible to a re-reading process); fsync only bounds the loss window against a
# power-cut, so a small-ish cadence is a fine cost/durability trade.
_FSYNC_EVERY = 8

#: A per-line callback receives ``(raw_line_without_newline, parsed_or_None)``.
EventCallback = Callable[[str, "dict[str, Any] | None"], None]


@dataclass
class CaptureResult:
    """Outcome of one streamed ``codex exec`` invocation.

    Records *observations* only — exit code, whether we killed it for timeout,
    the signal it died on — and leaves "did the attempt succeed" to the caller,
    which also owns the parsed event stream and the grading context.
    """

    returncode: int | None = None
    #: POSIX signal name when the process died on a signal (``"SIGKILL"``), else None.
    signal: str | None = None
    timed_out: bool = False
    raw_path: str = ""
    stderr_path: str = ""
    stderr_text: str = ""
    started_at: str = ""
    ended_at: str = ""
    lines_written: int = 0
    #: Parsed event dicts, in arrival order (``parse_line`` returned a dict).
    events: list[dict[str, Any]] = field(default_factory=list)
    #: Raw lines that were not a JSON object, kept verbatim (never dropped).
    unparsed_lines: list[str] = field(default_factory=list)

    @property
    def active_wall_s(self) -> float:
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.ended_at)
        except (ValueError, TypeError):
            return 0.0
        return max((end - start).total_seconds(), 0.0)


def _signal_name(returncode: int | None) -> str | None:
    """Map a negative Popen returncode to its signal name (POSIX)."""
    if returncode is None or returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except (ValueError, AttributeError):
        return f"SIG{-returncode}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stream_codex(
    cmd: Sequence[str],
    *,
    input_text: str | None = None,
    stdin_path: str | None = None,
    raw_path: str,
    stderr_path: str,
    timeout: float | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    on_event: EventCallback | None = None,
    timestamped_path: str | None = None,
    fsync_every: int = _FSYNC_EVERY,
    popen: Callable[..., "subprocess.Popen"] | None = None,
) -> CaptureResult:
    """Run *cmd*, teeing stdout to *raw_path* byte-for-byte as it arrives.

    The prompt is delivered to the child's stdin from a *file* (``stdin_path``,
    or a temp file holding ``input_text``): a regular file cannot deadlock
    against a child that emits before it has drained stdin, so no feeder thread
    is needed, and the on-disk bytes are the reproducible prompt artifact.

    stdout is read at the fd level in chunks (``os.read``, never ``readline`` —
    a line lacking a trailing newline must not be able to block the reader past
    its deadline) and mirrored to *raw_path* verbatim. ``on_event`` is called
    per complete line with the decoded text and its parsed form (``None`` when
    the line is not a JSON object).

    On *timeout* the process is killed and ``timed_out`` is set, but *raw_path*
    still holds everything the child had flushed — including a final PARTIAL
    line (no newline) written when it was killed mid-write. That salvage
    guarantee is the whole reason this streams rather than buffers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)) or ".", exist_ok=True)
    _popen = popen or subprocess.Popen
    result = CaptureResult(raw_path=raw_path, stderr_path=stderr_path, started_at=_now())
    deadline = None if timeout is None else time.monotonic() + timeout

    # Prompt via a file, not a pipe (avoids the classic write-stdin-while-reading
    # -stdout deadlock without a thread). A temp file is used only when the
    # caller passed raw text instead of an on-disk prompt path.
    tmp_stdin: str | None = None
    if stdin_path is None:
        fd, tmp_stdin = tempfile.mkstemp(prefix="codex_prompt_")
        with os.fdopen(fd, "w", encoding="utf-8") as pf:
            pf.write(input_text or "")
        stdin_path = tmp_stdin
    stdin_f = open(stdin_path, "rb")

    proc = _popen(
        list(cmd),
        stdin=stdin_f,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,  # unbuffered bytes; we frame lines ourselves
        cwd=cwd,
        env=env,
    )

    stdout_fd = proc.stdout.fileno()
    raw_file = open(raw_path, "wb")
    ts_file = open(timestamped_path, "w", encoding="utf-8") if timestamped_path else None
    line_buf = b""
    since_fsync = 0
    _closed = False

    def _close_files() -> None:
        # Idempotent: an early-return path and the finally both call this.
        nonlocal _closed
        if _closed:
            return
        _closed = True
        try:
            raw_file.flush()
            os.fsync(raw_file.fileno())
        except (OSError, ValueError):
            pass
        try:
            raw_file.close()
        except (OSError, ValueError):
            pass
        if ts_file is not None:
            try:
                ts_file.close()
            except (OSError, ValueError):
                pass

    def _emit_line(bline: bytes) -> None:
        """Account one complete logical line for parsing/callback (already tee'd)."""
        nonlocal since_fsync
        text = bline.decode("utf-8", "replace")
        result.lines_written += 1
        event = parse_line(text)
        if event is None:
            if text.strip():
                result.unparsed_lines.append(text)
        else:
            result.events.append(event)
        if ts_file is not None:
            ts_file.write(json.dumps({"ts": _now(), "line": text}) + "\n")
            ts_file.flush()
        if on_event is not None:
            try:
                on_event(text, event)
            except Exception:
                # A live recorder is telemetry; it must never break capture.
                pass
        since_fsync += 1
        if since_fsync >= fsync_every:
            try:
                os.fsync(raw_file.fileno())
            except OSError:
                pass
            since_fsync = 0

    def _consume(chunk: bytes) -> None:
        """Tee raw bytes to disk immediately, then frame and emit whole lines."""
        nonlocal line_buf
        raw_file.write(chunk)
        raw_file.flush()  # visible to a concurrent re-reader at once (salvage)
        line_buf += chunk
        while b"\n" in line_buf:
            line, line_buf = line_buf.split(b"\n", 1)
            _emit_line(line)

    try:
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    result.timed_out = True
                    _kill(proc)
                    break
            else:
                remaining = None

            ready, _, _ = select.select([stdout_fd], [], [], remaining)
            if not ready:
                continue  # select woke on the deadline; loop re-checks it

            chunk = os.read(stdout_fd, 65536)
            if chunk == b"":
                break  # EOF: the child closed stdout (normal completion)
            _consume(chunk)

        # Drain whatever the child had already written into the pipe before we
        # broke out — a timeout kill in particular. Non-blocking, to EOF/empty,
        # so no flushed byte is lost from the raw file.
        while True:
            ready, _, _ = select.select([stdout_fd], [], [], 0)
            if not ready:
                break
            try:
                chunk = os.read(stdout_fd, 65536)
            except OSError:
                break
            if chunk == b"":
                break
            _consume(chunk)

        # A trailing PARTIAL line (no newline) — the last thing a mid-write kill
        # leaves — is already on disk byte-for-byte via the raw tee; account it
        # for parsing too so it survives verbatim as an (unparsed) record.
        if line_buf.strip():
            _emit_line(line_buf)
            line_buf = b""
    finally:
        _close_files()

    # stdout is already drained; collect stderr and reap.
    try:
        _, stderr_bytes = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _kill(proc)
        _, stderr_bytes = proc.communicate()
    stderr_text = (stderr_bytes or b"").decode("utf-8", "replace")
    with open(stderr_path, "w", encoding="utf-8") as sf:
        sf.write(stderr_text)

    try:
        stdin_f.close()
    except OSError:
        pass
    if tmp_stdin is not None:
        try:
            os.unlink(tmp_stdin)
        except OSError:
            pass

    result.returncode = proc.returncode
    result.signal = _signal_name(proc.returncode)
    result.stderr_text = stderr_text
    result.ended_at = _now()
    return result


def _kill(proc: "subprocess.Popen") -> None:
    """Best-effort terminate → kill of a child that overran its deadline."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
    except (ProcessLookupError, OSError):
        pass
