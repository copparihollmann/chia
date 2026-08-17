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
import threading
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
    input_text: str,
    raw_path: str,
    stderr_path: str,
    timeout: float | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    on_event: EventCallback | None = None,
    timestamped_path: str | None = None,
    fsync_every: int = _FSYNC_EVERY,
    popen: Callable[..., "subprocess.Popen[str]"] | None = None,
) -> CaptureResult:
    """Run *cmd*, teeing stdout JSONL to *raw_path* line by line as it arrives.

    ``input_text`` is written to the child's stdin (in a helper thread so a large
    prompt cannot deadlock against a child that starts emitting before it has
    drained stdin). ``on_event`` is called for every stdout line with the raw
    line and its parsed form (or ``None`` when the line is not a JSON object).

    On *timeout* the process is killed and ``timed_out`` is set, but *raw_path*
    remains a complete, parseable JSONL of everything seen up to the kill — that
    salvage guarantee is the whole point of streaming rather than buffering.
    """
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)) or ".", exist_ok=True)
    _popen = popen or subprocess.Popen
    result = CaptureResult(raw_path=raw_path, stderr_path=stderr_path, started_at=_now())
    deadline = None if timeout is None else time.monotonic() + timeout

    proc = _popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=cwd,
        env=env,
    )

    def _feed_stdin() -> None:
        try:
            if proc.stdin is not None:
                proc.stdin.write(input_text)
                proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass

    stdin_thread = threading.Thread(target=_feed_stdin, daemon=True)
    stdin_thread.start()

    raw_file = open(raw_path, "w", encoding="utf-8")
    ts_file = open(timestamped_path, "w", encoding="utf-8") if timestamped_path else None
    since_fsync = 0

    def _tee(line: str) -> None:
        """Write one stdout line to disk immediately, parse it, and notify."""
        nonlocal since_fsync
        raw_file.write(line)
        raw_file.flush()
        since_fsync += 1
        if since_fsync >= fsync_every:
            os.fsync(raw_file.fileno())
            since_fsync = 0
        result.lines_written += 1
        stripped = line.rstrip("\n")
        event = parse_line(stripped)
        if event is None:
            if stripped.strip():
                result.unparsed_lines.append(stripped)
        else:
            result.events.append(event)
        if ts_file is not None:
            ts_file.write(json.dumps({"ts": _now(), "line": stripped}) + "\n")
            ts_file.flush()
        if on_event is not None:
            try:
                on_event(stripped, event)
            except Exception:
                # A live recorder is telemetry; it must never break capture.
                pass

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

            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                # select timed out; loop recomputes the deadline (and exits).
                continue

            line = proc.stdout.readline()
            if line == "":
                break  # EOF: the child closed stdout (normal completion).
            _tee(line)

        # Salvage any lines that were already sitting in the pipe when we broke
        # out (a timeout kill in particular): drain without blocking so nothing
        # the child had already flushed is lost from the raw file.
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            line = proc.stdout.readline()
            if line == "":
                break
            _tee(line)
    finally:
        raw_file.flush()
        try:
            os.fsync(raw_file.fileno())
        except OSError:
            pass
        raw_file.close()
        if ts_file is not None:
            ts_file.close()

    # Drain whatever the child wrote to stderr and reap it. stdout is already
    # drained above, so only stderr is collected here.
    try:
        _, stderr_text = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _kill(proc)
        _, stderr_text = proc.communicate()
    stderr_text = stderr_text or ""
    with open(stderr_path, "w", encoding="utf-8") as sf:
        sf.write(stderr_text)

    stdin_thread.join(timeout=1)
    result.returncode = proc.returncode
    result.signal = _signal_name(proc.returncode)
    result.stderr_text = stderr_text
    result.ended_at = _now()
    return result


def _kill(proc: "subprocess.Popen[str]") -> None:
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
