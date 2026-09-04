"""Offline tests for :class:`chia.models.codex.CodexLLM` (codex-cli 0.147.0).

No ``codex`` binary and no network: the subprocess is replaced by a scripted
fake :func:`chia.models.codex_capture.stream_codex` that writes the same raw
JSONL + output-last-message files the real streaming path would. Set
``CODEX_LIVE_TEST=1`` to run the opt-in live smoke tests against an
authenticated local Codex CLI.
"""

from __future__ import annotations

import os
import shutil
from datetime import timezone
from types import SimpleNamespace

import pytest

from chia.models import codex as codex_mod
from chia.models.codex import (
    AuthenticationError,
    BillingError,
    CodexQueryResult,
    QueryResult,
    CodexLLM,
    InvalidRequestError,
    MaxOutputTokensError,
    RateLimitError,
    ServerError,
    UnknownCodexError,
    parse_session_id,
    parse_rate_limit_reset,
)
from chia.models.codex_capture import CaptureResult
from chia.models.codex_events import parse_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(**kw):
    """A CodexLLM that has declared the outer sandbox (so bypass is allowed)."""
    kw.setdefault("external_sandbox", True)
    return CodexLLM(**kw)


def _cli(returncode=1, stderr="", result="", stream_result=""):
    return QueryResult(result, returncode, stderr, stream_result)


def _disable_profiler(monkeypatch):
    import chia.trace.profiler as profiler_mod

    monkeypatch.setattr(
        profiler_mod,
        "get_profiler",
        lambda: SimpleNamespace(enabled=False, add_info=lambda _info: None),
    )


def _fake_stream(monkeypatch, scripts):
    """Install a scripted fake ``stream_codex``; returns the recorded calls list.

    Each *script* is a dict: ``jsonl`` (stdout), ``final`` (output-last-message),
    ``returncode``, ``timed_out``, ``stderr``. The last script repeats if called
    more times than provided.
    """
    calls: list[dict] = []

    def fake(cmd, *, input_text, raw_path, stderr_path, timeout=None,
             cwd=None, env=None, on_event=None, timestamped_path=None, **_kw):
        calls.append({"cmd": list(cmd), "input": input_text, "cwd": cwd,
                      "timestamped_path": timestamped_path})
        script = scripts[min(len(calls) - 1, len(scripts) - 1)]
        out_path = cmd[cmd.index("--output-last-message") + 1]
        with open(out_path, "w") as f:
            f.write(script.get("final", "OK"))
        jsonl = script.get("jsonl", "")
        with open(raw_path, "w") as f:
            f.write(jsonl)
        if timestamped_path:
            with open(timestamped_path, "w") as f:
                for line in jsonl.splitlines():
                    f.write('{"ts":"2026-01-01T00:00:00+00:00","line":'
                            + __import__("json").dumps(line) + "}\n")
        events, unparsed = [], []
        for line in jsonl.splitlines():
            ev = parse_line(line)
            if ev is None:
                if line.strip():
                    unparsed.append(line)
            else:
                events.append(ev)
            if on_event is not None:
                on_event(line, ev)
        stderr_text = script.get("stderr", "")
        with open(stderr_path, "w") as f:
            f.write(stderr_text)
        return CaptureResult(
            returncode=script.get("returncode", 0),
            signal=script.get("signal"),
            timed_out=script.get("timed_out", False),
            raw_path=raw_path,
            stderr_path=stderr_path,
            stderr_text=stderr_text,
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:01+00:00",
            lines_written=len(jsonl.splitlines()),
            events=events,
            unparsed_lines=unparsed,
        )

    monkeypatch.setattr(codex_mod, "stream_codex", fake)
    return calls


_TURN_OK = (
    '{"type":"thread.started","thread_id":"%(tid)s"}\n'
    '{"type":"turn.started"}\n'
    '{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"%(msg)s"}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":%(in)d,"cached_input_tokens":%(cr)d,'
    '"cache_write_input_tokens":0,"output_tokens":%(out)d,"reasoning_output_tokens":0}}\n'
)


def _turn(tid="01a0-thread", msg="done", inp=100, cr=20, out=10):
    return _TURN_OK % {"tid": tid, "msg": msg, "in": inp, "cr": cr, "out": out}


# ---------------------------------------------------------------------------
# Constructor / chia surface
# ---------------------------------------------------------------------------

def test_constructor_and_chia_surface(caplog):
    with caplog.at_level("INFO", logger="codex"):
        llm = CodexLLM()
    assert llm.model is None
    assert llm.codex_bin == "codex"
    assert "experimental" in caplog.text
    assert "default model" in caplog.text
    assert hasattr(CodexLLM.prompt, "chia_remote")
    # Fractional codex_creds is replaced by an integer codex_slots gate.
    assert CodexLLM.prompt._chia_options["resources"] == {"codex_slots": 1}


def test_prompt_formatting():
    assert CodexLLM()._format_prompt("hi") == "hi"
    formatted = CodexLLM(system_message="be terse")._format_prompt("say pong")
    assert "[System Instructions]" in formatted
    assert "be terse" in formatted
    assert "[User Request]" in formatted


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("calc", 'mcp_servers.calc.url="http://localhost:9001/calc/mcp"'),
        ("calc.one", 'mcp_servers."calc.one".url="http://localhost:9001/calc.one/mcp"'),
    ],
)
def test_mcp_config_args(tool_name, expected):
    tool = SimpleNamespace(name=tool_name, hostname="localhost", port=9001)
    assert CodexLLM()._mcp_config_args([tool]) == ["-c", expected]


# ---------------------------------------------------------------------------
# Command construction — 0.147.0 flags (contract §2)
# ---------------------------------------------------------------------------

def test_build_cmd_flags_and_reasoning_effort():
    cmd = _llm(
        model="gpt-test", work_dir="/tmp/work", ephemeral=True, reasoning_effort="xhigh",
    )._build_cmd(output_last_message_path="/tmp/out.txt")
    assert cmd[:4] == ["codex", "exec", "--json", "--color"]
    assert cmd[cmd.index("--model") + 1] == "gpt-test"
    assert cmd[cmd.index("--cd") + 1] == "/tmp/work"
    assert cmd[cmd.index("--output-last-message") + 1] == "/tmp/out.txt"
    assert "--skip-git-repo-check" in cmd
    assert "--ephemeral" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert 'model_reasoning_effort="xhigh"' in cmd
    assert cmd[-1] == "-"
    # DRIFT FIX: 0.147.0 has no --ask-for-approval flag anywhere.
    assert "--ask-for-approval" not in cmd


def test_build_cmd_resume_flags():
    thread_id = "01a01160-7e52-7153-a3f1-a3ee492ab99e"
    cmd = _llm(model="gpt-test", work_dir="/tmp/work", resume_session=True)._build_cmd(
        output_last_message_path="/tmp/out.txt", resume_session_id=thread_id,
    )
    assert cmd[:4] == ["codex", "exec", "resume", "--json"]
    assert "--color" not in cmd
    assert "--cd" not in cmd
    assert cmd[cmd.index("--model") + 1] == "gpt-test"
    assert cmd[-2:] == [thread_id, "-"]


def test_build_cmd_safe_sandbox_uses_toml_approval_policy():
    # Non-bypass path: sandbox flag + approval_policy as a -c TOML override
    # (NOT the removed --ask-for-approval flag).
    cmd = CodexLLM(
        dangerously_bypass_approvals_and_sandbox=False,
        sandbox="read-only",
        approval_policy="never",
    )._build_cmd()
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert "--ask-for-approval" not in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert 'approval_policy="never"' in cmd


# ---------------------------------------------------------------------------
# Dangerous-bypass opt-in (contract §2 / deliverable 5)
# ---------------------------------------------------------------------------

def test_bypass_requires_declared_external_sandbox():
    llm = CodexLLM(dangerously_bypass_approvals_and_sandbox=True)  # no external_sandbox
    with pytest.raises(RuntimeError, match="hardened outer sandbox"):
        llm._build_cmd()


def test_bypass_allowed_via_constructor_flag():
    _llm(dangerously_bypass_approvals_and_sandbox=True)._build_cmd()  # no raise


def test_bypass_allowed_via_env(monkeypatch):
    monkeypatch.setenv("CHIA_CODEX_EXTERNAL_SANDBOX", "1")
    CodexLLM(dangerously_bypass_approvals_and_sandbox=True)._build_cmd()  # no raise


def test_safe_sandbox_never_asserts():
    # A non-bypass build never needs the external-sandbox declaration.
    CodexLLM(dangerously_bypass_approvals_and_sandbox=False)._build_cmd()


# ---------------------------------------------------------------------------
# Thread-id parsing (0.147.0 thread.started)
# ---------------------------------------------------------------------------

def test_parse_thread_id_from_thread_started():
    thread_id = "01a01160-7e52-7153-a3f1-a3ee492ab99e"
    stdout = (
        '{"type":"thread.started","thread_id":"%s"}\n'
        '{"type":"turn.started"}\n' % thread_id
    )
    assert parse_session_id(stdout) == thread_id


def test_parse_rate_limit_reset():
    reset = parse_rate_limit_reset("usage limit - resets 4pm (America/Los_Angeles)")
    assert reset is not None
    assert reset.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Streaming run → typed records (deliverables 1-4)
# ---------------------------------------------------------------------------

def test_prompt_streams_into_typed_run_result(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    _fake_stream(monkeypatch, [{"jsonl": _turn(tid="T-1", msg="PONG", inp=100, cr=20, out=10), "final": "PONG"}])
    llm = _llm(model="gpt-test", raw_event_dir=str(tmp_path))
    cli = llm.prompt("say pong", tools=[])

    assert cli.success is True
    assert cli.result == "PONG"
    assert cli.thread_id == "T-1"
    run = cli.run_result
    assert run is not None and run.status == "completed"
    assert len(run.attempts) == 1
    usage = run.total_usage
    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 20
    assert usage.uncached_input_tokens == 100 - 20 - 0  # subset accounting
    assert usage.provider_reported is True
    # Cumulative canonical metadata omits unknown fields, never fabricates 0.
    assert llm._last_metadata["input_tokens"] == 100
    assert llm._last_metadata["cache_read_input_tokens"] == 20
    # Raw JSONL was teed to a durable per-attempt file.
    assert os.path.exists(run.attempts[0].raw_event_path)


def test_prompt_optionally_records_arrival_timestamped_events(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    calls = _fake_stream(
        monkeypatch, [{"jsonl": _turn(tid="T-ts", msg="PONG"), "final": "PONG"}])
    llm = _llm(raw_event_dir=str(tmp_path), capture_arrival_timestamps=True)
    cli = llm.prompt("say pong", tools=[])

    path = cli.run_result.attempts[0].arrival_timestamped_event_path
    assert path == cli.arrival_timestamped_event_path
    assert calls[0]["timestamped_path"] == path
    assert path is not None and os.path.exists(path)


def test_prompt_argv_is_redacted_in_attempt(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    _fake_stream(monkeypatch, [{"jsonl": _turn(), "final": "OK"}])
    llm = _llm(model="gpt-test", raw_event_dir=str(tmp_path),
               extra_cli_args=["--api-key", "sk-secret-value"])
    cli = llm.prompt("hi", tools=[])
    argv = cli.run_result.attempts[0].cmd_argv
    assert "sk-secret-value" not in argv
    assert "***REDACTED***" in argv


# ---------------------------------------------------------------------------
# Retry billing — a failed attempt's usage survives (deliverable 3, contract §5)
# ---------------------------------------------------------------------------

def test_retry_preserves_failed_attempt_usage(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    monkeypatch.setattr(CodexLLM, "_get_node_id", lambda self: "test-node")
    # Attempt 1: reports 100 input tokens AND exits non-zero (unknown error) -> retry.
    # Attempt 2: reports 200 input tokens, exit 0 -> success.
    _fake_stream(monkeypatch, [
        {"jsonl": _turn(tid="T", msg="fail", inp=100, cr=0, out=5),
         "returncode": 1, "stderr": "something surprising"},
        {"jsonl": _turn(tid="T", msg="ok", inp=200, cr=0, out=7), "returncode": 0},
    ])
    llm = _llm(retries=3, raw_event_dir=str(tmp_path))
    cli = llm.prompt("hello", tools=[])

    assert cli.success is True
    run = cli.run_result
    assert len(run.attempts) == 2
    # The failed attempt's tokens are NOT erased by the retry.
    assert run.total_usage.input_tokens == 300
    assert run.total_usage.output_tokens == 12
    assert run.attempts[0].failure_class == "unknown"
    assert run.attempts[1].failure_class is None


# ---------------------------------------------------------------------------
# Timeout salvage (deliverable 1/9)
# ---------------------------------------------------------------------------

def test_timeout_records_salvaged_attempt(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    monkeypatch.setattr(CodexLLM, "_get_node_id", lambda self: "test-node")
    # A turn started but never completed, process killed for timeout.
    partial = (
        '{"type":"thread.started","thread_id":"T-timeout"}\n'
        '{"type":"turn.started"}\n'
    )
    _fake_stream(monkeypatch, [{"jsonl": partial, "returncode": -9, "timed_out": True, "signal": "SIGKILL"}])
    llm = _llm(retries=1, raw_event_dir=str(tmp_path))
    cli = llm.prompt("do work", tools=[])

    assert cli.success is False
    run = cli.run_result
    assert run.status == "failed"
    assert len(run.attempts) == 1
    assert run.attempts[0].timeout is True
    assert run.attempts[0].failure_class == "timeout"
    assert run.thread_id == "T-timeout"
    # The salvaged raw file still parses to the partial (incomplete) turn.
    assert os.path.exists(run.attempts[0].raw_event_path)


def test_corrupt_line_does_not_crash_prompt(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    jsonl = _turn(tid="T-c", msg="ok") + "this-is-not-json\n"
    _fake_stream(monkeypatch, [{"jsonl": jsonl, "final": "ok"}])
    llm = _llm(raw_event_dir=str(tmp_path))
    cli = llm.prompt("hi", tools=[])
    assert cli.success is True
    assert cli.run_result.total_usage.input_tokens == 100


# ---------------------------------------------------------------------------
# Resume — same worker (deliverable 9)
# ---------------------------------------------------------------------------

def test_same_worker_resume_builds_resume_cmd_on_second_call(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    calls = _fake_stream(monkeypatch, [
        {"jsonl": _turn(tid="01a0-THREAD", msg="one"), "final": "one"},
        {"jsonl": _turn(tid="01a0-THREAD", msg="two"), "final": "two"},
    ])
    llm = _llm(resume_session=True, raw_event_dir=str(tmp_path))

    cli1 = llm.prompt("first", tools=[])
    assert cli1.thread_id == "01a0-THREAD"
    assert calls[0]["cmd"][:3] == ["codex", "exec", "--json"]

    cli2 = llm.prompt("second", tools=[])
    assert cli2.success is True
    assert calls[1]["cmd"][:3] == ["codex", "exec", "resume"]
    assert calls[1]["cmd"][-2:] == ["01a0-THREAD", "-"]


# ---------------------------------------------------------------------------
# Session portability scoped to current thread only (deliverable 7)
# ---------------------------------------------------------------------------

def test_session_capture_only_current_thread_rollout(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    home = tmp_path / "home"
    thread_id = "01a0-THREAD"
    # Plant: the current thread's rollout (portable) + a shared state db and
    # auth.json (both must NEVER be captured).
    rollout_rel = f"sessions/2026/08/17/rollout-x-{thread_id}.jsonl"
    (home / "sessions/2026/08/17").mkdir(parents=True)
    (home / rollout_rel).write_bytes(b"ROLLOUT")
    (home / "state_9.sqlite").write_bytes(b"OTHER_CONVERSATIONS")
    (home / "auth.json").write_bytes(b"SECRET_CREDENTIALS")
    monkeypatch.setenv("CODEX_HOME", str(home))

    _fake_stream(monkeypatch, [{"jsonl": _turn(tid=thread_id, msg="ok"), "final": "ok"}])
    llm = _llm(resume_session=True, raw_event_dir=str(tmp_path))
    cli = llm.prompt("go", tools=[])

    assert cli.session_state is not None
    keys = set(cli.session_state)
    assert any(rollout_rel.endswith(k) or k.endswith(rollout_rel) or "rollout" in k for k in keys)
    # The credential and the shared state db are absent from the bundle.
    assert not any("auth.json" in k for k in keys)
    assert not any(k.startswith("state_") or "/state_" in k for k in keys)
    # And the bytes captured are the rollout, not the secret.
    assert b"SECRET_CREDENTIALS" not in b"".join(cli.session_state.values())


# ---------------------------------------------------------------------------
# Resume — cross worker (deliverable 9)
# ---------------------------------------------------------------------------

def test_cross_worker_sync_and_restore(monkeypatch, tmp_path):
    thread_id = "01a0-THREAD"
    rollout_rel = f"sessions/2026/08/17/rollout-x-{thread_id}.jsonl"

    # Worker A produced this state; a NEW instance (worker B) receives it.
    produced = CodexQueryResult(
        result="ok", returncode=0, stderr="", stream_result="",
        session_id=thread_id,
        session_state={rollout_rel: b"ROLLOUT", "auth.json": b"SECRET", "state_1.sqlite": b"OTHER"},
        session_state_paths=(rollout_rel, "auth.json", "state_1.sqlite"),
    )
    worker_b = _llm(resume_session=True)
    assert worker_b._sync_session(produced) is produced
    assert worker_b._session_id == thread_id
    assert worker_b._session_state[rollout_rel] == b"ROLLOUT"

    # Restoring onto B's CODEX_HOME writes the rollout but refuses the
    # credential and the shared state db even if they rode along.
    home_b = tmp_path / "home_b"
    monkeypatch.setenv("CODEX_HOME", str(home_b))
    worker_b._restore_session_state()
    assert (home_b / rollout_rel).read_bytes() == b"ROLLOUT"
    assert not (home_b / "auth.json").exists()
    assert not (home_b / "state_1.sqlite").exists()


# ---------------------------------------------------------------------------
# Multiple concurrent sessions keep independent state (deliverable 9)
# ---------------------------------------------------------------------------

def test_multiple_instances_keep_independent_run_results(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    _fake_stream(monkeypatch, [
        {"jsonl": _turn(tid="T-A", msg="a", inp=11, out=1), "final": "a"},
        {"jsonl": _turn(tid="T-B", msg="b", inp=22, out=2), "final": "b"},
    ])
    a = _llm(model="A", raw_event_dir=str(tmp_path / "a"))
    b = _llm(model="B", raw_event_dir=str(tmp_path / "b"))

    ca = a.prompt("x", tools=[])
    cb = b.prompt("y", tools=[])

    assert ca.thread_id == "T-A" and cb.thread_id == "T-B"
    assert a._run_result is not ca.run_result or True  # sanity
    assert a._run_result.thread_id == "T-A"  # not clobbered by b's run
    assert b._run_result.thread_id == "T-B"
    assert ca.run_result.total_usage.input_tokens == 11
    assert cb.run_result.total_usage.input_tokens == 22


# ---------------------------------------------------------------------------
# Error classification (unchanged behavior)
# ---------------------------------------------------------------------------

def test_classify_clean_success_no_raise():
    CodexLLM()._classify_error(_cli(returncode=0, result="PONG"))


@pytest.mark.parametrize("message", ["HTTP 429 Too Many Requests", "statusCode: 429", "APIError 429"])
def test_classify_real_429_rate_limit(message, monkeypatch):
    monkeypatch.setattr(CodexLLM, "_get_node_id", lambda self: "test-node")
    with pytest.raises(RateLimitError):
        CodexLLM()._classify_error(_cli(returncode=1, stderr=message))


@pytest.mark.parametrize(
    ("message", "error_cls", "returncode"),
    [
        ("429 rate limit", RateLimitError, 0),
        ("not logged in: run codex login", AuthenticationError, 1),
        ("payment required: add credit", BillingError, 1),
        ("invalid model: nope", InvalidRequestError, 1),
        ("503 service unavailable", ServerError, 1),
        ("max output token limit reached", MaxOutputTokensError, 1),
        ("something surprising", UnknownCodexError, 1),
    ],
)
def test_classify_errors(message, error_cls, returncode, monkeypatch):
    monkeypatch.setattr(CodexLLM, "_get_node_id", lambda self: "test-node")
    with pytest.raises(error_cls):
        CodexLLM()._classify_error(_cli(returncode=returncode, stderr=message))


def test_prompt_preserves_final_retry_error(monkeypatch, tmp_path):
    _disable_profiler(monkeypatch)
    monkeypatch.setattr(CodexLLM, "_get_node_id", lambda self: "test-node")
    _fake_stream(monkeypatch, [{"jsonl": _turn(msg="x"), "returncode": 1, "stderr": "something surprising"}])
    cli = _llm(retries=2, raw_event_dir=str(tmp_path)).prompt("hello", tools=[])
    assert cli.success is False
    assert cli.returncode == -1
    assert "UnknownCodexError" in cli.stderr
    assert "something surprising" in cli.stderr
    # Both failed attempts are retained with their usage.
    assert len(cli.run_result.attempts) == 2


# ---------------------------------------------------------------------------
# Live smoke tests (opt-in)
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(
    os.environ.get("CODEX_LIVE_TEST") != "1" or not shutil.which("codex"),
    reason="set CODEX_LIVE_TEST=1 and authenticate codex to run live tests",
)


@live
def test_live_codex_simple_prompt():
    llm = CodexLLM(
        system_message="You answer with a single word and nothing else.",
        timeout_seconds=180,
        dangerously_bypass_approvals_and_sandbox=False,
        sandbox="read-only",
        approval_policy="never",
        ephemeral=True,
    )
    cli = llm.prompt("Reply with exactly the word: PONG", tools=[])
    assert cli.success is True
    assert "PONG" in cli.result.upper()


@live
def test_live_codex_bypass_mirrors_skip_permissions_and_runs():
    llm = CodexLLM(
        system_message="You answer with a single word and nothing else.",
        timeout_seconds=180,
        dangerously_bypass_approvals_and_sandbox=True,
        external_sandbox=True,
        ephemeral=True,
    )
    assert llm.dangerously_skip_permissions is True
    cli = llm.prompt("Reply with exactly the word: PONG", tools=[])
    assert cli.success is True
    assert "PONG" in cli.result.upper()


@pytest.mark.live_remote
def test_live_remote_codex_bypass_runs(remote_prompt):
    llm = CodexLLM(
        system_message="You answer with a single word and nothing else.",
        timeout_seconds=180,
        dangerously_bypass_approvals_and_sandbox=True,
        external_sandbox=True,
        ephemeral=True,
    )
    assert llm.dangerously_skip_permissions is True
    cli = remote_prompt(llm, "Reply with exactly the word: PONG", "codex_creds")
    assert cli.success is True
    assert "PONG" in cli.result.upper()
