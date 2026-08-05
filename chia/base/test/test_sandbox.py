"""Tests for per-call agent isolation (:mod:`chia.base.sandbox`).

Four layers:

* **Golden argv** — the order the builder emits, because order *is* the
  specification: deny masks must land after the allow binds or a broad allow
  re-exposes a denied sub-path.
* **Executable resolution** — the failure that costs a debugging round: a bare
  program name, or a symlinked launcher whose target directory is unbound, makes the
  command not-found inside the sandbox, and an empty response is indistinguishable
  from a model that declined.
* **Real enforcement** — actually runs ``bwrap`` and checks that a withheld file is
  *present but unreadable*, that a denied directory lists empty, and that a
  non-allowed path is gone. Skipped where bwrap cannot create a namespace.
* **Drift against aet** — chia's builder is written from the design aet documents;
  when aet's source is on this host, both are compared so the four independent
  copies of this design cannot silently diverge.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chia.base.sandbox import (
    DEFAULT_SYSTEM_DIRS,
    SANDBOX_BACKENDS,
    SandboxError,
    SandboxSpec,
    available_backends,
    binds_for_executable,
    bwrap_available,
    bwrap_prefix,
    docker_prefix,
    _kind,
    resolve_executable,
    spec_for_command,
    wrap_argv,
)

needs_bwrap = pytest.mark.skipif(
    not bwrap_available(),
    reason="bwrap missing, or unprivileged user namespaces are disabled",
)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _index(argv, *pair):
    """Index of the first occurrence of the consecutive pair *pair* in *argv*."""
    for i in range(len(argv) - len(pair) + 1):
        if tuple(argv[i:i + len(pair)]) == pair:
            return i
    raise AssertionError(f"{pair} not found in {argv}")


# ---------------------------------------------------------------------------
# Golden argv — order is the specification
# ---------------------------------------------------------------------------


def test_deny_is_masked_after_the_allow_binds(workspace, tmp_path):
    """The property the whole design rests on: deny wins. If the tmpfs mask were
    emitted before the ro-bind, a broad allow over a parent would re-expose the
    denied sub-path and the isolation claim would be false."""
    corpus = tmp_path / "corpus"
    answers = corpus / "answers"
    answers.mkdir(parents=True)

    argv = bwrap_prefix(SandboxSpec(workspace=workspace, allow=[corpus],
                                    deny=[answers]))

    assert _index(argv, "--tmpfs", str(answers)) > _index(argv, "--ro-bind",
                                                          str(corpus), str(corpus))


def test_rw_binds_land_between_the_allows_and_the_denies(workspace, tmp_path):
    """A writable state dir must override a broader read-only allow (a CLI's
    ~/.claude under an otherwise read-only $HOME) but must not survive a deny."""
    home = tmp_path / "home"
    state = home / ".claude"
    denied = home / "secrets"
    state.mkdir(parents=True)
    denied.mkdir()

    argv = bwrap_prefix(SandboxSpec(workspace=workspace, allow=[home],
                                    rw_binds=[state], deny=[denied]))

    ro_home = _index(argv, "--ro-bind", str(home), str(home))
    rw_state = _index(argv, "--bind", str(state), str(state))
    tmpfs_denied = _index(argv, "--tmpfs", str(denied))
    assert ro_home < rw_state < tmpfs_denied


def test_a_masked_file_is_overlaid_with_dev_null(workspace, tmp_path):
    golden = tmp_path / "golden.json"
    golden.write_text("{}")

    argv = bwrap_prefix(SandboxSpec(workspace=workspace, mask_files=[golden]))

    assert _index(argv, "--ro-bind", "/dev/null", str(golden)) > 0


def test_the_workspace_is_the_only_writable_bind(workspace, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    argv = bwrap_prefix(SandboxSpec(workspace=workspace, allow=[corpus]))

    writable = [argv[i + 1] for i, token in enumerate(argv) if token == "--bind"]
    assert writable == [str(workspace)]
    assert _index(argv, "--chdir", str(workspace)) > 0


def test_missing_paths_are_skipped_rather_than_bound(workspace, tmp_path):
    """bwrap fails hard on a bind whose source does not exist, so a spec listing a
    path that a given host happens not to have must not break every call."""
    argv = bwrap_prefix(SandboxSpec(workspace=workspace,
                                    allow=[tmp_path / "not-there"],
                                    deny=[tmp_path / "also-not-there"]))

    assert "not-there" not in " ".join(argv)
    assert "also-not-there" not in " ".join(argv)


def test_a_denied_file_is_not_tmpfs_masked(workspace, tmp_path):
    """tmpfs only applies to directories; a file listed under deny would make bwrap
    fail, so it is skipped and mask_files is the documented route."""
    a_file = tmp_path / "answer.txt"
    a_file.write_text("42")

    argv = bwrap_prefix(SandboxSpec(workspace=workspace, deny=[a_file]))

    assert "--tmpfs" not in argv[argv.index(str(a_file)) - 1:] if str(a_file) in argv \
        else str(a_file) not in argv


def test_unsetenv_and_lifecycle_flags(workspace):
    argv = bwrap_prefix(SandboxSpec(workspace=workspace,
                                    unsetenv=["CLAUDECODE", "CLAUDE_CODE_SSE_PORT"]))

    assert "--die-with-parent" in argv
    assert "--unshare-pid" in argv
    assert _index(argv, "--unsetenv", "CLAUDECODE") > 0
    assert _index(argv, "--unsetenv", "CLAUDE_CODE_SSE_PORT") > 0


def test_system_dirs_are_read_only(workspace):
    argv = bwrap_prefix(SandboxSpec(workspace=workspace))

    for directory in DEFAULT_SYSTEM_DIRS:
        if Path(directory).exists():
            assert _index(argv, "--ro-bind", directory, directory) > 0


# ---------------------------------------------------------------------------
# Executable resolution
# ---------------------------------------------------------------------------


def test_no_sandbox_leaves_the_argv_untouched():
    """An unsandboxed call must be byte-identical to what it was before this module
    existed: there is no alternate filesystem view to resolve against, subprocess does
    the same PATH lookup, and rewriting argv[0] would change what every existing
    caller logs and inspects."""
    assert wrap_argv(["sh", "-c", "true"], backend="none") == ["sh", "-c", "true"]


@needs_bwrap
def test_bwrap_resolves_the_program_to_an_absolute_host_path(workspace):
    """Inside bwrap a bare name resolves against the *sandbox's* view, where the
    binary's directory may not be bound — and a not-found binary reads as an empty
    response rather than as a broken sandbox."""
    argv = wrap_argv(["sh", "-c", "true"], spec_for_command(workspace, "sh"),
                     backend="bwrap")

    program = argv[argv.index("-c") - 1]
    assert program == resolve_executable("sh")
    assert Path(program).is_absolute()


@needs_bwrap
def test_an_unresolvable_program_is_left_alone(workspace):
    """So subprocess raises its own clear FileNotFoundError rather than this layer
    inventing a different failure."""
    argv = wrap_argv(["chia-no-such-program-9e3f"],
                     SandboxSpec(workspace=workspace), backend="bwrap")

    assert argv[-1] == "chia-no-such-program-9e3f"


def test_binds_for_executable_includes_a_symlink_target(tmp_path, monkeypatch):
    """The measured failure: an agent CLI is commonly ~/.local/bin/<name> symlinked
    into a versioned install tree. Binding only the launcher's directory leaves the
    link dangling and every call returns empty."""
    versions = tmp_path / "share" / "versions" / "2.1.222"
    versions.mkdir(parents=True)
    real = versions / "cli.js"
    real.write_text("#!/bin/sh\ntrue\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    launcher = bindir / "fake-agent"
    launcher.symlink_to(real)
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))

    binds = binds_for_executable("fake-agent")

    assert bindir in binds
    assert versions in binds


def test_binds_for_executable_is_empty_for_an_unknown_program():
    assert binds_for_executable("chia-no-such-program-9e3f") == []


def test_spec_for_command_binds_what_it_takes_to_run_the_command(workspace):
    spec = spec_for_command(workspace, "sh", deny=[])

    resolved = Path(resolve_executable("sh"))
    assert resolved.parent in spec.extra_binds


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_none_backend_is_always_available():
    assert "none" in available_backends()
    assert set(available_backends()) <= set(SANDBOX_BACKENDS)


def test_an_unknown_backend_is_rejected(workspace):
    with pytest.raises(ValueError) as exc:
        wrap_argv(["true"], SandboxSpec(workspace=workspace), backend="apptainer")

    assert "apptainer" in str(exc.value)


def test_a_sandbox_backend_without_a_spec_is_rejected():
    with pytest.raises(ValueError) as exc:
        wrap_argv(["true"], None, backend="bwrap")

    assert "SandboxSpec" in str(exc.value)


def test_an_unusable_backend_raises_rather_than_running_unisolated(workspace,
                                                                  monkeypatch):
    """Deliberately fatal. Falling back to no sandbox would produce numbers whose
    isolation claim is false, which is worse than a run that failed."""
    monkeypatch.setattr("chia.base.sandbox.bwrap_available", lambda: False)

    with pytest.raises(SandboxError) as exc:
        wrap_argv(["true"], SandboxSpec(workspace=workspace), backend="bwrap")

    assert "bwrap" in str(exc.value)


def test_docker_backend_expresses_the_same_intent(workspace, tmp_path):
    """docker has no per-file overlay, so a withheld file is a /dev/null bind mount
    and a denied directory an anonymous volume. Weaker than bwrap's view, and the
    argv is not comparable — but the outcomes are."""
    corpus = tmp_path / "corpus"
    answers = corpus / "answers"
    answers.mkdir(parents=True)
    golden = corpus / "golden.json"
    golden.write_text("{}")

    argv = docker_prefix(SandboxSpec(workspace=workspace, allow=[corpus],
                                     deny=[answers], mask_files=[golden],
                                     docker_image="chia-claude-code"))

    assert argv[:3] == ["docker", "run", "--rm"]
    assert f"{workspace}:{workspace}:rw" in argv
    assert f"{corpus}:{corpus}:ro" in argv
    assert str(answers) in argv                      # anonymous volume shadow
    assert f"/dev/null:{golden}:ro" in argv
    assert argv[-1] == "chia-claude-code"


def test_docker_ignores_host_toolchain_binds(workspace):
    """Measured the hard way: mounting the host's /usr/bin over an Alpine image's
    replaces musl-linked binaries with glibc-linked ones, and every command in the
    container dies with the dynamic loader's "no such file or directory". In the
    container model the tools come from the image, so extra_binds — a bwrap concept —
    is not translated."""
    spec = spec_for_command(workspace, "sh")
    assert spec.extra_binds, "spec_for_command should have found sh's directory"

    argv = docker_prefix(spec)

    for bind in spec.extra_binds:
        assert f"{bind}:{bind}:ro" not in argv


def test_docker_leaves_the_program_name_bare(workspace):
    """The command is resolved by the *image*, not the host: an absolute host path
    would either not exist in the container or point at a differently-linked binary."""
    argv = wrap_argv(["sh", "-c", "true"], SandboxSpec(workspace=workspace),
                     backend="docker")

    assert argv[-3:] == ["sh", "-c", "true"]


def test_bwrap_does_bind_the_toolchain(workspace):
    """The mirror of the docker case: bwrap starts from the host root, so without
    these binds the command genuinely is not there."""
    spec = spec_for_command(workspace, "sh")

    argv = bwrap_prefix(spec)

    for bind in spec.extra_binds:
        assert _index(argv, "--ro-bind", str(bind), str(bind)) > 0


# ---------------------------------------------------------------------------
# Real enforcement (needs bwrap)
# ---------------------------------------------------------------------------


def _run_in_sandbox(spec, script):
    argv = wrap_argv(["sh", "-c", script], spec, backend="bwrap")
    return subprocess.run(argv, capture_output=True, text=True, timeout=60)


@needs_bwrap
def test_a_withheld_file_is_present_but_unreadable(workspace, tmp_path):
    """The measured property, and the reason for /dev/null rather than deletion: an
    ENOENT would tell the agent *which* file was withheld, and that is itself
    information about the answer."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    golden = corpus / "golden.json"
    golden.write_text("THE ANSWER IS 42")

    spec = spec_for_command(workspace, "sh", allow=[corpus], mask_files=[golden])
    exists = _run_in_sandbox(spec, f"test -e {golden} && echo present")
    content = _run_in_sandbox(spec, f"cat {golden}")

    assert exists.stdout.strip() == "present"
    assert "42" not in content.stdout


@needs_bwrap
def test_a_denied_directory_exists_and_lists_empty(workspace, tmp_path):
    corpus = tmp_path / "corpus"
    answers = corpus / "answers"
    answers.mkdir(parents=True)
    (answers / "solution.txt").write_text("THE ANSWER IS 42")

    spec = spec_for_command(workspace, "sh", allow=[corpus], deny=[answers])
    listing = _run_in_sandbox(spec, f"ls -A {answers}; echo rc=$?")

    assert "solution.txt" not in listing.stdout
    assert "rc=0" in listing.stdout


@needs_bwrap
def test_deny_beats_a_broader_allow_in_practice(workspace, tmp_path):
    """Not just the argv order — the actual read fails."""
    corpus = tmp_path / "corpus"
    answers = corpus / "answers"
    answers.mkdir(parents=True)
    (answers / "solution.txt").write_text("THE ANSWER IS 42")
    (corpus / "inputs.txt").write_text("the question")

    spec = spec_for_command(workspace, "sh", allow=[corpus], deny=[answers])
    granted = _run_in_sandbox(spec, f"cat {corpus / 'inputs.txt'}")
    withheld = _run_in_sandbox(spec, f"cat {answers / 'solution.txt'}")

    assert "the question" in granted.stdout
    assert "42" not in withheld.stdout


@needs_bwrap
def test_a_path_that_was_never_allowed_is_simply_gone(workspace, tmp_path):
    """Deny-by-default: an unlisted sibling directory needs no explicit deny."""
    sibling = tmp_path / "someone-elses-run"
    sibling.mkdir()
    (sibling / "notes.txt").write_text("THE ANSWER IS 42")

    spec = spec_for_command(workspace, "sh")
    result = _run_in_sandbox(spec, f"cat {sibling / 'notes.txt'}")

    assert "42" not in result.stdout


@needs_bwrap
def test_the_workspace_is_writable_and_nothing_else_is(workspace, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "given.txt").write_text("x")

    spec = spec_for_command(workspace, "sh", allow=[corpus])
    wrote = _run_in_sandbox(spec, f"echo ok > {workspace / 'out.txt'} && echo wrote")
    blocked = _run_in_sandbox(spec, f"echo bad > {corpus / 'given.txt'} && echo wrote")

    assert wrote.stdout.strip() == "wrote"
    assert "wrote" not in blocked.stdout


@needs_bwrap
def test_unsetenv_actually_clears_the_variable(workspace, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    spec = spec_for_command(workspace, "sh", unsetenv=["CLAUDECODE"])

    result = _run_in_sandbox(spec, 'echo "[${CLAUDECODE-unset}]"')

    assert result.stdout.strip() == "[unset]"


# ---------------------------------------------------------------------------
# Drift against aet's builder
# ---------------------------------------------------------------------------


def _load_aet_sandbox():
    """aet's isolation.sandbox module, or ``None`` when it is not on this host.

    Loaded by path rather than imported as a dependency: aet is optional, and the
    point is only to compare two independent implementations of one documented
    design.
    """
    candidates = []
    try:
        import aet  # noqa: F401

        root = Path(aet.__file__).parent
        candidates.append(root / "isolation" / "sandbox.py")
    except Exception:
        pass
    env_root = os.environ.get("AET_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "src" / "aet" / "isolation" / "sandbox.py")
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_aet_sandbox_drift", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["_aet_sandbox_drift"] = module
            spec.loader.exec_module(module)
            return module
    return None


def test_the_builder_agrees_with_aets_where_the_designs_overlap(workspace, tmp_path):
    """This design exists four times over (aet, oscar-merlin, spec, and now chia).
    Comparing against aet's is how a divergence gets caught rather than discovered
    later as an isolation bug — the same drift check spec already carries.

    Only the shared subset is compared: chia's default ``system_dirs`` is wider than
    aet's, and chia resolves ``argv[0]``, so the specs are aligned explicitly rather
    than assuming the defaults match.
    """
    aet_sandbox = _load_aet_sandbox()
    if aet_sandbox is None:
        pytest.skip("aet's isolation/sandbox.py is not available on this host")

    corpus = tmp_path / "corpus"
    answers = corpus / "answers"
    answers.mkdir(parents=True)
    golden = corpus / "golden.json"
    golden.write_text("{}")
    state = tmp_path / "home" / ".claude"
    state.mkdir(parents=True)

    common = dict(
        allow=[corpus], rw_binds=[state], deny=[answers], mask_files=[golden],
        system_dirs=list(aet_sandbox.DEFAULT_SYSTEM_DIRS),
        tmpfs=["/tmp"], unsetenv=["CLAUDECODE"], dns=True,
        die_with_parent=True, unshare_pid=True,
    )
    ours = bwrap_prefix(SandboxSpec(workspace=workspace, **common))
    theirs = aet_sandbox.bwrap_argv(aet_sandbox.SandboxSpec(
        workspace=Path(workspace), **{k: (list(map(Path, v)) if k in
                                          ("allow", "rw_binds", "deny", "mask_files")
                                          else v)
                                      for k, v in common.items()}
    ))

    assert ours == theirs, (
        "chia's bwrap argv has drifted from aet's documented design:\n"
        f"  chia: {ours}\n  aet : {theirs}"
    )
