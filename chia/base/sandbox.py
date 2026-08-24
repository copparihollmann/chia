"""Per-call filesystem isolation for an agent process.

chia can already isolate a *worker*: a node type's ``docker:`` block puts every task
on that worker inside a long-lived container. What it cannot do is isolate a single
*call*. For AI-for-hardware research that is the granularity that matters, because the
thing that has to be withheld is not the toolchain — it is the answer.

An agent asked to generate a design, evaluated against a golden reference that sits on
the same filesystem, can read the reference. Denying it by tool name does not close
this: a deny-list names the tools it knows about, and a CLI's tool set changes between
versions. Downstream, a live agent enumerated its denied tools, found a shell-capable
one the list did not name, and used it. Only a filesystem view stopped it.

So the model here is an **allow-list filesystem view around one command**:

* ``workspace`` is bound read-write and is the only writable place.
* ``allow`` / ``extra_binds`` are bound read-only — granted inputs and the toolchain.
* ``rw_binds`` are bound read-write *after* the read-only binds, so an agent's state
  directory (a CLI's ``~/.claude``) stays writable under an otherwise read-only home.
* ``deny`` is masked with tmpfs **after** the allow binds, so deny wins: a broad allow
  cannot re-expose a denied sub-path.
* ``mask_files`` overlays individual files with ``/dev/null``.
* ``tmpfs`` blanks whole directories (``/tmp``, a project parent so siblings vanish).
* ``unsetenv`` clears variables inside the sandbox.

**A withheld file is present but unreadable, not absent.** ``/dev/null`` over a file
means ``test -e`` succeeds while ``cat`` yields nothing; a denied directory exists and
lists empty. Deleting instead would leak *which* files were withheld through ``ENOENT``,
and that is itself information about the answer.

Three backends, selected by name, because the right one depends on the host:

``bwrap``
    Bubblewrap. No daemon, no group membership, no image — usable on a shared login
    node where nobody will add you to ``docker``. Needs user namespaces.
``docker``
    One ``docker run --rm`` per call. For hosts with a daemon and no user namespaces.
    Slower to start, and the cost is measurable at grid scale.
``none``
    Explicit passthrough. Named so that "not isolated" is a recorded choice rather
    than the absence of one.

The design — allow-list view, deny after allow, per-file ``/dev/null`` overlay,
writable state dir under a read-only home — follows aet's ``aet.isolation.sandbox``
(Apache-2.0). chia is BSD-3, so this is written from the documented design rather than
copied, and :mod:`chia.base.test.test_sandbox` carries a drift test that loads aet's
module by path when present and asserts both builders agree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

PathLike = Union[str, os.PathLike]

#: The OS, bound read-only. A sandboxed agent still needs an interpreter and libc.
DEFAULT_SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt")

#: ``/etc/resolv.conf`` symlinks in here on systemd-resolved hosts, and bwrap does not
#: bind it by default. Without it a sandboxed CLI cannot resolve the API endpoint and
#: every call fails — which reads as a model failure, not an isolation change. A
#: network-less sandbox would be stronger, but an agent CLI *is* an API client.
DNS_DIR = "/run/systemd/resolve"

#: Backend names accepted by :func:`wrap_argv`.
SANDBOX_BACKENDS = ("bwrap", "docker", "none")


class SandboxError(RuntimeError):
    """A sandbox was requested but cannot be built on this host."""


@dataclass
class SandboxSpec:
    """What one sandboxed call may see.

    :param workspace: The only writable directory, and the command's cwd.
    :param allow: Read-only binds — granted inputs and in-tree tools.
    :param rw_binds: Read-write binds applied after the read-only ones, for agent
        state directories that must survive under an otherwise read-only parent.
    :param deny: Directories tmpfs-masked *after* the allow binds, so deny wins.
    :param mask_files: Individual files overlaid with ``/dev/null``.
    :param extra_binds: Read-only binds for toolchain paths outside the workspace.
    :param system_dirs: OS directories bound read-only.
    :param tmpfs: Directories blanked entirely.
    :param unsetenv: Environment variables cleared inside the sandbox.
    :param dns: Bind the resolver stub so name resolution works.
    :param unshare_pid: Give the command its own PID namespace.
    :param die_with_parent: Kill the sandbox when the parent exits, so an abandoned
        agent cannot outlive the run that launched it.
    :param docker_image: Image for the ``docker`` backend.
    :type workspace: PathLike
    :type allow: Sequence[PathLike]
    :type rw_binds: Sequence[PathLike]
    :type deny: Sequence[PathLike]
    :type mask_files: Sequence[PathLike]
    :type extra_binds: Sequence[PathLike]
    :type system_dirs: Sequence[str]
    :type tmpfs: Sequence[str]
    :type unsetenv: Sequence[str]
    :type dns: bool
    :type unshare_pid: bool
    :type die_with_parent: bool
    :type docker_image: str

    Nothing here is project-specific. Which directories hold answers and which hold
    tools is the caller's knowledge, not the framework's.
    """

    workspace: PathLike
    allow: Sequence[PathLike] = field(default_factory=list)
    rw_binds: Sequence[PathLike] = field(default_factory=list)
    deny: Sequence[PathLike] = field(default_factory=list)
    mask_files: Sequence[PathLike] = field(default_factory=list)
    extra_binds: Sequence[PathLike] = field(default_factory=list)
    system_dirs: Sequence[str] = field(default_factory=lambda: list(DEFAULT_SYSTEM_DIRS))
    tmpfs: Sequence[str] = field(default_factory=lambda: ["/tmp"])
    unsetenv: Sequence[str] = field(default_factory=list)
    dns: bool = True
    unshare_pid: bool = True
    die_with_parent: bool = True
    docker_image: str = "chia-claude-code"


# ---------------------------------------------------------------------------
# Path classification and executable resolution
# ---------------------------------------------------------------------------


def _kind(path: PathLike) -> str:
    """``"dir"``, ``"file"`` or ``"missing"``, without ever raising.

    A ``chmod 000`` parent makes ``stat()`` raise ``PermissionError``. Treating that
    as a directory means a locked answer surface is still masked, and the builder
    never crashes on the very paths it exists to hide.
    """
    candidate = Path(path)
    try:
        if candidate.is_dir():
            return "dir"
        if candidate.exists():
            return "file"
        return "missing"
    except PermissionError:
        return "dir"


def resolve_executable(program: str) -> Optional[str]:
    """Absolute path to *program*, or ``None`` when it is not on ``PATH``.

    :param program: A bare program name or a path.
    :type program: str
    :rtype: Optional[str]

    Backends build a bare argv (``["claude", ...]``) and rely on ``PATH`` lookup by
    the shell or by ``subprocess``. That stops working the moment the command is
    wrapped: the sandbox has its own filesystem view, and a bare name resolves inside
    it — where the binary's directory may not be bound at all.
    """
    if not program:
        return None
    found = shutil.which(program)
    if found:
        return found
    candidate = Path(program)
    return str(candidate) if candidate.exists() else None


def binds_for_executable(program: str) -> List[Path]:
    """Directories that must be bound for *program* to be runnable in a sandbox.

    :param program: A bare program name or a path.
    :type program: str
    :rtype: List[Path]

    Binding the directory a CLI lives in is not sufficient, and this is worth stating
    because it fails silently: an agent CLI is commonly a symlink from
    ``~/.local/bin`` into a versioned install tree. Bind only the launcher's directory
    and the link dangles — the command is not found, and every call returns an empty
    response, which looks exactly like a model that refused to answer.

    Returns the launcher's directory *and* the resolved target's, existing paths only,
    so the result can be handed straight to :attr:`SandboxSpec.extra_binds`.
    """
    found = resolve_executable(program)
    if not found:
        return []
    launcher = Path(found)
    out = [launcher.parent]
    try:
        target = launcher.resolve()
    except OSError:
        return [p for p in out if p.exists()]
    if target != launcher:
        out.append(target if target.is_dir() else target.parent)
    return [p for p in out if p.exists()]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def bwrap_available() -> bool:
    """Whether ``bwrap`` is on ``PATH`` *and* can actually create a namespace.

    The second half matters: bwrap installs fine on hosts where unprivileged user
    namespaces are disabled, so a ``which`` check alone reports a sandbox that fails
    at first use.

    The probe binds the same system directories a real spec would and invokes the test
    binary by **absolute** path. A bare name here would be looked up inside the
    sandbox, where ``/bin`` may not be bound — reporting "bwrap unusable" for a host
    where bwrap works perfectly, which is the same class of mistake
    :func:`resolve_executable` exists to prevent.
    """
    if shutil.which("bwrap") is None:
        return False
    probe_binary = shutil.which("true")
    if probe_binary is None:
        return False
    argv = ["bwrap"]
    for directory in DEFAULT_SYSTEM_DIRS:
        if _kind(directory) != "missing":
            argv += ["--ro-bind", str(directory), str(directory)]
    argv += ["--dev", "/dev", probe_binary]
    try:
        return subprocess.run(argv, capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


def docker_available() -> bool:
    """Whether a reachable docker daemon exists (``docker info`` succeeds)."""
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        return probe.returncode == 0
    except Exception:
        return False


def available_backends() -> List[str]:
    """The backends usable on this host, most isolating first.

    :rtype: List[str]

    ``"none"`` is always present: it is a deliberate choice, not a fallback.
    """
    out = []
    if bwrap_available():
        out.append("bwrap")
    if docker_available():
        out.append("docker")
    out.append("none")
    return out


# ---------------------------------------------------------------------------
# The three backends
# ---------------------------------------------------------------------------


def bwrap_prefix(spec: SandboxSpec) -> List[str]:
    """The ``bwrap`` argv prefix for *spec*, up to but excluding the command.

    :param spec: What the call may see.
    :type spec: SandboxSpec
    :rtype: List[str]

    Order is the specification, not an implementation detail. System binds and tmpfs
    blanks first; then the workspace and the read-only allows; then read-write binds
    (so a writable state dir overrides a broader read-only allow); then the deny masks
    — **after** the allows, which is what makes deny win; then the per-file overlays;
    then the env unsets. Reordering any of these changes what the agent can read.
    """
    workspace = str(spec.workspace)
    parts: List[str] = ["bwrap"]
    if spec.die_with_parent:
        parts += ["--die-with-parent"]
    if spec.unshare_pid:
        parts += ["--unshare-pid"]
    for directory in spec.system_dirs:
        if _kind(directory) != "missing":
            parts += ["--ro-bind", str(directory), str(directory)]
    for directory in spec.tmpfs:
        parts += ["--tmpfs", str(directory)]
    parts += ["--bind", workspace, workspace]
    parts += ["--proc", "/proc", "--dev", "/dev", "--chdir", workspace]
    if spec.dns and _kind(DNS_DIR) != "missing":
        parts += ["--ro-bind", DNS_DIR, DNS_DIR]
    for path in list(spec.allow) + list(spec.extra_binds):
        if _kind(path) != "missing":
            parts += ["--ro-bind", str(path), str(path)]
    for path in spec.rw_binds:
        if _kind(path) != "missing":
            parts += ["--bind", str(path), str(path)]
    for path in spec.deny:
        if _kind(path) == "dir":
            parts += ["--tmpfs", str(path)]
    for path in spec.mask_files:
        if _kind(path) != "missing":
            # Present-but-empty rather than deleted: an ENOENT would tell the agent
            # that this particular file was withheld.
            parts += ["--ro-bind", "/dev/null", str(path)]
    for name in spec.unsetenv:
        parts += ["--unsetenv", str(name)]
    return parts


def docker_prefix(spec: SandboxSpec) -> List[str]:
    """The ``docker run`` argv prefix for *spec*.

    :param spec: What the call may see.
    :type spec: SandboxSpec
    :rtype: List[str]

    A weaker statement than bwrap's, and deliberately so: docker has no per-file
    overlay, so :attr:`SandboxSpec.mask_files` is expressed by bind-mounting
    ``/dev/null`` over each file, and :attr:`SandboxSpec.deny` by mounting an
    anonymous volume that shadows the path. Both hold, but the container starts from
    an image rather than the host root, so what is *absent* differs from bwrap's view
    and the two backends are not argv-comparable — only outcome-comparable.

    **:attr:`SandboxSpec.extra_binds` is deliberately ignored here.** Those are host
    toolchain directories, which belong to bwrap's host-root model: in a container the
    tools come from the image. Mounting them anyway is not merely redundant, it breaks
    the container — binding the host's ``/usr/bin`` over an Alpine image's replaces
    musl-linked binaries with glibc-linked ones and every command dies with
    ``no such file or directory`` from the dynamic loader. So the docker backend
    requires an image that already contains the agent CLI, which is what chia's
    ``chia-claude-code`` image is for.
    """
    workspace = str(spec.workspace)
    parts: List[str] = ["docker", "run", "--rm", "-i",
                        "-w", workspace,
                        "-v", f"{workspace}:{workspace}:rw"]
    for path in spec.allow:
        if _kind(path) != "missing":
            parts += ["-v", f"{path}:{path}:ro"]
    for path in spec.rw_binds:
        if _kind(path) != "missing":
            parts += ["-v", f"{path}:{path}:rw"]
    for path in spec.deny:
        if _kind(path) == "dir":
            # An anonymous volume shadows the path with an empty directory: the path
            # exists and lists empty, matching bwrap's tmpfs mask.
            parts += ["-v", str(path)]
    for path in spec.mask_files:
        if _kind(path) != "missing":
            parts += ["-v", f"/dev/null:{path}:ro"]
    for directory in spec.tmpfs:
        parts += ["--tmpfs", str(directory)]
    for name in spec.unsetenv:
        # docker has no --unsetenv; an empty assignment overrides an inherited value.
        parts += ["-e", f"{name}="]
    parts += [spec.docker_image]
    return parts


def wrap_argv(
    argv: Sequence[str],
    spec: Optional[SandboxSpec] = None,
    backend: str = "none",
) -> List[str]:
    """Wrap *argv* so it runs under *backend*, resolving ``argv[0]`` either way.

    :param argv: The command to run.
    :param spec: What the call may see; required for every backend but ``"none"``.
    :param backend: One of :data:`SANDBOX_BACKENDS`.
    :type argv: Sequence[str]
    :type spec: Optional[SandboxSpec]
    :type backend: str
    :rtype: List[str]
    :raises ValueError: Unknown *backend*, or a sandbox backend with no *spec*.
    :raises SandboxError: The requested backend is not usable on this host.

    Under ``"bwrap"``, ``argv[0]`` is resolved to an absolute host path: a bare name
    would be looked up inside the sandbox's own filesystem view, where the binary's
    directory may not be bound at all — and a not-found binary surfaces as an empty
    response, indistinguishable from a model that declined to answer. When the name
    cannot be resolved it is left alone, so ``subprocess`` raises its own clear
    ``FileNotFoundError`` rather than this function inventing a different failure.

    Under ``"docker"`` the name is left **bare** because the command comes from the
    image: a host path would either not exist in the container or point at a binary
    built against a different libc. Under ``"none"`` it is left bare too — there is no
    alternate filesystem view to resolve against, ``subprocess`` does the same ``PATH``
    lookup, and rewriting the argv of an unsandboxed call would change what every
    existing caller logs and inspects for no benefit.
    """
    if backend not in SANDBOX_BACKENDS:
        raise ValueError(
            f"unknown sandbox backend {backend!r}; expected one of {SANDBOX_BACKENDS}"
        )
    argv = list(argv)
    if backend == "none":
        return argv
    if spec is None:
        raise ValueError(f"sandbox backend {backend!r} needs a SandboxSpec")

    if backend == "bwrap":
        if argv:
            resolved = resolve_executable(argv[0])
            if resolved:
                argv[0] = resolved
        if not bwrap_available():
            raise SandboxError(
                "bwrap is not usable on this host (missing, or unprivileged user "
                "namespaces are disabled). Available: "
                f"{', '.join(available_backends())}."
            )
        return bwrap_prefix(spec) + argv
    if not docker_available():
        raise SandboxError(
            "no reachable docker daemon. Available: "
            f"{', '.join(available_backends())}."
        )
    return docker_prefix(spec) + argv


def spec_for_command(
    workspace: PathLike,
    program: str,
    *,
    deny: Iterable[PathLike] = (),
    mask_files: Iterable[PathLike] = (),
    allow: Iterable[PathLike] = (),
    **kwargs,
) -> SandboxSpec:
    """A :class:`SandboxSpec` that can actually run *program*.

    :param workspace: The writable directory and cwd.
    :param program: The command whose binary must be reachable inside the sandbox.
    :param deny: Directories to mask.
    :param mask_files: Files to withhold.
    :param allow: Extra read-only binds.
    :type workspace: PathLike
    :type program: str
    :type deny: Iterable[PathLike]
    :type mask_files: Iterable[PathLike]
    :type allow: Iterable[PathLike]
    :rtype: SandboxSpec

    A convenience over constructing the spec by hand, because forgetting
    :func:`binds_for_executable` is the mistake that costs a debugging round: the
    sandbox builds, the command is not found, and the run looks like a model failure.
    Any other :class:`SandboxSpec` field may be passed through as a keyword.
    """
    return SandboxSpec(
        workspace=workspace,
        allow=list(allow),
        deny=list(deny),
        mask_files=list(mask_files),
        extra_binds=list(kwargs.pop("extra_binds", [])) + binds_for_executable(program),
        **kwargs,
    )
