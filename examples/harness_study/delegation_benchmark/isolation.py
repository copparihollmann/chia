"""Task-local network and filesystem isolation for paid delegation arms.

The CLI sees one archived source snapshot and a single loopback model endpoint.
That endpoint crosses a Unix socket to a host-side usage relay. No other host
network connection or source checkout is visible inside the namespace.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from chia.base.sandbox import SandboxSpec, wrap_argv


_LAUNCHER = """
import os, socket, subprocess, sys, time
socket_path, port, *command = sys.argv[1:]
proxy = subprocess.Popen(
    ['/usr/bin/socat', f'TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork',
     f'UNIX-CONNECT:{socket_path}'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
for _ in range(100):
    if proxy.poll() is not None:
        raise SystemExit('local model bridge stopped')
    try:
        with socket.create_connection(('127.0.0.1', int(port)), timeout=0.1):
            break
    except OSError:
        time.sleep(0.02)
else:
    raise SystemExit('local model bridge did not start')
os.execvp(command[0], command)
"""


def executable_binds(program: str) -> list[Path]:
    """Bind a launcher plus its install tree, without the user's home/config."""
    found = shutil.which(program)
    if not found:
        raise FileNotFoundError(program)
    launcher = Path(found)
    target = launcher.resolve()
    binds = [launcher.parent]
    node_install = next((parent for parent in target.parents
                         if parent.parent.name == "node"), None)
    if node_install is not None:
        binds.append(node_install)
    else:
        binds.append(target.parent)
    return list(dict.fromkeys(binds))


def arm_spec(workspace: Path, socket_path: Path, program: str,
             *, preserve_env: tuple[str, ...] = ()) -> SandboxSpec:
    """The only visible task data is *workspace* and the single socket dir."""
    workspace = workspace.resolve()
    socket_path = socket_path.resolve()
    if not workspace.is_dir() or not socket_path.parent.is_dir():
        raise ValueError("workspace and bridge directory must already exist")
    sensitive = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AWS_")
    unset = [name for name in os.environ
             if any(word in name.upper() for word in sensitive)
             and name not in preserve_env]
    return SandboxSpec(
        workspace=workspace,
        allow=[socket_path.parent],
        extra_binds=executable_binds(program),
        tmpfs=["/tmp"],
        unsetenv=unset,
    )


def isolated_argv(command: list[str], spec: SandboxSpec,
                  socket_path: Path, local_port: int) -> list[str]:
    """Wrap one command in bwrap with no network except the UDS relay bridge."""
    if not command or not 1 <= local_port <= 65535:
        raise ValueError("command and a valid local port are required")
    launcher = ["/usr/bin/python3", "-c", _LAUNCHER,
                str(socket_path), str(local_port), *command]
    argv = wrap_argv(launcher, spec, "bwrap")
    argv.insert(1, "--unshare-net")
    return argv


class RelayBridge:
    """Host-side UDS to loopback TCP bridge; no provider secret enters the agent."""

    def __init__(self, *, task_tmp: Path, host_port: int):
        if not 1 <= host_port <= 65535:
            raise ValueError("valid host_port required")
        task_tmp = task_tmp.resolve()
        if not task_tmp.is_dir():
            raise ValueError("task temporary directory must exist")
        self.directory = Path(tempfile.mkdtemp(prefix="b", dir=task_tmp))
        self.socket_path = self.directory / "r.sock"
        if len(os.fsencode(self.socket_path)) >= 108:
            raise ValueError("Unix socket path exceeds Linux address length")
        self.host_port = host_port
        self.process: subprocess.Popen | None = None

    def __enter__(self):
        self.process = subprocess.Popen(
            ["socat", f"UNIX-LISTEN:{self.socket_path},fork",
             f"TCP:127.0.0.1:{self.host_port}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if self.process.poll() is not None:
                self.__exit__()
                raise RuntimeError("host relay bridge stopped during startup")
            if self.socket_path.exists():
                return self
            time.sleep(0.02)
        self.__exit__()
        raise RuntimeError("host relay bridge did not start")

    def __exit__(self, *_exc):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.directory.rmdir()
