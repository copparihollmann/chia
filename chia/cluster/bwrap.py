"""Persistent bubblewrap execution context for logical CHIA workers.

Unlike a per-command sandbox, a Ray worker needs a namespace that survives the
SSH session which created it.  :class:`BubblewrapManager` starts a tiny command
supervisor as PID 1 in a detached bubblewrap PID/IPC/UTS namespace.  Later SSH
sessions submit scripts through a private host directory bind-mounted at
``/run/chia-bwrap``.  Ray and every process it launches therefore remain in one
namespace until ``chia down`` terminates the recorded process group.

The state file records both the outer PID and Linux process start time.  Every
operation validates both values, preventing a stale PID file from killing or
reusing an unrelated process after PID recycling.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import time
import uuid
from dataclasses import replace

from chia.cluster.config import BwrapConfig
from chia.cluster.log import get_logger
from chia.cluster.ssh import SSHClient

logger = get_logger("bwrap")

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_CONTROL_DIR = "/run/chia-bwrap"


class BubblewrapManager:
    """Manage one long-lived rootless bubblewrap worker namespace."""

    def __init__(self, ssh: SSHClient, config: BwrapConfig):
        if not _NAME_RE.fullmatch(config.worker_name):
            raise ValueError(f"unsafe bubblewrap worker name: {config.worker_name!r}")
        self.ssh = ssh
        self.config = config
        self.state_dir = f"{config.state_dir.rstrip('/')}/{config.worker_name}"
        self._oci_environment: dict[str, str] = {}
        self._oci_working_dir: str | None = None

    @staticmethod
    def for_worker(config: BwrapConfig, worker_index: int) -> BwrapConfig:
        """Return a worker-specific config, mirroring Docker name suffixing."""
        return replace(
            config,
            worker_name=f"{config.worker_name}-{worker_index}",
        )

    def _q(self, value: str) -> str:
        return shlex.quote(value)

    def _state_path(self, name: str) -> str:
        return f"{self.state_dir}/{name}"

    def _pid_identity_test(self) -> str:
        pid = self._q(self._state_path("pid"))
        start = self._q(self._state_path("start_time"))
        return (
            f"test -s {pid} && test -s {start} && "
            f"p=$(cat {pid}) && expected=$(cat {start}) && "
            "test -r /proc/$p/stat && "
            "actual=$(awk '{print $22}' /proc/$p/stat) && "
            "test \"$actual\" = \"$expected\" && kill -0 \"$p\" 2>/dev/null"
        )

    def _active_test(self) -> str:
        ready = self._q(self._state_path("ready"))
        return f"{self._pid_identity_test()} && test -s {ready}"

    def _check_binary(self, binary: str, purpose: str) -> None:
        result = self.ssh.run(
            f"command -v {self._q(binary)}", check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"[{self.ssh.ip}] {purpose} binary {binary!r} was not found "
                "in the login-shell PATH")

    def _inspect_image(self) -> None:
        image = self.config.image
        if not image:
            return
        engine = self.config.engine
        self._check_binary(engine, "OCI engine")
        if self.config.pull_before_prepare:
            logger.info(f"[{self.ssh.ip}] Pulling OCI image {image}")
            self.ssh.run(
                f"{self._q(engine)} pull {self._q(image)}",
                timeout=self.config.pull_timeout,
            )
        result = self.ssh.run(
            f"{self._q(engine)} image inspect {self._q(image)}")
        try:
            records = json.loads(result.stdout)
            record = records[0]
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"[{self.ssh.ip}] could not parse {engine} image metadata for "
                f"{image!r}") from exc

        identities = [record.get("Id", ""), *(record.get("RepoDigests") or [])]
        expected = self.config.image_digest
        if expected and expected not in identities and not any(
            identity.endswith(f"@{expected}") for identity in identities
        ):
            # Permit the common sha256:<hex> spelling when RepoDigests include
            # repository@sha256:<hex>.
            raise RuntimeError(
                f"[{self.ssh.ip}] image {image!r} resolved to {identities}, "
                f"not pinned digest {expected!r}")

        image_cfg = record.get("Config") or {}
        for entry in image_cfg.get("Env") or []:
            if "=" in entry:
                key, value = entry.split("=", 1)
                self._oci_environment[key] = value
        self._oci_working_dir = image_cfg.get("WorkingDir") or None
        self._resolved_identities = identities

    def _prepare_rootfs(self) -> None:
        rootfs = self.config.rootfs.rstrip("/") or "/"
        marker = f"{rootfs}/.chia-oci-rootfs.json"
        exists = self.ssh.run(
            f"test -d {self._q(rootfs)} && test -x {self._q(rootfs + '/bin/bash')}",
            check=False,
        )
        if exists.returncode == 0:
            if self.config.image:
                metadata = self.ssh.run(
                    f"test -f {self._q(marker)} && cat {self._q(marker)}",
                    check=False,
                )
                if metadata.returncode != 0:
                    raise RuntimeError(
                        f"[{self.ssh.ip}] rootfs {rootfs!r} exists but was not "
                        "created by CHIA; refusing to overwrite it")
                try:
                    saved = json.loads(metadata.stdout)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"[{self.ssh.ip}] invalid rootfs metadata at {marker}") from exc
                if saved.get("image") != self.config.image:
                    raise RuntimeError(
                        f"[{self.ssh.ip}] rootfs {rootfs!r} belongs to image "
                        f"{saved.get('image')!r}, not {self.config.image!r}")
                saved_ids = saved.get("resolved_identities") or []
                if not set(saved_ids) & set(self._resolved_identities):
                    raise RuntimeError(
                        f"[{self.ssh.ip}] rootfs {rootfs!r} was exported from "
                        f"{saved_ids}, but {self.config.image!r} now resolves "
                        f"to {self._resolved_identities}; use a new rootfs path")
                if self.config.image_digest:
                    expected = self.config.image_digest
                    if expected not in saved_ids and not any(
                        identity.endswith(f"@{expected}") for identity in saved_ids
                    ):
                        raise RuntimeError(
                            f"[{self.ssh.ip}] rootfs {rootfs!r} does not match "
                            f"pinned digest {expected!r}")
            return

        if not self.config.image:
            raise RuntimeError(
                f"[{self.ssh.ip}] bubblewrap rootfs {rootfs!r} is missing or "
                "does not contain executable /bin/bash; set bwrap.image to "
                "export an OCI image automatically")

        metadata = {
            "schema": 1,
            "image": self.config.image,
            "resolved_identities": self._resolved_identities,
            "environment": self._oci_environment,
            "working_dir": self._oci_working_dir,
        }
        metadata_b64 = base64.b64encode(
            json.dumps(metadata, sort_keys=True).encode()).decode()
        engine = self._q(self.config.engine)
        image = self._q(self.config.image)
        root = self._q(rootfs)
        marker_q = self._q(marker)
        script = [
            "set -e",
            f"root={root}",
            'parent=$(dirname "$root")',
            'tmp="${root}.chia-tmp-$$"',
            'test ! -e "$root" || { echo "rootfs path exists but is unusable: $root" >&2; exit 1; }',
            'mkdir -p "$parent" "$tmp"',
            f"cid=$({engine} create {image})",
            (f"trap '{self.config.engine} rm -f \"$cid\" >/dev/null 2>&1 || true; "
             "rm -rf \"$tmp\"' EXIT"),
            f"{engine} export \"$cid\" | tar -xpf - -C \"$tmp\"",
            f"printf %s {self._q(metadata_b64)} | base64 -d > \"$tmp/.chia-oci-rootfs.json\"",
            f"{engine} rm -f \"$cid\" >/dev/null",
            "cid=",
            'mv "$tmp" "$root"',
            "trap - EXIT",
            f"test -x {self._q(rootfs + '/bin/bash')} && test -f {marker_q}",
        ]
        logger.info(
            f"[{self.ssh.ip}] Exporting {self.config.image} to rootfs {rootfs}")
        self.ssh.run_script(script, timeout=max(self.config.pull_timeout, 600))

    def _supervisor_script(self) -> str:
        return """set -u
control=/run/chia-bwrap
printf '%s\n' "$CHIA_BWRAP_IDENTITY" > "$control/ready"
trap 'exit 0' TERM INT HUP
while :; do
  found=0
  for request in "$control"/request-*.sh; do
    [ -e "$request" ] || continue
    found=1
    id=${request##*/request-}
    id=${id%.sh}
    running="$control/running-$id.sh"
    mv "$request" "$running" || continue
    /bin/bash --login "$running" > "$control/stdout-$id.tmp" 2> "$control/stderr-$id.tmp"
    rc=$?
    mv "$control/stdout-$id.tmp" "$control/stdout-$id"
    mv "$control/stderr-$id.tmp" "$control/stderr-$id"
    printf '%s\n' "$rc" > "$control/rc-$id.tmp"
    mv "$control/rc-$id.tmp" "$control/rc-$id"
    rm -f "$running"
  done
  [ "$found" -eq 1 ] || sleep 0.05
done
"""

    def _bwrap_argv(self) -> list[str]:
        cfg = self.config
        argv = [
            cfg.binary,
            # Never degrade to a host user namespace: a reported bwrap worker
            # must have the isolation it claims or setup fails.
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--hostname", cfg.hostname or cfg.worker_name,
            "--ro-bind", cfg.rootfs, "/",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/run",
            "--bind", self.state_dir, _CONTROL_DIR,
        ]
        for path in cfg.tmpfs:
            if path in ("/run", "/dev"):
                continue
            argv.extend(["--tmpfs", path])
        for source, destination in cfg.read_only_binds.items():
            argv.extend(["--ro-bind", source, destination])
        for source, destination in cfg.read_write_binds.items():
            argv.extend(["--bind", source, destination])

        # OCI environments are the baseline, explicit config wins, and only
        # named host variables are inherited.  This avoids accidentally
        # exposing credentials from the worker host.
        environment = dict(self._oci_environment)
        environment.update(cfg.environment)
        for key in cfg.unset_environment:
            environment.pop(key, None)
        argv.append("--clearenv")
        for key, value in environment.items():
            argv.extend(["--setenv", key, value])
        # Host allowlisted values are appended by _launch_command because their
        # values must expand on the remote host, not on the CHIA driver.
        argv.extend(["--setenv", "CHIA_BWRAP_IDENTITY", cfg.worker_name])
        working_dir = cfg.working_dir or self._oci_working_dir or "/"
        argv.extend(["--chdir", working_dir])
        argv.extend(["/bin/bash", "--login", "-c", self._supervisor_script()])
        return argv

    def _launch_command(self) -> str:
        argv = self._bwrap_argv()
        # Split immediately before /bin/bash to add remotely-expanded allowlist
        # values without allowing shell splitting of those values.
        command_index = argv.index("/bin/bash")
        prefix = shlex.join(argv[:command_index])
        suffix = shlex.join(argv[command_index:])
        inherited = ""
        for key in self.config.environment_allowlist:
            if key in self.config.unset_environment:
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid environment variable name: {key!r}")
            inherited += f' --setenv {shlex.quote(key)} "${{{key}-}}"'
        return f"{prefix}{inherited} {suffix}"

    def setup_worker(self) -> None:
        """Prepare the rootfs and start (or reuse) the worker supervisor."""
        self._check_binary(self.config.binary, "bubblewrap")
        self._inspect_image()
        self._prepare_rootfs()
        self.ssh.run(
            f"mkdir -p {self._q(self.state_dir)} && "
            f"chmod 700 {self._q(self.state_dir)}")
        active = self.ssh.run(self._active_test(), check=False)
        if active.returncode == 0:
            logger.info(
                f"[{self.ssh.ip}] bubblewrap worker "
                f"{self.config.worker_name!r} is already running")
            return

        # A previous setup can die after bwrap starts but before it creates the
        # ready marker. Its PID/start-time pair is still safe to identify, so
        # terminate that partial namespace before reusing the control path.
        if self.ssh.run(self._pid_identity_test(), check=False).returncode == 0:
            self.stop_worker()

        # Clear only transient control files.  Logs and OCI metadata remain as
        # experiment evidence across a stale-process recovery.
        state = self._q(self.state_dir)
        launch = self._launch_command()
        log = self._q(self._state_path("worker.log"))
        script = [
            (f"rm -f {state}/pid {state}/start_time {state}/ready "
             f"{state}/request-* {state}/running-* {state}/stdout-* "
             f"{state}/stderr-* {state}/rc-*"),
            f"nohup setsid {launch} >> {log} 2>&1 < /dev/null &",
            "pid=$!",
            f"printf '%s\n' \"$pid\" > {self._q(self._state_path('pid'))}",
            "start=$(awk '{print $22}' /proc/$pid/stat)",
            f"printf '%s\n' \"$start\" > {self._q(self._state_path('start_time'))}",
            f"deadline=$((SECONDS + {self.config.start_timeout}))",
            f"until {self._active_test()}; do",
            "  if ! kill -0 \"$pid\" 2>/dev/null; then",
            f"    tail -n 100 {log} >&2 || true",
            "    exit 1",
            "  fi",
            '  test "$SECONDS" -lt "$deadline" || { echo "bubblewrap supervisor readiness timeout" >&2; exit 1; }',
            "  sleep 0.1",
            "done",
        ]
        logger.info(
            f"[{self.ssh.ip}] Starting bubblewrap worker "
            f"{self.config.worker_name!r}")
        try:
            self.ssh.run_script(
                script, timeout=self.config.start_timeout + 10)
            if self.config.run_setup_commands:
                self.exec_script(self.config.run_setup_commands)
        except Exception:
            self.stop_worker()
            raise

    def exec_script(
        self,
        commands: list[str],
        timeout: int | None = None,
        check: bool = True,
    ):
        """Execute commands inside the persistent namespace as one login shell."""
        commands = [command for command in commands if command.strip()]
        if not commands:
            return None
        timeout = timeout or self.config.command_timeout
        request_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
        payload = base64.b64encode(
            ("set -e\n" + "\n".join(commands) + "\n").encode()).decode()
        request = self._q(self._state_path(f"request-{request_id}.sh"))
        temp = self._q(self._state_path(f"request-{request_id}.tmp"))
        stdout = self._q(self._state_path(f"stdout-{request_id}"))
        stderr = self._q(self._state_path(f"stderr-{request_id}"))
        rcfile = self._q(self._state_path(f"rc-{request_id}"))
        command = (
            f"{self._active_test()} || {{ echo 'bubblewrap worker is not active' >&2; exit 125; }}; "
            f"rm -f {stdout} {stderr} {rcfile} {temp}; "
            f"printf %s {self._q(payload)} | base64 -d > {temp}; "
            f"chmod 600 {temp}; mv {temp} {request}; "
            f"deadline=$((SECONDS + {timeout})); "
            f"while test ! -f {rcfile}; do "
            f"{self._active_test()} || {{ echo 'bubblewrap worker exited' >&2; exit 125; }}; "
            f"test \"$SECONDS\" -lt \"$deadline\" || {{ echo 'bubblewrap command timeout' >&2; exit 124; }}; "
            "sleep 0.05; done; "
            f"cat {stdout}; cat {stderr} >&2; rc=$(cat {rcfile}); "
            f"rm -f {stdout} {stderr} {rcfile}; exit \"$rc\""
        )
        return self.ssh.run(command, timeout=timeout + 10, check=check)

    def stop_worker(self) -> None:
        """Stop the namespace process group if its PID identity still matches."""
        if self.ssh.run(self._pid_identity_test(), check=False).returncode != 0:
            logger.info(
                f"[{self.ssh.ip}] bubblewrap worker "
                f"{self.config.worker_name!r} is already stopped")
            return
        pid_path = self._q(self._state_path("pid"))
        state = self._q(self.state_dir)
        command = (
            f"{self._pid_identity_test()} || exit 0; p=$(cat {pid_path}); "
            "kill -TERM -- \"-$p\" 2>/dev/null || kill -TERM \"$p\" 2>/dev/null || true; "
            "i=0; while kill -0 \"$p\" 2>/dev/null && test \"$i\" -lt 50; "
            "do sleep 0.1; i=$((i+1)); done; "
            "if kill -0 \"$p\" 2>/dev/null; then "
            f"  {self._pid_identity_test()} && "
            "  (kill -KILL -- \"-$p\" 2>/dev/null || kill -KILL \"$p\" 2>/dev/null || true); "
            "fi; "
            f"rm -f {state}/pid {state}/start_time {state}/ready "
            f"{state}/request-* {state}/running-*"
        )
        logger.info(
            f"[{self.ssh.ip}] Stopping bubblewrap worker "
            f"{self.config.worker_name!r}")
        self.ssh.run(command, timeout=15, check=False)
