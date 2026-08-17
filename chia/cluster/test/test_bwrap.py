from __future__ import annotations

import base64
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from chia.cluster.bwrap import BubblewrapManager
from chia.cluster.config import BwrapConfig, ConfigError, NodeAssignment, build_config
from chia.cluster import node_setup


def _raw(node: dict, **top_level) -> dict:
    return {
        "cluster_name": "bwrap-test",
        "provider": {"head_ip": "host-a"},
        "auth": {"ssh_user": "tester"},
        "available_node_types": {
            "worker": {
                "resources": {"test": 1},
                "num_workers": 1,
                "compatible_ips": ["host-a"],
                **node,
            },
        },
        **top_level,
    }


def test_build_config_parses_bwrap() -> None:
    config = build_config(_raw({
        "bwrap": {
            "rootfs": "/srv/chia/rootfs",
            "worker_name": "circt-worker",
            "image": "example/chia@sha256:deadbeef",
            "image_digest": "sha256:deadbeef",
            "read_only_binds": {"/opt/tools": "/opt/tools"},
            "read_write_binds": {"/srv/work": "/workspace"},
            "environment_allowlist": ["SSH_AUTH_SOCK"],
        },
    }))

    bwrap = config.node_types["worker"].bwrap
    assert bwrap is not None
    assert bwrap.rootfs == "/srv/chia/rootfs"
    assert bwrap.worker_name == "circt-worker"
    assert bwrap.read_write_binds == {"/srv/work": "/workspace"}
    assert config.global_bwrap is None


@pytest.mark.parametrize(
    "raw",
    [
        _raw({
            "docker": {"image": "image"},
            "bwrap": {"rootfs": "/rootfs"},
        }),
        _raw({}, docker={"image": "image"}, bwrap={"rootfs": "/rootfs"}),
    ],
)
def test_docker_and_bwrap_are_mutually_exclusive(raw: dict) -> None:
    with pytest.raises(ConfigError, match="mutually exclusive"):
        build_config(raw)


@pytest.mark.parametrize(
    ("node,top_level,expected"),
    [
        ({"bwrap": {"rootfs": "/node-root"}},
         {"docker": {"image": "global-image"}}, "bwrap"),
        ({"docker": {"image": "node-image"}},
         {"bwrap": {"rootfs": "/global-root"}}, "docker"),
        ({}, {"docker": {"image": "global-image"}}, "docker"),
        ({}, {"bwrap": {"rootfs": "/global-root"}}, "bwrap"),
        ({}, {}, "bare"),
    ],
)
def test_explicit_node_backend_shadows_global_default(
    node: dict, top_level: dict, expected: str,
) -> None:
    config = build_config(_raw(node, **top_level))
    backend, backend_config = config.get_worker_backend(
        config.node_types["worker"])
    assert backend == expected
    assert (backend_config is None) == (expected == "bare")


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({}, "missing required field 'rootfs'"),
        ({"rootfs": "relative"}, "must be an absolute path"),
        ({"rootfs": "/rootfs", "worker_name": "bad/name"}, "must contain only"),
        ({"rootfs": "/rootfs", "unknown": True}, "unknown field"),
        ({
            "rootfs": "/rootfs",
            "read_only_binds": {"relative": "/inside"},
        }, "paths must be absolute"),
        ({"rootfs": "/rootfs", "environment": {"COUNT": 2}},
         "must map string names"),
        ({"rootfs": "/rootfs", "environment_allowlist": ["BAD-NAME"]},
         "valid environment variable names"),
    ],
)
def test_bwrap_config_validation(section: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        build_config(_raw({"bwrap": section}))


class _RecordingSSH:
    ip = "worker.example"

    def __init__(self):
        self.commands: list[str] = []

    def run(self, command, **kwargs):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


class _LocalSSH:
    ip = "localhost"

    def run(self, command, timeout=300, check=True, **kwargs):
        result = subprocess.run(
            ["bash", "--login", "-c", command], capture_output=True,
            text=True, timeout=timeout, check=False)
        if check and result.returncode:
            raise RuntimeError(result.stderr)
        return result

    def run_script(self, commands, timeout=600, check=True):
        result = subprocess.run(
            ["bash", "--login"], input="set -e\n" + "\n".join(commands),
            capture_output=True, text=True, timeout=timeout, check=False)
        if check and result.returncode:
            raise RuntimeError(result.stderr)
        return result


def test_bwrap_argv_has_persistent_namespace_and_allowlist(tmp_path: Path) -> None:
    config = BwrapConfig(
        rootfs="/rootfs",
        worker_name="worker-3",
        state_dir=str(tmp_path),
        read_only_binds={"/host/tools": "/tools"},
        read_write_binds={"/host/work": "/workspace"},
        environment={"HOME": "/home/ray", "SECRET": "remove-me"},
        environment_allowlist=["SSH_AUTH_SOCK"],
        unset_environment=["SECRET"],
        working_dir="/workspace",
    )
    manager = BubblewrapManager(_RecordingSSH(), config)
    command = manager._launch_command()

    assert "--unshare-user" in command
    assert "--unshare-user-try" not in command
    assert "--unshare-pid" in command
    assert "--unshare-ipc" in command
    assert "--unshare-uts" in command
    assert "--ro-bind /rootfs /" in command
    assert "--ro-bind /host/tools /tools" in command
    assert "--bind /host/work /workspace" in command
    assert "--clearenv" in command
    assert "--setenv HOME /home/ray" in command
    assert "remove-me" not in command
    assert '--setenv SSH_AUTH_SOCK "${SSH_AUTH_SOCK-}"' in command
    assert "--chdir /workspace" in command


def test_unset_environment_wins_over_host_allowlist(tmp_path: Path) -> None:
    manager = BubblewrapManager(
        _RecordingSSH(),
        BwrapConfig(
            rootfs="/rootfs",
            state_dir=str(tmp_path),
            environment_allowlist=["TOKEN", "SSH_AUTH_SOCK"],
            unset_environment=["TOKEN"],
        ),
    )
    command = manager._launch_command()
    assert "TOKEN" not in command
    assert "SSH_AUTH_SOCK" in command


def test_worker_suffix_is_deterministic() -> None:
    config = BwrapConfig(rootfs="/rootfs", worker_name="circt")
    assert BubblewrapManager.for_worker(config, 7).worker_name == "circt-7"
    assert config.worker_name == "circt"


def test_exec_script_submits_atomic_base64_request(tmp_path: Path) -> None:
    ssh = _RecordingSSH()
    manager = BubblewrapManager(
        ssh,
        BwrapConfig(rootfs="/rootfs", state_dir=str(tmp_path)),
    )
    manager.exec_script(["export A=1", "printf '%s' \"$A\""], timeout=12)

    command = ssh.commands[-1]
    assert "base64 -d" in command
    assert "mv " in command
    assert "bubblewrap command timeout" in command
    payloads = []
    for word in command.split():
        try:
            decoded = base64.b64decode(word, validate=True).decode()
        except (ValueError, UnicodeDecodeError):
            continue
        if "export A=1" in decoded:
            payloads.append(decoded)
    assert payloads == ["set -e\nexport A=1\nprintf '%s' \"$A\"\n"]


def test_active_check_includes_linux_process_start_time(tmp_path: Path) -> None:
    manager = BubblewrapManager(
        _RecordingSSH(),
        BwrapConfig(rootfs="/rootfs", state_dir=str(tmp_path)),
    )
    active = manager._active_test()
    assert "awk '{print $22}' /proc/$p/stat" in active
    assert 'test "$actual" = "$expected"' in active


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_real_bwrap_exec_and_stop_cleans_child(tmp_path: Path) -> None:
    child_name = f"chia-bwrap-child-{uuid.uuid4().hex}"
    manager = BubblewrapManager(
        _LocalSSH(),
        BwrapConfig(
            rootfs="/",
            worker_name=f"live-{uuid.uuid4().hex}",
            state_dir=str(tmp_path),
            working_dir="/tmp",
            environment={"HOME": "/tmp", "PATH": "/usr/bin:/bin"},
        ),
    )
    try:
        try:
            manager.setup_worker()
        except RuntimeError as exc:
            if "namespace" in str(exc).lower() or "operation not permitted" in str(exc).lower():
                pytest.skip(f"user namespaces unavailable: {exc}")
            raise
        state = Path(manager.state_dir)
        assert (state / "ready").read_text().strip() == manager.config.worker_name
        outer_pid = int((state / "pid").read_text())
        outer_start = (state / "start_time").read_text().strip()

        result = manager.exec_script([
            "test -d /run/chia-bwrap",
            "printf namespace-ok",
            f"bash -c 'exec -a {child_name} sleep 30' >/dev/null 2>&1 &",
        ])
        assert result.stdout == "namespace-ok"
        assert subprocess.run(
            ["pgrep", "-f", child_name], capture_output=True, check=False,
        ).returncode == 0

        manager.stop_worker()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current_start = None
            try:
                current_start = Path(f"/proc/{outer_pid}/stat").read_text().split()[21]
            except (FileNotFoundError, IndexError):
                pass
            child_alive = subprocess.run(
                ["pgrep", "-f", child_name], capture_output=True, check=False,
            ).returncode == 0
            if current_start != outer_start and not child_alive:
                break
            time.sleep(0.05)
        assert current_start != outer_start
        assert not child_alive
        assert not (state / "ready").exists()
    finally:
        manager.stop_worker()


def test_setup_cleans_matching_partial_process_before_launch(monkeypatch) -> None:
    manager = BubblewrapManager(
        _RecordingSSH(), BwrapConfig(rootfs="/rootfs"))
    checks = iter([1, 0])  # inactive without ready; PID/start identity is live
    stopped = []
    monkeypatch.setattr(manager, "_check_binary", lambda *_: None)
    monkeypatch.setattr(manager, "_inspect_image", lambda: None)
    monkeypatch.setattr(manager, "_prepare_rootfs", lambda: None)

    def run(command, **kwargs):
        manager.ssh.commands.append(command)
        if command == manager._active_test():
            return subprocess.CompletedProcess(command, next(checks), "", "")
        if command == manager._pid_identity_test():
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(manager.ssh, "run", run)
    monkeypatch.setattr(manager, "stop_worker", lambda: stopped.append(True))
    monkeypatch.setattr(
        manager.ssh, "run_script", lambda *args, **kwargs: None, raising=False)
    manager.setup_worker()
    assert stopped == [True]


class _WorkerSSH:
    ip = "host-a"

    def wait_for_ssh(self):
        pass

    def run_commands(self, commands):
        pass

    def run_script(self, commands, **kwargs):
        raise AssertionError("worker script escaped the bubblewrap manager")


class _FakeBwrapManager:
    instances = []

    def __init__(self, ssh, config):
        self.ssh = ssh
        self.config = config
        self.calls = []
        self.instances.append(self)

    @staticmethod
    def for_worker(config, worker_index):
        return BubblewrapManager.for_worker(config, worker_index)

    def setup_worker(self):
        self.calls.append(("setup", None))

    def exec_script(self, commands, **kwargs):
        self.calls.append(("exec", commands))
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def stop_worker(self):
        self.calls.append(("stop", None))


def test_node_setup_routes_complete_worker_script_through_bwrap(monkeypatch) -> None:
    config = build_config(_raw({
        "bwrap": {"rootfs": "/rootfs", "worker_name": "logical"},
        "compatible_ips": ["host-b"],
        "worker_env_commands": ["export WORKER_ENV=1"],
        "worker_setup_commands": ["echo setup"],
    }))
    config.worker_start_ray_commands = [
        "ray stop", "ray start --address=$RAY_HEAD_IP:6379"]
    assignment = NodeAssignment(
        "host-b", config.node_types["worker"], {"test": 1}, worker_index=4)
    _FakeBwrapManager.instances.clear()
    monkeypatch.setattr(node_setup, "_make_ssh", lambda *_: _WorkerSSH())
    monkeypatch.setattr(node_setup, "_rsync_file_mounts", lambda *_: None)
    monkeypatch.setattr(node_setup, "BubblewrapManager", _FakeBwrapManager)

    node_setup.setup_worker_node(config, assignment, skip_ray_stop=True)

    manager = _FakeBwrapManager.instances[0]
    assert manager.config.worker_name == "logical-4"
    assert manager.calls[0] == ("setup", None)
    assert manager.calls[1][0] == "exec"
    script = manager.calls[1][1]
    assert "export WORKER_ENV=1" in script
    assert "echo setup" in script
    assert "ray stop" in script  # safe because bwrap owns a PID namespace
    assert any(command.startswith("ray start") for command in script)


def test_unscoped_teardown_stops_bwrap_namespace(monkeypatch) -> None:
    config = build_config(_raw({
        "bwrap": {"rootfs": "/rootfs", "worker_name": "logical"},
    }))
    config.scoped_teardown = False
    assignment = NodeAssignment(
        "host-a", config.node_types["worker"], {"test": 1}, worker_index=2)
    _FakeBwrapManager.instances.clear()
    monkeypatch.setattr(node_setup, "_make_ssh", lambda *_: _WorkerSSH())
    monkeypatch.setattr(node_setup, "BubblewrapManager", _FakeBwrapManager)

    node_setup.tear_down_worker_node(config, assignment)

    manager = _FakeBwrapManager.instances[0]
    assert manager.config.worker_name == "logical-2"
    assert manager.calls == [("exec", ["ray stop"]), ("stop", None)]
