"""Exercise the actual bwrap view used by the held-out delegation study."""

import importlib.util
import os
import shutil
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

from chia.base.sandbox import bwrap_available


SPEC = importlib.util.spec_from_file_location(
    "delegation_isolation", Path(__file__).with_name("isolation.py"))
isolation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(isolation)


@pytest.mark.skipif(not bwrap_available() or shutil.which("socat") is None
                    or shutil.which("claude") is None
                    or shutil.which("opencode") is None,
                    reason="requires working bwrap, socat, and both CLIs")
def test_agent_sees_only_snapshot_and_bridged_model_endpoint(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "visible.txt").write_text("source")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "answer.txt").write_text("private")

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(8192)
            self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with isolation.RelayBridge(task_tmp=Path(os.environ["TMPDIR"]),
                                   host_port=server.server_address[1]) as bridge:
            spec = isolation.arm_spec(workspace, bridge.socket_path, "claude")
            code = ("import pathlib,socket,urllib.request;"
                    "assert pathlib.Path('visible.txt').read_text()=='source';"
                    f"assert not pathlib.Path({str(oracle)!r}).exists();"
                    "print(urllib.request.urlopen('http://127.0.0.1:18543/healthz').read().decode());"
                    "s=socket.socket();assert s.connect_ex(('1.1.1.1',443)) != 0")
            cmd = isolation.isolated_argv(["/usr/bin/python3", "-c", code],
                                          spec, bridge.socket_path, 18543)
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=15, env=dict(os.environ, HOME=str(workspace)))
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == "OK"
            for program in ("claude", "opencode"):
                spec = isolation.arm_spec(workspace, bridge.socket_path, program)
                cmd = isolation.isolated_argv([program, "--version"], spec,
                                              bridge.socket_path, 18543)
                version = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=20, env=dict(os.environ, HOME=str(workspace)))
                assert version.returncode == 0, version.stderr
                assert version.stdout.strip()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
