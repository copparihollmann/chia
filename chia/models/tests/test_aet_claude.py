import sys
import types
from http.server import BaseHTTPRequestHandler

import pytest

from chia.models.aet_claude import AetClaudeCodeLLM


def test_receiver_lifecycle_restores_child_environment_on_failure(monkeypatch, tmp_path):
    class QuietHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    aet = types.ModuleType("aet")
    tracking = types.ModuleType("aet.tracking")
    sink = types.ModuleType("aet.tracking.otel_sink")
    sink.make_handler = lambda _path: QuietHandler
    monkeypatch.setitem(sys.modules, "aet", aet)
    monkeypatch.setitem(sys.modules, "aet.tracking", tracking)
    monkeypatch.setitem(sys.modules, "aet.tracking.otel_sink", sink)

    llm = AetClaudeCodeLLM(telemetry_dir=str(tmp_path), extra_env={"UNCHANGED": "yes"})
    with pytest.raises(RuntimeError, match="agent failed"):
        with llm._aet_receiver():
            assert llm.extra_env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
            assert llm.extra_env["OTEL_EXPORTER_OTLP_ENDPOINT"].startswith("http://127.0.0.1:")
            raise RuntimeError("agent failed")

    assert llm.extra_env == {"UNCHANGED": "yes"}
    assert len(llm.otel_captures) == 1
    assert (tmp_path / llm.otel_captures[0].rsplit("/", 1)[-1]).is_file()
