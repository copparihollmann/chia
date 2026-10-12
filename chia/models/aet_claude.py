"""Claude Code experiment that captures one OTLP stream per invocation.

This adapter demonstrates Chia's external-profiler seam.  The normal profiler
does not depend on AET; importing or starting the receiver fails open and the
Claude invocation continues without telemetry.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from chia.models.claude import ClaudeCodeLLM

logger = logging.getLogger(__name__)


class AetClaudeCodeLLM(ClaudeCodeLLM):
    """Capture Claude OTel logs with a worker-local ephemeral AET receiver."""

    def __init__(self, *args, telemetry_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry_dir = telemetry_dir
        self._otel_captures: list[str] = []

    @property
    def otel_captures(self) -> tuple[str, ...]:
        """Files produced by completed invocations, suitable for ``aet import``."""
        return tuple(self._otel_captures)

    @contextmanager
    def _aet_receiver(self) -> Iterator[None]:
        if not self.telemetry_dir:
            yield
            return
        try:
            from aet.tracking.otel_sink import make_handler

            out_dir = Path(self.telemetry_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"claude_otel_{uuid4().hex}.jsonl"
            out.touch()
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(out)))
        except Exception as exc:
            logger.debug("AET receiver unavailable; continuing without it: %s", exc)
            yield
            return

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = int(server.server_address[1])
        previous = dict(self.extra_env)
        self.extra_env.update({
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_METRICS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{port}",
            "OTEL_LOGS_EXPORT_INTERVAL": "250",
            "OTEL_METRIC_EXPORT_INTERVAL": "250",
        })
        try:
            yield
        finally:
            # shutdown waits for request handlers, providing the receiver's
            # flush boundary even when the agent call raises.
            self.extra_env.clear()
            self.extra_env.update(previous)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            self._otel_captures.append(str(out))

    def _run_claude(self, user_message, tools=None):
        with self._aet_receiver():
            return super()._run_claude(user_message, tools)

    def _run_claude_streaming(self, user_message, tools=None):
        with self._aet_receiver():
            return super()._run_claude_streaming(user_message, tools)
