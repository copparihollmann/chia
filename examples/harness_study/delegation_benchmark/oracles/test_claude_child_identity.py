import json
import time

from chia.models.claude import ClaudeCodeLLM
from chia.trace.profile_events import ProfileContext


def test_child_lifecycle_uses_task_id_without_leaking_prompt(monkeypatch):
    events = []

    class Profiler:
        enabled = True

        @staticmethod
        def profile_context():
            return ProfileContext(call_id="call", agent_id="lead")

        @staticmethod
        def log_profile_event(event):
            events.append(event)

    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: Profiler())
    llm = ClaudeCodeLLM(model="sonnet")
    now = time.monotonic()
    llm._record_child_profiles([
        (now, json.dumps({"type": "system", "subtype": "task_started",
                          "task_id": "child-one", "subagent_type": "explorer",
                          "spawn_depth": 1, "prompt": "a private prompt"})),
        (now + 0.2, json.dumps({"type": "system", "subtype": "task_notification",
                                "task_id": "child-one", "status": "completed"})),
    ])
    assert [event["type"] for event in events] == ["agent_start", "agent_end"]
    assert all(event["agent_id"] == "claude:child-one" for event in events)
    assert all(event["parent_agent_id"] == "lead" for event in events)
    assert events[1]["status"] == "completed"
    assert "private prompt" not in json.dumps(events)
