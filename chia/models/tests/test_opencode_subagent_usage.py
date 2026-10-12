"""OpenCode child sessions must be counted once or marked incomplete."""

from chia.models.opencode import OpenCodeLLM


def _task(session_id):
    return {"type": "tool", "tool": "task", "state": {
        "status": "completed", "metadata": {"sessionId": session_id},
    }}


def _export(agent, model, parts=()):
    return {"info": {"agent": agent, "model": {"providerID": "fixture", "id": model}},
            "messages": [{"info": {"role": "assistant", "tokens": {
                "input": 10, "output": 2}}, "parts": list(parts)}]}


def test_nested_subagent_sessions_are_exported_once(monkeypatch):
    llm = OpenCodeLLM(retries=1)
    root = _export("lead", "strong", (_task("child"), _task("child")))
    exports = {"child": _export("explorer", "weak", (_task("grandchild"),)),
               "grandchild": _export("checker", "weak")}
    fetched = []

    def fetch(session_id, _env):
        fetched.append(session_id)
        return exports[session_id]

    monkeypatch.setattr(llm, "_run_export", fetch)
    children, complete = llm._collect_subagent_usage(root, {}, root_session_id="root")
    assert complete is True
    assert fetched == ["child", "grandchild"]
    assert [(item["agent"], item["parent"], item["usage"]["input_tokens"])
            for item in children] == [
                ("explorer", "root", 10), ("checker", "child", 10),
            ]


def test_missing_child_export_marks_usage_incomplete(monkeypatch):
    llm = OpenCodeLLM(retries=1)
    monkeypatch.setattr(llm, "_run_export", lambda _session_id, _env: {})
    children, complete = llm._collect_subagent_usage(
        _export("lead", "strong", (_task("missing"),)), {}, root_session_id="root",
    )
    assert children == []
    assert complete is False


def test_running_child_marks_usage_incomplete_even_when_export_exists(monkeypatch):
    llm = OpenCodeLLM(retries=1)
    running = _task("child")
    running["state"]["status"] = "running"
    monkeypatch.setattr(llm, "_run_export", lambda _session_id, _env: _export("explorer", "weak"))
    children, complete = llm._collect_subagent_usage(
        _export("lead", "strong", (running,)), {}, root_session_id="root",
    )
    assert len(children) == 1
    assert complete is False
