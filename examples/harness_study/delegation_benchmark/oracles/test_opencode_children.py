from chia.models.opencode import OpenCodeLLM


def _export(agent, model, children=()):
    parts = [{"type": "tool", "tool": "task", "state": {
        "status": "completed", "metadata": {"sessionId": child},
    }} for child in children]
    return {"info": {"agent": agent, "model": {"providerID": "fixture", "id": model}},
            "messages": [{"info": {"role": "assistant",
                                   "tokens": {"input": 10, "output": 2}},
                          "parts": parts}]}


def test_nested_child_exports_are_followed_once(monkeypatch):
    llm = OpenCodeLLM(retries=1)
    exports = {
        "child": _export("explorer", "weak", ("grandchild",)),
        "grandchild": _export("checker", "weak"),
    }
    fetched = []

    def fetch(session_id, _env):
        fetched.append(session_id)
        return exports[session_id]

    monkeypatch.setattr(llm, "_run_export", fetch)
    children, complete = llm._collect_subagent_usage(
        _export("lead", "strong", ("child", "child")), {}, root_session_id="root",
    )
    assert complete is True
    assert fetched == ["child", "grandchild"]
    assert [(item["agent"], item["usage"]["input_tokens"])
            for item in children] == [("explorer", 10), ("checker", 10)]


def test_missing_child_export_is_not_marked_complete(monkeypatch):
    llm = OpenCodeLLM(retries=1)
    monkeypatch.setattr(llm, "_run_export", lambda _session_id, _env: {})
    children, complete = llm._collect_subagent_usage(
        _export("lead", "strong", ("lost",)), {}, root_session_id="root",
    )
    assert children == []
    assert complete is False
