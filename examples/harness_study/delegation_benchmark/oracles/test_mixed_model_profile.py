from chia.base.llm_call import QueryResult
from chia.models.claude import ClaudeCodeLLM
from chia.trace.profile_events import ProfileContext


def test_multi_model_total_is_not_labeled_as_primary(monkeypatch):
    events = []

    class Profiler:
        enabled = True

        @staticmethod
        def profile_context():
            return ProfileContext(call_id="call")

        @staticmethod
        def log_profile_event(event):
            events.append(event)

    monkeypatch.setattr("chia.trace.profiler.get_profiler", lambda: Profiler())
    llm = ClaudeCodeLLM(model="sonnet", retries=1)
    llm.begin_retry_ledger()
    llm.attach_usage(
        result=QueryResult(result="ok", returncode=0, stderr="", stream_result=""),
        meta={"input_tokens": 17, "output_tokens": 5,
              "model_usage": {"sonnet": {"input_tokens": 10},
                              "fixture/weak": {"input_tokens": 7}}},
    )
    requests = [event for event in events if event["type"] == "llm_request"]
    assert len(requests) == 1
    assert (requests[0]["model"], requests[0]["provider"]) == ("mixed", "multiple")
    assert (requests[0]["input_tokens"], requests[0]["output_tokens"]) == (17, 5)
